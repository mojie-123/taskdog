# 为夹爪手指 link7/link8 补写 Mesh 碰撞体

## 背景：问题是什么

在 `M20_Piper_lightweight.urdf` 里，机械臂大部分 link 的碰撞体被替换成了长方体/圆柱体等
primitive，以节省物理计算开销。但夹爪手指（link7、link8）对夹取精度要求高，希望保留原始
STL mesh 作为碰撞体。

因此我们在 URDF 里把 link7/link8 的 collision 改回了 mesh 引用：

```xml
<collision>
  <origin xyz="0 0 0" rpy="0 0 0" />
  <geometry>
    <mesh filename=".../meshes/link7.STL" />
  </geometry>
</collision>
```

然而运行 `convert_m20_piper_urdf.py`（URDF→USD 转换）后，仿真里两根手指**完全没有碰撞体**。

---

## 根本原因：UrdfConverter 无法处理 mesh collision

Isaac Sim 的 `UrdfConverter` 在内部调用 `URDFParseFile` + `URDFImportRobot` 命令。
`collider_type` 参数（`convex_hull` / `convex_decomposition`）**只对 visual→collision
转换路径有效**，对 `<collision><geometry><mesh>` 类型的碰撞体无法处理：
URDF Importer 会为它创建一个占位的空 Xform prim，但不会写入任何几何数据或 CollisionAPI。

---

## USD 文件结构

URDF→USD 转换后，`M20_Piper.usd` 是入口文件，通过 sublayer 引用三个子文件：

| 文件 | 作用 |
|---|---|
| `M20_Piper_base.usd` | 存储所有几何体形状（mesh 顶点数组、primitive 尺寸等）|
| `M20_Piper_physics.usd` | 存储物理属性（刚体质量、关节参数、碰撞声明等）|
| `M20_Piper_sensor.usd` | 存储传感器（lidar 等）|

每个 link 的碰撞体几何在 base.usd 里存放于 `/colliders/<link_name>/` 下，
对应的物理声明在 physics.usd 里存放于 `/M20_Piper/<link_name>/collisions/` 下。

### 正常 link（gripper_base）的碰撞体结构

```
[base.usd]
/colliders/gripper_base/simplified_collision   Xform  (translate=位置, scale=尺寸)
    /box                                       Cube   apiSchemas=[PhysicsCollisionAPI]
                                                      physics:collisionEnabled = True
```

### link7/link8 转换后的现状（问题所在）

```
[base.usd]
/colliders/link7/link7   Xform  (空！只有 xformOp，无几何子 prim)

[physics.usd]
/M20_Piper/link7/collisions/link7   Xform  (apiSchemas = None，无碰撞声明)
```

缺少两样东西：
1. **几何数据**（base.usd 里缺）
2. **物理碰撞声明**（physics.usd 里缺）

---

## 解决方案：post-process 脚本补写

### 几何数据从哪来

base.usd 里已经存在 link7 的完整 3D 网格数据，位于 visuals（视觉）那边：

```
/visuals/link7/link7/node_STL_BINARY_/mesh   Mesh
    points:            Vec3fArray  (6258 个顶点坐标)
    faceVertexCounts:  IntArray    (2086 个三角面，每面3顶点)
    faceVertexIndices: IntArray    (6258 个顶点索引)
    normals:           Vec3fArray  (法向量)
```

STL 被 URDF Importer 转换时，几何数据以这三个数组写入了 visual mesh prim。
碰撞体需要同样的数据——直接从 visuals 读取，写入 colliders。

### 第一步：修改 base.usd（补几何数据）

```
修改前：/colliders/link7/link7   Xform（空容器）
修改后：/colliders/link7/link7   Mesh
        points / faceVertexCounts / faceVertexIndices / normals / extent
        数据来源：/visuals/link7/link7/node_STL_BINARY_/mesh
```

### 第二步：修改 physics.usd（补碰撞声明）

```
修改前：/M20_Piper/link7/collisions/link7   Xform  apiSchemas=None
修改后：/M20_Piper/link7/collisions/link7   Xform
            apiSchemas = [PhysicsCollisionAPI]
            physics:collisionEnabled = True
            physics:approximation = "convexDecomposition"
```

link8 完全对称，做同样操作。

### 为什么选 convexDecomposition？

PhysX 要求碰撞体必须是凸体，对于 Mesh 类型的 prim 需要声明近似方法：

- **`convexHull`**：对整个 mesh 算一个凸包。L 形手指的凸包会把两段之间的空隙填满，
  体积偏大，导致手指还没接触到香蕉就产生排斥力。
- **`convexDecomposition`**：把 mesh 分解成多个小凸体拼合，贴合度远高于单一凸包，
  接触点更准确。

---

## 使用的 pxr API

| 操作 | API |
|---|---|
| 打开 USD 文件 | `Usd.Stage.Open(path)` |
| 获取 prim | `stage.GetPrimAtPath("/colliders/link7/link7")` |
| 读取属性值 | `prim.GetAttribute("points").Get()` |
| 修改 prim 类型 | `layer.GetPrimAtPath(sdf_path).typeName = "Mesh"` |
| 创建属性 | `prim.CreateAttribute(name, type_name)` |
| 写入属性值 | `attr.Set(value)` |
| 添加 apiSchemas | `prim.SetMetadata("apiSchemas", Sdf.TokenListOp.CreateExplicit([...]))` |
| 保存 | `stage.GetRootLayer().Save()` |

---

## 脚本：patch_gripper_finger_collision.py

**位置**：`custom_envs/scripts/assets/patch_gripper_finger_collision.py`

### 运行方式

脚本不需要启动 Isaac Sim，只需要 Isaac Sim 内置的 pxr 库（通过环境变量注入）：

```bash
conda activate env_isaaclab

PXR_PATH=$CONDA_PREFIX/lib/python3.11/site-packages/isaacsim/extscache/omni.usd.libs-1.0.1+69cbf6ad.lx64.r.cp311

LD_LIBRARY_PATH=$PXR_PATH/bin:$CONDA_PREFIX/lib:$LD_LIBRARY_PATH \
PYTHONPATH=$PXR_PATH \
python3 custom_envs/scripts/assets/patch_gripper_finger_collision.py
```

### 参数

| 参数 | 说明 | 默认值 |
|---|---|---|
| `--dry-run` | 只打印将要做的操作，不写入任何文件 | 关闭 |
| `--usd-dir PATH` | USD 配置文件目录 | `custom_envs/assets/m20_piper_single/configuration` |

### 运行输出示例

```
[备份] M20_Piper_base.usd -> M20_Piper_base.usd.bak
[备份] M20_Piper_physics.usd -> M20_Piper_physics.usd.bak

[base.usd] 打开 .../M20_Piper_base.usd
  [link7] src=/visuals/link7/link7/node_STL_BINARY_/mesh
  [link7] dst=/colliders/link7/link7  当前 typeName=Xform
  [link7] typeName -> Mesh
  [link7] 复制属性 points
  [link7] 复制属性 faceVertexCounts
  [link7] 复制属性 faceVertexIndices
  [link7] 复制属性 normals
  [link7] 复制属性 extent
  [link7] purpose -> guide
  [link8] ...
[base.usd] 已保存

[physics.usd] 打开 .../M20_Piper_physics.usd
  [link7] 当前 apiSchemas: None
  [link7] apiSchemas -> ['PhysicsCollisionAPI']
  [link7] physics:collisionEnabled -> True
  [link7] physics:approximation -> convexDecomposition
  [link8] ...
[physics.usd] 已保存

=== 验证 ===
  M20_Piper_base.usd:    Mesh                           [OK]
  M20_Piper_physics.usd: PhysicsCollisionAPI            [OK]
  M20_Piper_physics.usd: convexDecomposition            [OK]

完成。请重启 Isaac Sim 使修改生效。
```

---

## 完整工作流程

```
修改 URDF 或 build 脚本
        ↓
python3 convert_m20_piper_urdf.py --headless
        ↓  （生成新的 base.usd / physics.usd，link7/link8 碰撞体为空）
LD_LIBRARY_PATH=... PYTHONPATH=... \
python3 patch_gripper_finger_collision.py
        ↓  （补写 Mesh 几何数据 + PhysicsCollisionAPI）
重启 Isaac Sim，验证碰撞体形状
```

> **注意**：每次重新运行 `convert_m20_piper_urdf.py` 后，都必须重新运行 patch 脚本，
> 因为 convert 会覆盖 base.usd 和 physics.usd。

---

## 注意事项

1. **备份机制**：脚本运行时自动创建 `.usd.bak` 备份。再次运行前若备份已存在则跳过，
   不会覆盖上一次的备份。如需强制重新备份，先手动删除 `.usd.bak` 文件。

2. **验证方式**：脚本末尾调用 `strings` 做子串匹配。USDC 格式使用 token 前缀压缩，
   `strings` 可能匹配不到部分 token。若出现 NOT FOUND 警告，可在 Isaac Sim Script Editor
   里用以下代码直接验证：
   ```python
   stage = omni.usd.get_context().get_stage()
   p = stage.GetPrimAtPath("/M20_Piper/link7/collisions/link7")
   print(p.GetMetadata("apiSchemas"))
   p2 = stage.GetPrimAtPath("/colliders/link7/link7")
   print(p2.GetTypeName())  # 应为 Mesh
   ```

3. **首次加载时间**：`convexDecomposition` 需要 PhysX 在首次加载时做预计算（约几秒），
   之后会被缓存。

4. **显存影响**：Mesh 碰撞体的 CPU 侧数据（顶点数组）约 1~2 MB，
   PhysX GPU 端使用 convexDecomposition 的结果（多个小凸体），显存增加极小。

5. **为什么不合并进 convert 脚本**：`convert_m20_piper_urdf.py` 需要 Isaac Sim 完整环境，
   而 patch 脚本只需要轻量的 pxr 库，两者分开便于单独复用和调试。

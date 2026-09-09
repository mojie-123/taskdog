# 基本情况
## 工作区中与香蕉有关的文件

### USD / USDA 资产文件（5个）

| 文件 | 大小 | 说明 |
|------|------|------|
| `custom_envs/objects/011_banana.usd` | 452K | **主文件**（USDC 二进制），当前实际使用的香蕉 USD 资产 |
| `custom_envs/objects/011_banana.usda` | 1.5M | **展开的人类可读版本**（USDA 文本），由 Isaac Sim 从 011_banana.usd 组合 Stage 后导出，内容与 .usd 等价，多了编辑器视口相机等 meta 信息 |
| `custom_envs/objects/011_banana_new.usd` | 452K | 与 011_banana.usd 内容几乎一致的副本（strings 输出相同结构） |
| `custom_envs/objects/011_banana_backup.usd` | 456K | **备份版本**，结构略有不同（有 `material:binding` 路径），是修改前的历史快照 |
| `custom_envs/objects/011_banana.usda` | （同上） | USDA 是 .usd 的展开导出，并非独立制作 |

### 贴图文件（1个）

| 文件 | 大小 | 规格 |
|------|------|------|
| `custom_envs/objects/Materials/Textures/011_banana_COLOR.png` | 5.9M | **4096×4096 RGBA，8-bit PNG**（非交错）—— 香蕉的漫反射/颜色贴图 |

### 引用香蕉 USD 的 Python 脚本（间接引用，不含几何数据）

- `custom_envs/utils/target_spawner.py`
- `custom_envs/utils/arm_ik.py`
- `custom_envs/tasks/deeprobotics_m20_pro/single_piper_env_cfg.py`
- `custom_envs/tasks/deeprobotics_m20_pro/lidar_flat_env_cfg.py`
- `custom_envs/scripts/navigation/*.py`（navigate_to_goal, teleop_mapping, view_3d）
- `grasp_vis.py`
- `docs/usd_fix.md` / `add_navigation.md`（文档）

---

## 贴图是怎么贴到香蕉上的？

### 完整链路（来自 011_banana.usda 的真实内容）

贴图绑定通过 **USD 的材质系统 + OmniPBR MDL 着色器**实现，分三层：

```
/Root                           ← 根 Xform（kind=component）
  └── /Root/Looks              ← Scope，材质库
        └── /Root/Looks/OmniPBR    ← Material prim
              outputs:mdl:surface  → /Root/Looks/OmniPBR/Shader.outputs:out
              outputs:mdl:displacement → 同上
              outputs:mdl:volume   → 同上
              └── /Root/Looks/OmniPBR/Shader    ← Shader prim
                    info:implementationSource = "sourceAsset"
                    info:mdl:sourceAsset = @OmniPBR.mdl@          ← 使用 OmniPBR 着色器
                    info:mdl:sourceAsset:subIdentifier = "OmniPBR"
                    inputs:diffuse_texture = 
                      @/home/mojie/taskdog/custom_envs/objects/
                        Materials/Textures/011_banana_COLOR.png@  ← ★ 贴图绑定在这里
                          (colorSpace = "auto")                   ← 自动颜色空间

  └── /Root/_11_banana         ← Mesh prim（香蕉几何体）
        apiSchemas = [
          "MaterialBindingAPI",           ← ★ 声明材质绑定能力
          "PhysicsRigidBodyAPI",          ← 刚体物理
          "PhysxRigidBodyAPI",
          "PhysicsCollisionAPI",          ← 碰撞体
          "PhysxCollisionAPI",
          "PhysicsMeshCollisionAPI",
          "PhysicsMassAPI",               ← 质量
          "PhysxConvexDecompositionCollisionAPI"  ← 凸分解碰撞
        ]
        material:binding = </Root/Looks/OmniPBR>   ← ★ Mesh 指向上面的 Material
        （points/faceVertexCounts/faceVertexIndices：完整三角网格数据）
```

### 贴图路径说明

- 贴图使用的是**绝对路径** `/home/mojie/taskdog/custom_envs/objects/Materials/Textures/011_banana_COLOR.png`
- 这是 USDA 展开导出时固化的路径；原始 .usd 文件中很可能是相对路径 `./Materials/Textures/011_banana_COLOR.png`（由 strings 输出 `_COLOR.png` 片段可推断）
- 若要移植到其他机器，需要将绝对路径改为相对路径，或使用 USD 的资产路径解析器

### 贴图只用了哪个通道？

| OmniPBR 参数 | 值 | 说明 |
|---|---|---|
| `inputs:diffuse_texture` | `011_banana_COLOR.png` | 漫反射颜色（RGBA 4通道均可用） |
| 其他参数（法线贴图、粗糙度等） | **未设置** | 全部使用 OmniPBR 默认值 |

这意味着香蕉只用了一张颜色贴图（4096×4096 RGBA）贴在漫反射通道上，没有法线贴图或高光贴图，是最简单的单贴图 PBR 设置。

---

### 一句话总结

> **`011_banana.usd`** 是主资产，包含香蕉的三角网格 + OmniPBR 材质定义；**`Materials/Textures/011_banana_COLOR.png`** 是唯一一张贴图（4K RGBA），通过 `Shader.inputs:diffuse_texture` 绑定到材质，再由 Mesh 上的 `MaterialBindingAPI` 将材质连接到几何体，最终完成贴图显示。

---
# 整个香蕉的贴图过程都是自动完成的吗？讲讲脚本中是怎么通过找到香蕉的.usd然后配置然后放到环境中的（举具体的脚本例子说明这个流程）

## 贴图是全自动的吗？

**贴图本身是全自动的**——颜色贴图路径已经写死在 `011_banana.usd` 内部，Isaac Sim 加载 USD 时自动解析并渲染，脚本完全不需要手动处理贴图。但把香蕉放到环境里这件事是手动用代码配置的，下面逐步拆解。

---

## 完整流程：从 .usd 文件到仿真环境

以 `lidar_flat_env_cfg.py` 为主例，流程分五个步骤。

### Step 1：定位 .usd 文件路径

```python
# lidar_flat_env_cfg.py 第16-18行
BANANA_USD = os.path.join(
    os.path.dirname(__file__), "..", "..", "objects", "011_banana.usd"
)
```

用 `os.path.dirname(__file__)` 得到当前脚本所在目录，再用相对路径拼接，最终指向 `/home/mojie/taskdog/custom_envs/objects/011_banana.usd`。这只是一个普通字符串，不做任何 USD 解析，只是记录路径。

### Step 2：构建 RigidObjectCfg（配置对象，不是真实 prim）

```python
# lidar_flat_env_cfg.py 第22-49行
def _banana_cfg(pos=(4.9, 5.0, 0.75), rot=(1.0, 0.0, 0.0, 0.0)):
    return RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/banana",
        spawn=sim_utils.UsdFileCfg(
            usd_path=BANANA_USD,
            scale=(1, 1, 1),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                kinematic_enabled=False,  # 受重力，会掉到桌子上
                linear_damping=2.0,
                angular_damping=4.0,
                max_depenetration_velocity=0.5,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.5),
            collision_props=sim_utils.CollisionPropertiesCfg(
                contact_offset=0.01,
                rest_offset=0.0,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=pos, rot=rot),
    )
```

这一步只是创建一个纯 Python 配置对象（dataclass），不打开 USD 文件，不创建任何 prim。关键点：

- `UsdFileCfg` 里的 `rigid_props`/`mass_props`/`collision_props` 会在 spawn 时覆盖 USD 文件内已有的物理属性
- `scale=(1,1,1)` 说明香蕉 USD 单位本身已是米制（`metersPerUnit=1`），无需缩放
- `pos=(4.9, 5.0, 0.75)` 是初始位置：桌子中心 x=5.0，香蕉稍靠近机械臂一侧，z=0.75m 悬在桌面上方，依靠重力落到桌面

### Step 3：注册到 Scene（声明期，仍不创建 prim）

```python
# lidar_flat_env_cfg.py 第72-77行
@configclass
class TaskdogSceneCfg(MySceneCfg):
    table:  RigidObjectCfg = _table_cfg()
    banana: RigidObjectCfg = _banana_cfg()
```

`TaskdogSceneCfg` 继承自 `MySceneCfg`（IsaacLab 的 `InteractiveSceneCfg`），只要在类体中声明一个 `RigidObjectCfg` 类型的字段，IsaacLab 就会在场景初始化时自动识别并 spawn 它。字段名 `banana` 就是这个对象在代码中的引用键名。

`SinglePiperEnvCfg` 继承这个类后，`scene.banana` 就已经存在，所以注释说：

```python
# single_piper_env_cfg.py 第39-41行
# table and banana are already provided by TaskdogSceneCfg
# no need to add them again here.
```

### Step 4：IsaacLab 运行时 spawn（真正创建 USD prim 的时刻）

当 `env.reset()` 或仿真启动时，IsaacLab 的 `InteractiveScene` 遍历所有已注册的 `RigidObjectCfg` 字段，底层执行：

```
spawn_from_usd(
    prim_path = "/World/envs/env_0/banana",  # {ENV_REGEX_NS} 解析为 env_0
    cfg       = UsdFileCfg 对象
)
  1. stage.DefinePrim("/World/envs/env_0/banana")   -- 创建空 prim
  2. prim.GetReferences().AddReference(BANANA_USD)  -- 引用 .usd 文件
     此时 USD 几何/材质/贴图全部通过引用加载，自动出现
  3. 应用 RigidBodyPropertiesCfg -> 写入 PhysxRigidBodyAPI 属性
  4. 应用 MassPropertiesCfg     -> 写入 physics:mass = 0.5
  5. 应用 CollisionPropertiesCfg -> 写入 contact_offset 等
```

**贴图在第 2 步完全自动生效**：`AddReference()` 让新 prim 引用 `011_banana.usd`，而该文件内部已有完整的 `Material -> OmniPBR Shader -> diffuse_texture -> 011_banana_COLOR.png` 链，USD 运行时自动解析，无需任何额外代码。

### Step 5：配置 LiDAR 感知香蕉

```python
# lidar_flat_env_cfg.py 第101-113行
self.scene.mid360_lidar.mesh_prim_paths = [
    "/World/ground",
    MultiMeshRayCasterCfg.RaycastTargetCfg(
        prim_expr="{ENV_REGEX_NS}/Shop_Table",
        track_mesh_transforms=True,
        is_shared=True,
    ),
    MultiMeshRayCasterCfg.RaycastTargetCfg(
        prim_expr="{ENV_REGEX_NS}/banana",   # 香蕉作为 LiDAR raycasting 目标
        track_mesh_transforms=True,          # 追踪香蕉移动
        is_shared=True,
    ),
]
```

这是导航任务专属的，让 Mid-360 LiDAR 的光线能打到香蕉网格并返回点云，`track_mesh_transforms=True` 使得香蕉在物理仿真中移动后 raycasting 也能跟上。

---

## 总结：各部分职责

| 步骤 | 代码位置 | 做了什么 | 是否自动 |
|------|----------|----------|----------|
| 贴图封装进 USD | `011_banana.usd` 内 | OmniPBR Shader 绑定 PNG | 完全自动 |
| 路径定位 | `lidar_flat_env_cfg.py:16` | os.path.join 拼路径 | 手动 |
| 物理参数配置 | `_banana_cfg()` | 质量/阻尼/碰撞参数 | 手动 |
| 初始位置 | `InitialStateCfg` | (4.9, 5.0, 0.75) | 手动 |
| 注册到场景 | `TaskdogSceneCfg.banana` 字段 | 字段声明后框架自动 spawn | 半自动 |
| 实际创建 prim | IsaacLab 运行时 | AddReference + 写物理 attrs | 框架自动 |
| LiDAR 感知 | `mesh_prim_paths` 配置 | 手动把香蕉加入雷达目标 | 手动 |

**一句话总结**：贴图 100% 自动（封装在 USD 里，AddReference 时自动生效）；香蕉的物理属性和位置需要手动在 `_banana_cfg()` 里写；创建 prim 和加载材质由 IsaacLab 的 `UsdFileCfg` 框架自动完成，你只需把 `RigidObjectCfg` 挂到 `SceneCfg` 的字段上即可。
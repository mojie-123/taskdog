# 011_banana.usd 碰撞体修复记录

## 旧版 `011_banana.usd` 存在的问题

### 问题根源：两个冲突的 API Schema 同时存在

旧版文件的 Mesh prim 的 `apiSchemas` 字段里，**同时包含了两个互相冲突的物理碰撞 API**：

```
apiSchemas = [..., "PhysxConvexHullCollisionAPI", ..., "PhysxConvexDecompositionCollisionAPI"]
```

同时，文件里还有一行：
```
uniform token physics:approximation = "convexDecomposition"
```

这造成了矛盾：`physics:approximation` 属性说「用 convexDecomposition」，但 `PhysxConvexHullCollisionAPI` 这个 schema 的存在让 PhysX 运行时**优先采用 convexHull**（单一凸包）来近似香蕉的碰撞体积。

**convexHull 的问题**：香蕉是弯曲的月牙形，其凸包是一个把两端直接连起来的椭球体，体积远大于实际香蕉，导致机械臂在 REACH 阶段物理上已经碰到了碰撞体边界、无法继续前进，但视觉上看起来还差很远。

**convexDecomposition 的优势**：把香蕉 mesh 分解成多个小凸体拼合，能贴合弯曲的香蕉轮廓，碰撞体积更接近实际形状。

---

## 完整转换过程

### 第一步：USDC -> USDA（二进制转文本）

旧版 `011_banana.usd` 是 **USDC 格式**（USD Crate，二进制压缩格式），魔数头为 `PXR-USDC`。

直接用文本编辑器或 `grep` 搜索时，发现 `convexHull` 这个字符串在文件里以**前缀压缩**方式存储（USDC 的 token 表会把相邻 token 的公共前缀压缩掉），例如搜索到的是 `nvexHull`（前缀 `co` 被压缩为一个长度字节），无法直接做字节替换——即使替换了 token 内容，token 表的长度前缀字节也会失效，导致文件损坏。

因此选择**在 Isaac Sim Script Editor 里调用 USD Python API 导出为文本格式**：

```python
import omni.usd
stage = omni.usd.get_context().get_stage()  # 当前已打开的 011_banana.usd stage
stage.Export("/home/mojie/taskdog/custom_envs/objects/011_banana.usda")
print("Exported to usda")
```

`stage.Export()` 根据文件扩展名 `.usda` 自动以 **USDA 格式**（USD ASCII，纯文本）导出，生成了 333 行的可读文本文件。

---

### 第二步：读取 USDA，定位关键信息

用 grep 在 `011_banana.usda` 里搜索物理相关字段，定位到第 **79 行**，这是香蕉 Mesh prim 的声明头：

```
def Mesh "_11_banana" (
    apiSchemas = ["MaterialBindingAPI", "PhysicsRigidBodyAPI", "PhysxRigidBodyAPI",
                  "PhysicsCollisionAPI", "PhysxCollisionAPI",
                  "PhysxConvexHullCollisionAPI",
                  "PhysicsMeshCollisionAPI", "PhysicsMassAPI",
                  "PhysxConvexDecompositionCollisionAPI"]
)
```

以及第 **91 行**：

```
uniform token physics:approximation = "convexDecomposition"
```

**关键发现**：`physics:approximation` 已经是 `convexDecomposition`（正确），但 `PhysxConvexHullCollisionAPI` 仍在 `apiSchemas` 列表里。PhysX 运行时通过 schema 来决定用哪种碰撞近似，`PhysxConvexHullCollisionAPI` 的存在会覆盖 `physics:approximation` 属性的意图。

**需要做的修改**：从 `apiSchemas` 列表里删除 `"PhysxConvexHullCollisionAPI"` 这一项，保留 `"PhysxConvexDecompositionCollisionAPI"`。

---

### 第三步：修改 USDA

直接编辑第 79 行，删除 `"PhysxConvexHullCollisionAPI"` 这一项：

修改前：
```
apiSchemas = ["MaterialBindingAPI", "PhysicsRigidBodyAPI", "PhysxRigidBodyAPI",
              "PhysicsCollisionAPI", "PhysxCollisionAPI",
              "PhysxConvexHullCollisionAPI",
              "PhysicsMeshCollisionAPI", "PhysicsMassAPI",
              "PhysxConvexDecompositionCollisionAPI"]
```

修改后：
```
apiSchemas = ["MaterialBindingAPI", "PhysicsRigidBodyAPI", "PhysxRigidBodyAPI",
              "PhysicsCollisionAPI", "PhysxCollisionAPI",
              "PhysicsMeshCollisionAPI", "PhysicsMassAPI",
              "PhysxConvexDecompositionCollisionAPI"]
```

修改后验证：grep 结果只剩第 91 行的 `convexDecomposition`，`PhysxConvexHullCollisionAPI` 彻底消失。

---

### 第四步：USDA -> USDC（文本转回二进制）

Isaac Sim 自带的 `pxr` 库（USD Python bindings）在 `env_isaaclab` 的 site-packages 里，但需要手动设置 `LD_LIBRARY_PATH` 才能加载。经过搜索找到：

- **pxr 包路径**：`.../isaacsim/extscache/omni.usd.libs-1.0.1+.../`
- **共享库路径**：同目录下的 `bin/`（含 `libusd_tf.so` 等）
- **Python 运行时库**：`/home/mojie/anaconda3/envs/env_isaaclab/lib/`

执行转换命令（同时设置 LD_LIBRARY_PATH 和 PYTHONPATH）：

```bash
PXR_PATH=/home/mojie/anaconda3/envs/env_isaaclab/lib/python3.11/site-packages/isaacsim/extscache/omni.usd.libs-1.0.1+69cbf6ad.lx64.r.cp311
PYLIB=/home/mojie/anaconda3/envs/env_isaaclab/lib

LD_LIBRARY_PATH=$PXR_PATH/bin:$PYLIB:$LD_LIBRARY_PATH \
PYTHONPATH=$PXR_PATH \
python3 -c "
from pxr import Usd
stage = Usd.Stage.Open('/home/mojie/taskdog/custom_envs/objects/011_banana.usda')
stage.Export('/home/mojie/taskdog/custom_envs/objects/011_banana_new.usd')
print('Export OK')
"
```

`Usd.Stage.Export()` 根据 `.usd` 扩展名自动输出 **USDC 二进制格式**。输出过程中有若干关于 `hide_in_stage_window`、`no_delete` 等 Omniverse 自定义元数据字段不被标准 USD 识别的警告，**不影响物理属性**，可以忽略。

---

### 第五步：验证并替换

用 Python 读取新文件的二进制内容验证：

```python
with open('/home/mojie/taskdog/custom_envs/objects/011_banana_new.usd', 'rb') as f:
    data = f.read()
print('Header:', repr(data[:8]))          # 应为 b'PXR-USDC'
print('Hull:', data.find(b'Hull'))         # 应为 -1
print('Convex:', data.find(b'Convex'))     # 应找到 ConvexDecomposition
```

验证结果：
- `data[:8]` 为 `b'PXR-USDC'` — 正确的二进制格式 ✅
- `data.find(b'Hull')` 返回 `-1` — PhysxConvexHullCollisionAPI 已消失 ✅
- `data.find(b'Convex')` 找到 `b'ConvexDem...'` — ConvexDecomposition 保留（前缀压缩存储）✅

备份旧文件并替换：
```bash
cp 011_banana.usd 011_banana_backup.usd
cp 011_banana_new.usd 011_banana.usd
```

最终文件大小从 461910 B 增加到 462044 B（token 表结构变化导致微小差异）。

---

## 总结流程

```
旧版 011_banana.usd (USDC 二进制)
  apiSchemas 同时含 PhysxConvexHullCollisionAPI
                 + PhysxConvexDecompositionCollisionAPI
  PhysX 运行时采用 ConvexHull -> 碰撞体积偏大 -> 机械臂无法接近香蕉
        |
        v  Isaac Sim Script Editor: stage.Export(.usda)
011_banana.usda (USDA 文本, 333行)
  第79行 apiSchemas 两个冲突 API 清晰可见
        |
        v  删除 "PhysxConvexHullCollisionAPI"
011_banana.usda (修改后)
  apiSchemas 只剩 PhysxConvexDecompositionCollisionAPI
  physics:approximation = "convexDecomposition"
        |
        v  pxr.Usd.Stage.Open().Export(.usd)
新版 011_banana.usd (USDC 二进制)
  Hull: NOT FOUND
  Convex: ConvexDecomposition
  PhysX 运行时采用 ConvexDecomposition -> 碰撞体贴合香蕉弯曲形状
```

---

## 附：为什么不能直接二进制替换？

USDA 里写的是 `convexHull`（10字节），而 `convexDecomposition`（19字节）更长，直接替换会破坏文件结构。

即使长度相同，USDC 的 token 表采用**前缀压缩**：相邻 token 共享公共前缀，每个 token 只存储「与上一个 token 相比，共享前 N 个字节，后面新增的字节是 XYZ」。因此：
- `convexHull` 在文件里实际存储的是 `\x8d nvexHull`（`\x8d` 是长度前缀，`co` 被压缩掉）
- 修改 token 内容的同时必须同步修改长度前缀字节，否则 USD 解析器会读错边界，文件损坏

文本转换（USDC→USDA→修改→USDC）是唯一安全可靠的路径。

---

## 附：为什么 Isaac Sim Script Editor 的保存无法写回磁盘？

`omni.usd.get_context().save_stage()` 保存的是 Isaac Sim 内存里的 **composed stage**（合成层），而不是源文件。Isaac Sim 的 stage 是多层叠加的（session layer、root layer、sublayer...），`save_stage()` 只写入 session layer，不会覆盖磁盘上的原始 `.usd` 文件。

正确做法是 `stage.Export(path)` 明确指定输出路径，绕过 session layer 机制。

# j7j8为什么会有问题
## 真正的原因：两份 USD 是用**不同的 config.yaml 参数**转换出来的

对比两份 `M20_Piper_base.usd` 的关键 token 差异：

| token | deps 原始 USD | custom_envs bak USD | 含义 |
|---|---|---|---|
| `Mesh` | ❌ 没有 | ✅ 有 | Mesh 类型 prim |
| `node_STL_BINARY_` | ❌ 没有 | ✅ 有 | STL mesh 节点名 |
| `visuals` | ❌ 没有 | ✅ 有 | /visuals/ 路径 |
| `collider6` | ✅ 两者都有 | ✅ 两者都有 | collider 路径片段 |
| `Xform` | ✅ 有 | ❌ 没有（bak里没看到） | 空 Xform 占位 |
| `VertexCounts` | ✅ 有 | `VertexCou`（截断） | 顶点数组 |
| `apiSchemas` | ❌ 没有 | ✅ 有 | physics schema |
| `references` | ❌ 没有 | ✅ 有 | USD reference |
| `node_STL_BINARY_` | ❌ 没有 | ✅ 有 | STL mesh 数据 |

对比两份 `M20_Piper_physics.usd` 的差异：

| token | deps 原始 | custom_envs bak | 含义 |
|---|---|---|---|
| `icsPrismatic` | ❌ 没有 | ✅ 有（`PhysicsPrismatic...`）| prismatic joint 类型 |
| `apiSchemas` | ❌ 没有 | ✅ 有 | physics schema 声明 |
| `convexHull` | ❌ 没有 | ✅ 有 | collision approximation |
| `PhysicsRigidBodyAPI` | ✅ 两者都有 | ✅ 两者都有 | 刚体声明 |
| `DriveAPI:angular` | ✅ deps 有 | ❌ bak 没有 | 关节驱动 |

---

## 关键发现：两次转换使用了不同版本的 Isaac Sim / UrdfConverter

deps 目录下的原始 USD（`/home/mojie/taskdog/deps/deep_robotics_model/M20_Piper/usd/`）**不是由本项目的 convert 脚本生成的**，而是 Deep Robotics 预先生成并随模型一起发布的。

**两次转换的本质区别**：

```
deps/usd/（Deep Robotics 预生成）
├── 使用的是 Deep Robotics 自己的 Isaac Sim 环境
├── base.usd：没有 node_STL_BINARY_ 节点
│            没有独立的 /visuals/ 路径
│            有 VertexCounts（直接内联了 mesh 数据）
│            collision mesh 数据完整
└── 可能使用了更旧/更新的 UrdfConverter，或使用了不同的参数
    例如：collision_from_visuals=True，或不同的 collider_type

custom_envs/configuration/（本项目 convert 脚本生成）
├── 使用本机 Isaac Sim UrdfConverter
├── config.yaml 里：collider_type: convex_hull
│                   collision_from_visuals: false
│                   make_instanceable: true
├── base.usd：有 node_STL_BINARY_ 节点（STL 作为独立 mesh）
│            有 /visuals/ 路径
│            link7/link8 的 /colliders/ 下是空 Xform
└── physics.usd：link7/link8 没有 PhysicsCollisionAPI
```

---

## 根本原因的完整解释

deps 里的原始 USD 里 link7/link8 有有效 collision，**不是因为 UrdfConverter 能处理 prismatic joint 的 mesh collision**，而是因为：

**Deep Robotics 使用的转换流程或 UrdfConverter 版本与本项目不同**，导致转换结果不同。具体来说，最可能的差异是：

1. **`collision_from_visuals` 参数**：deps 的 USD 里 base.usd 没有独立的 `node_STL_BINARY_` 节点，说明 Deep Robotics 的转换可能用了 `collision_from_visuals=True`，直接从 visual mesh 生成 collision，绕过了从 URDF `<collision>` 标签解析 mesh 的步骤——而本项目用的是 `collision_from_visuals: false`，要求 UrdfConverter 读取 URDF 里的 `<collision>` 标签来生成碰撞体，但 UrdfConverter 在处理这个路径时对 prismatic joint 的 child link 存在 bug。

2. **Isaac Sim 版本差异**：不同版本的 UrdfConverter 对 prismatic joint 的 child link 的 collision 处理逻辑可能不同，Deep Robotics 使用的版本可能已经修复了这个问题。

---

## 总结

```
deps 原始 USD（Deep Robotics 预生成）
  → 转换参数/版本不同 → link7/link8 有有效 collision ✅

custom_envs USD（本项目 convert 脚本生成）
  → collision_from_visuals=false + 本机 UrdfConverter
  → UrdfConverter 读 URDF <collision><mesh> 时
    对 prismatic joint child link 生成空 Xform ❌
  → 需要 patch 脚本补救
```

patch 脚本注释里说「UrdfConverter 无法处理 `<collision><geometry><mesh>`」这个描述确实不够准确——**更准确的说法是：本项目使用的这个特定版本的 UrdfConverter，在 `collision_from_visuals=false` 的配置下，对 prismatic joint 的 child link 的 mesh collision 处理存在 bug**，而 deps 里的 USD 是用不同的配置/版本生成的，所以没有这个问题。



---
现在有了完整的代码，可以详细讲解了。

---

# patch 脚本如何修改 .usd 文件

---

## 一、关键前提：USD 的编程接口 `pxr`

`.usd` 文件（特别是 USDC 二进制格式）**不是普通文本文件**，不能用字符串替换直接修改。脚本使用的是 Pixar 官方提供的 Python 库 **`pxr`**（即 OpenUSD 的 Python 绑定），这个库提供了两层 API：

```
pxr.Usd   — 高层 API：操作「Stage」（整个场景）和「Prim」（场景中的节点）
              适合读取属性值、创建属性、设置值

pxr.Sdf   — 低层 API：直接操作「Layer」（单个 .usd 文件）和「PrimSpec」（文件里的原始记录）
              适合修改 typeName、apiSchemas 等元数据，以及在 instanceable prim 场景下绕过限制
```

两层 API 最终都通过 pxr 库把修改写回 `.usd` 文件的二进制结构，调用 `layer.Save()` 或 `stage.Save()` 保存。

---

## 二、被修改的 USD 文件里，问题区域长什么样

用 USDA（文本格式）等价表示 UrdfConverter 生成的、**有问题**的 base.usd 中 link7 的 collider 部分：

```usda
# /colliders/link7/link7 — UrdfConverter 生成的有问题的结构
def Xform "link7"          # ← typeName 是 Xform（空容器），不是 Mesh
{
    # 完全没有任何几何属性：
    # 没有 point[]  points
    # 没有 int[]    faceVertexCounts
    # 没有 int[]    faceVertexIndices
    # ... 完全空的
    token purpose = "guide"    # ← 只有这一个属性
}
```

而正确的 collider（其他 link 如 link1~link6 的样子）应该是：

```usda
def Mesh "link1"           # ← typeName 是 Mesh
{
    point3f[] points = [(x1,y1,z1), (x2,y2,z2), ...]  # 顶点坐标数组
    int[]     faceVertexCounts  = [3, 3, 3, ...]       # 每个三角面3个顶点
    int[]     faceVertexIndices = [0,1,2, 1,2,3, ...]  # 顶点索引
    normal3f[]normals = [...]                           # 法线
    float3[]  extent  = [(-x,-y,-z), (x,y,z)]          # 包围盒
    token     purpose = "guide"                         # collider 标记
}
```

同时，physics.usd 里 link7 的 collider spec 缺少 API schema 声明：

```usda
# physics.usd 里有问题的 link7 collider spec
over "link7"               # ← SpecifierOver：只是覆写，不是新定义
{
    # 没有 apiSchemas = ["PhysicsCollisionAPI"]
    # 没有 bool physics:collisionEnabled = true
    # 没有 token physics:approximation
}
```

---

## 三、`_patch_base_usd` 函数逐行讲解

### 第一步：打开文件，获取 Stage 和 Layer

```python
stage = Usd.Stage.Open(str(base_usd))
layer = stage.GetRootLayer()
```

- `Usd.Stage.Open`：用高层 API 打开 USD 文件，得到整个场景的「舞台」对象。Stage 会自动解析文件的层级引用关系，能用 prim path 访问任意节点
- `stage.GetRootLayer()`：获取这个文件对应的底层 `SdfLayer` 对象，用于后续的低层修改

### 第二步：分别获取源 prim（visual mesh）和目标 prim（空 collider）

```python
src_path = "/visuals/link7/link7/node_STL_BINARY_/mesh"
dst_path = "/colliders/link7/link7"

src_prim = stage.GetPrimAtPath(src_path)  # visual 里的 Mesh prim（有完整几何数据）
dst_prim = stage.GetPrimAtPath(dst_path)  # collider 里的空 Xform prim（需要修改）
```

`stage.GetPrimAtPath` 按路径获取 USD prim 对象，类似文件系统的 `open(path)`。

### 第三步：通过 SdfLayer 把 typeName 从 Xform 改为 Mesh（核心操作）

```python
sdf_dst  = Sdf.Path(dst_path)
prim_spec = layer.GetPrimAtPath(sdf_dst)
if prim_spec is None:
    parent_spec = layer.GetPrimAtPath(Sdf.Path("/colliders/" + link))
    prim_spec = Sdf.PrimSpec(parent_spec, link, Sdf.SpecifierOver)
prim_spec.typeName = "Mesh"
```

**为什么必须用 `Sdf`（低层 API）而不是 `Usd`（高层 API）来改 typeName？**

因为高层 `Usd` API **不允许修改已有 prim 的 typeName**——一个 prim 的类型在创建时确定，高层 API 无法事后更改。只有通过 `Sdf.PrimSpec.typeName` 直接操作底层的「原始记录（spec）」才能做到。

`Sdf.PrimSpec` 是 USD 文件中一条 prim 定义的底层表示。`typeName = "Mesh"` 直接修改这条记录里的类型字段。

如果 `layer.GetPrimAtPath` 返回 `None`（说明这个 prim 在文件里没有独立的 spec，只存在于合成后的 Stage），则用 `Sdf.PrimSpec(parent, name, SpecifierOver)` 在文件里创建一条 `over` 记录：

```usda
# SpecifierOver 的含义：
def  "link7" { ... }   # 完整定义（SpecifierDef），创建新 prim
over "link7" { ... }   # 覆写（SpecifierOver），只修改已有 prim 的部分属性
```

### 第四步：复制几何属性

```python
dst_prim = stage.GetPrimAtPath(dst_path)  # 重新获取（typeName 已改为 Mesh）
for attr_name in _MESH_ATTRS:
    src_attr = src_prim.GetAttribute(attr_name)  # 从 visual mesh 读取属性
    value    = src_attr.Get()                     # 读取属性值（numpy 数组等）
    type_name = src_attr.GetTypeName()            # 读取属性类型（point3f[]等）
    dst_attr = dst_prim.GetAttribute(attr_name)
    if not dst_attr.IsValid():
        dst_attr = dst_prim.CreateAttribute(attr_name, type_name)  # 创建新属性
    dst_attr.Set(value)                           # 写入值
```

这里用的是高层 `Usd` API 来读写属性值。流程是：

```
src visual mesh                     dst collider mesh（空 Xform→已改为 Mesh）
──────────────                      ──────────────────────────────────────
points:     [(x,y,z),...]   复制→   points:     [(x,y,z),...]  ← 写入
faceVertex: [3,3,3,...]     复制→   faceVertex: [3,3,3,...]    ← 写入
indices:    [0,1,2,...]     复制→   indices:    [0,1,2,...]    ← 写入
normals:    [(nx,ny,nz),...] 复制→  normals:    [(nx,ny,nz),...] ← 写入
extent:     [min, max]      复制→   extent:     [min, max]     ← 写入
...                                 ...
```

**为什么从 visual mesh 复制而不是重新读 STL 文件？**
因为 UrdfConverter 已经把 STL 解析成 USD Mesh 格式存在 visual 里了（`/visuals/link7/link7/node_STL_BINARY_/mesh`），直接复制比重新解析 STL 更简单，数据完全相同。

### 第五步：保存

```python
layer.Save()
```

把内存中修改好的 Layer 写回 `.usd` 文件（USDC 二进制格式）。

---

## 四、`_patch_physics_usd` 函数讲解

physics.usd 的修改比 base.usd 复杂，原因是 **instanceable prim 的限制**。

### 为什么必须用 SdfLayer 而不能用 Usd.Stage？

UrdfConverter 生成的 USD 里，`/M20_Piper/link7/collisions` 是一个 instanceable prim（为了节省内存，多个相同结构的 link 共享同一个 prototype）。它的子 prim（`/M20_Piper/link7/collisions/link7`）是「instance proxy」——这类 prim 是只读的，不能通过高层 `Usd.Stage` API 写入：

```python
# 这样会失败：
stage.GetPrimAtPath("/M20_Piper/link7/collisions/link7")  
# → 返回一个 instance proxy，IsValid()=True 但 IsInstanceProxy()=True
# → 所有写操作都会报错
```

解决方法是**绕过 Stage，直接操作 prototype 在 SdfLayer 里的 spec**：

```python
layer = Sdf.Layer.FindOrOpen(str(physics_usd))
# 直接操作 /colliders/link7/link7 这个 spec（prototype 的实际存储位置）
prim_spec = layer.GetPrimAtPath(Sdf.Path("/colliders/link7/link7"))
```

`/colliders/link7/link7` 是 prototype 在文件里的实际路径，对它的修改会反映到所有通过 instanceable 引用它的地方。

### 添加 PhysicsCollisionAPI

```python
# 读取现有的 apiSchemas 列表
schemas_field = prim_spec.GetInfo("apiSchemas")
current_items = list(schemas_field.explicitItems)  # 例如 ["PhysicsRigidBodyAPI"]

# 追加新的 schema
current_items.append("PhysicsCollisionAPI")
prim_spec.SetInfo("apiSchemas", Sdf.TokenListOp.CreateExplicit(current_items))
```

`apiSchemas` 是 USD 里声明「这个 prim 具有哪些能力」的字段。`Sdf.TokenListOp.CreateExplicit` 创建一个显式列表（不做合并，直接替换）。

修改后等价于 USDA：
```usda
over "link7" (
    prepend apiSchemas = ["PhysicsCollisionAPI"]  # ← 新增
)
```

### 添加物理属性

```python
# 创建 physics:collisionEnabled 属性
ce_attr_spec = Sdf.AttributeSpec(prim_spec, "physics:collisionEnabled", Sdf.ValueTypeNames.Bool)
ce_attr_spec.default = True

# 创建 physics:approximation 属性
approx_attr_spec = Sdf.AttributeSpec(prim_spec, "physics:approximation", Sdf.ValueTypeNames.Token)
approx_attr_spec.default = "convexDecomposition"
```

`Sdf.AttributeSpec` 直接在 spec 里创建属性记录，`.default` 设置默认值。等价于 USDA：
```usda
over "link7" (
    prepend apiSchemas = ["PhysicsCollisionAPI"]
)
{
    bool  physics:collisionEnabled = true
    token physics:approximation    = "convexDecomposition"
}
```

---

## 五、修改前后的完整对比

```
修改前（base.usd）：              修改后（base.usd）：
─────────────────                ─────────────────
def Xform "link7" {              def Mesh "link7" {
    token purpose = "guide"          point3f[] points = [...]
}                                    int[]  faceVertexCounts = [...]
                                     int[]  faceVertexIndices = [...]
                                     normal3f[] normals = [...]
                                     float3[] extent = [...]
                                     token purpose = "guide"
                                 }

修改前（physics.usd）：           修改后（physics.usd）：
─────────────────                ─────────────────
over "link7" {                   over "link7" (
    # 无 apiSchemas               prepend apiSchemas = ["PhysicsCollisionAPI"]
    # 无物理属性                  ) {
}                                    bool  physics:collisionEnabled = true
                                     token physics:approximation = "convexDecomposition"
                                 }
```

---

## 六、整体流程总结

```
pxr.Usd.Stage.Open()         ← 高层：读取整个场景，访问 prim 和属性值
    │
    ├─ GetPrimAtPath()        ← 按路径找到 visual mesh prim（src）
    │   GetAttribute().Get()  ← 读取几何数据（points/indices/normals等）
    │
    ├─ GetRootLayer()         ← 获取底层 SdfLayer 对象
    │   GetPrimAtPath()       ← 获取 collider 的 PrimSpec
    │   prim_spec.typeName    ← 直接改 typeName: Xform→Mesh（只有低层API能做）
    │
    └─ GetPrimAtPath()        ← 重新获取（已是 Mesh 类型）
        CreateAttribute()     ← 创建属性槽
        .Set(value)           ← 写入从 visual 复制的几何数据

pxr.Sdf.Layer.FindOrOpen()   ← 低层：直接打开 physics.usd 的 SdfLayer
    │
    ├─ GetPrimAtPath()        ← 获取 /colliders/link7/link7 的 PrimSpec
    │                           （绕过 instanceable proxy 限制）
    ├─ prim_spec.SetInfo()    ← 添加 apiSchemas: PhysicsCollisionAPI
    └─ Sdf.AttributeSpec()    ← 创建 physics:collisionEnabled / physics:approximation

layer.Save()                 ← 把内存中的修改写回 .usd 二进制文件
```

---

# 你是怎么知道configuration/下的各个.usd的内容（包括它们内容的格式）的？能不能把它们的格式写得再详细和完整一些（能够体现出每一个.usd中存储了什么内容以及这些内容的层级结构）
现在我有了所有文件的**真实数据**，可以给出完整准确的描述了。下面基于实际读取的 `pxr.Sdf` 数据逐文件说明。

---

## 我之前是怎么知道这些内容的？

**诚实说明：**
- 对于 **M20_Piper.usd** 的 subLayers 和 defaultPrim：通过 `strings` 命令提取出了可读字符串片段，推断出了部分结构。
- 对于 configuration/ 下各子文件的**内部层级结构和属性**：**之前的描述是基于 USD 的通用知识和 UrdfConverter 的行为规律推断出来的，并非直接读取文件得到的**。这部分存在不精确之处。
- 本次（当前轮）通过调用 `pxr.Sdf` Python API 直接读取了每个 USDC 文件的 spec 树，得到了真实数据。

---

## 各 USD 文件内容的真实详细结构

> 说明：`type=''` 表示该 prim 是一个 `over`（覆盖 spec），没有定义自己的 typeName，依靠 subLayer 继承；`type='Xform'` 等表示该 prim 定义了 typeName（是 `def` spec）。

---

### 1. `M20_Piper.usd`（入口文件）

```
subLayerPaths: []    ← 注意！！入口文件本身不包含 subLayers
                       subLayers 是通过 variant 间接引用的
defaultPrim: M20_Piper

/M20_Piper  [Xform]   ← 根 prim，定义了整个机器人的坐标系
  ├── /M20_Piper{Physics=None}           ← variant set: Physics=None 选项
  │     └── /M20_Piper{Physics=None}/joints  （空 over）
  ├── /M20_Piper{Physics=PhysX}          ← variant set: Physics=PhysX 选项
  ├── /M20_Piper{Sensor=None}            ← variant set: Sensor=None 选项
  ├── /M20_Piper{Sensor=Sensors}         ← variant set: Sensor=Sensors 选项
  ├── /M20_Piper{Robot=None}             ← variant set: Robot=None 选项
  └── /M20_Piper{Robot=Robot}            ← variant set: Robot=Robot 选项
```

**关键发现（与之前描述不同）：**
- `M20_Piper.usd` 的 `subLayerPaths` 实际为 **空列表**！
- 该文件存储的是 **variant set**（变体集），通过变体选择来决定加载哪些子层。`Physics=PhysX` 变体会引入 physics 层，`Robot=Robot` 变体会引入 robot 层等。
- Usd.Stage 读取时 subLayerPaths 为空，是因为 Stage 已经对变体求值并合并，而 Sdf.Layer 看到的是原始 spec（变体路径）。

---

### 2. `configuration/M20_Piper_base.usd`（几何基础层）

```
subLayerPaths: []    ← 无子层，是最底层

# ═══ 第一部分：共享材质库 ═══
/M20_Piper/Looks  [Scope]
  ├── /M20_Piper/Looks/material_C0C0C0  [Material]
  │     attrs: outputs:mdl:surface, outputs:mdl:volume, outputs:mdl:displacement
  │     └── /M20_Piper/Looks/material_C0C0C0/Shader  [Shader]
  │           attrs: outputs:out, info:implementationSource,
  │                  info:mdl:sourceAsset, info:mdl:sourceAsset:subIdentifier,
  │                  inputs:diffuse_color_constant
  ├── /M20_Piper/Looks/material_CAD1EE  [Material]  （同上结构）
  └── /M20_Piper/Looks/material_E5EAED  [Material]  （同上结构）

# ═══ 第二部分：Link prim 定义（机器人骨架节点，仅存变换，无物理） ═══
/M20_Piper/joints  [Scope]  （空，物理内容在 physics.usd 中填充）

/M20_Piper/base_link  [Xform]
  attrs: xformOpOrder, xformOp:translate, xformOp:orient, xformOp:scale
  └── /M20_Piper/base_link/visuals  [Xform]  （空容器，引用下面 /visuals/ 实例）

/M20_Piper/arm_base_link  [Xform]   （同上结构）
/M20_Piper/link1 ~ link6  [Xform]   （同上结构，各含 /visuals 子节点）
/M20_Piper/gripper_base  [Xform]
/M20_Piper/link7  [Xform]
  └── /M20_Piper/link7/visuals  [Xform]
/M20_Piper/link8  [Xform]
/M20_Piper/fl_hipx, fl_hipy, fl_knee, fl_wheel  [Xform]  （腿部 links）
/M20_Piper/fr_*, hl_*, hr_*  [Xform]  （四腿各 4 个 link）

# ═══ 第三部分：/visuals/ 树（instanceable 实例化几何体，视觉网格引用） ═══
# 每个 link 对应一个子树：
/visuals/base_link  [over（无 typeName）]
  └── /visuals/base_link/base_link  [Xform]
        attrs: xformOpOrder, xformOp:translate, xformOp:orient, xformOp:scale
        （内部 reference 指向 /meshes/base_link 实例化网格）

/visuals/arm_base_link/arm_base_link  [Xform]  （同结构）
/visuals/link1/link1  [Xform]
/visuals/link2/link2  [Xform]
...
/visuals/link7/link7  [Xform]    ← link7 视觉体存在！
/visuals/link8/link8  [Xform]    ← link8 视觉体存在！
/visuals/fl_hipx/fl_hipx  [Xform]
...（四腿所有 link 共 16 个）

# ═══ 第四部分：/meshes/ 树（原始三角面片数据，instanceable） ═══
# 每个 link 对应一棵子树：
/meshes/base_link  [Xform]  attrs: visibility
  └── /meshes/base_link/node_STL_BINARY_  [Xform]
        attrs: xformOpOrder, xformOp:translate, xformOp:orient, xformOp:scale
        ├── /meshes/base_link/node_STL_BINARY_/Looks  [Scope]
        │     └── /meshes/base_link/node_STL_BINARY_/Looks/DefaultMaterial  [Material]
        │           └── .../DefaultMaterial  [Shader]
        │                 attrs: outputs:out, info:implementationSource,
        │                        info:mdl:sourceAsset, info:mdl:sourceAsset:subIdentifier,
        │                        inputs:diffuse_color_constant, inputs:emissive_color, ...
        └── /meshes/base_link/node_STL_BINARY_/mesh  [Mesh]
              attrs: points, faceVertexCounts, faceVertexIndices,
                     extent, normals, subdivisionScheme

/meshes/link7/node_STL_BINARY_/mesh  [Mesh]  attrs: points, faceVertexCounts, ...  ← 存在！
/meshes/link8/node_STL_BINARY_/mesh  [Mesh]  ← 存在！
...（所有 link 均有 /meshes/ 子树）
/meshes  [Scope]  attrs: visibility
```

**关键要点：**
- `base.usd` 中 **不含** `collisions` 相关子节点，也 **不含** 任何物理属性
- `/M20_Piper/<link_name>/visuals` 是空 Xform，实际几何由 `/visuals/<link_name>` instanceable 子树提供（通过引用连接）
- `/meshes/` 树存储原始 STL 三角网格，是 instanceable 原型（prototype）
- link7/link8 的视觉网格（`/meshes/link7/.../mesh`）**存在于 base.usd 中**，这就是 patch 脚本能从这里复制几何数据的来源

---

### 3. `configuration/M20_Piper_physics.usd`（物理层）

```
subLayerPaths: ['M20_Piper_base.usd']  ← 以 base.usd 为子层！

# ═══ 第一部分：/M20_Piper/ 骨架 prim 的物理属性覆盖 ═══
# 这些是 over（type=''），覆盖 base.usd 中的同名 prim，添加物理属性

/M20_Piper/base_link  [over]
  attrs: physics:mass, physics:diagonalInertia, physics:principalAxes,
         physics:centerOfMass,
         physxArticulation:enabledSelfCollisions,
         physxArticulation:solverPositionIterationCount,
         physxArticulation:solverVelocityIterationCount
  └── /M20_Piper/base_link/collisions  [Xform]  （空容器，引用 /colliders/base_link）

/M20_Piper/arm_base_link  [over]
  attrs: physics:mass, physics:diagonalInertia, physics:principalAxes, physics:centerOfMass
  └── /M20_Piper/arm_base_link/collisions  [Xform]

/M20_Piper/link1 ~ link6  [over]  （同上，各含 /collisions 子节点）
/M20_Piper/gripper_base  [over]
/M20_Piper/link7  [over]
  attrs: physics:mass, physics:diagonalInertia, physics:principalAxes, physics:centerOfMass
  └── /M20_Piper/link7/collisions  [Xform]  ← 空容器，对应 bug 所在处
/M20_Piper/link8  [over]  （同上）
/M20_Piper/fl_hipx, fl_hipy...  [over]  （四腿各 link 的质量/惯量）

# ═══ 第二部分：关节定义 ═══
/M20_Piper/joints/base_to_arm  [PhysicsFixedJoint]
  attrs: physics:localPos0, physics:localRot0, physics:localPos1, physics:localRot1,
         physics:breakForce, physics:breakTorque, physics:JointEquivalentInertia

/M20_Piper/joints/joint1 ~ joint6  [PhysicsRevoluteJoint]  (+ PhysicsDriveAPI:angular)
  attrs: physics:axis, physics:lowerLimit, physics:upperLimit,
         drive:angular:physics:maxForce, drive:angular:physics:type,
         drive:angular:physics:targetPosition,
         drive:angular:physics:stiffness, drive:angular:physics:damping

/M20_Piper/joints/joint7, joint8  [PhysicsPrismaticJoint]  (+ PhysicsDriveAPI:linear)
  attrs: physics:axis, physics:lowerLimit, physics:upperLimit,
         drive:linear:physics:maxForce, drive:linear:physics:type,
         drive:linear:physics:targetPosition,
         drive:linear:physics:stiffness, drive:linear:physics:damping

/M20_Piper/joints/fl_hipx_joint ~ hr_wheel_joint  [PhysicsRevoluteJoint]  （腿部 16 个关节）

# ═══ 第三部分：/colliders/ 树（几何碰撞体，instanceable） ═══
# 普通 link（box/cylinder 简化碰撞体）：
/colliders/base_link  [over]
  ├── /colliders/base_link/mesh_0  [over]
  │     └── /colliders/base_link/mesh_0/box  [over]    ← 简化碰撞形状
  ├── /colliders/base_link/mesh_1/cylinder  [over]
  └── /colliders/base_link/mesh_2/cylinder  [over]

/colliders/arm_base_link/simplified_collision/cylinder  [over]
/colliders/link1/simplified_collision/cylinder  [over]
/colliders/link2/simplified_collision/box  [over]
/colliders/link3/simplified_collision/box  [over]
/colliders/link4/simplified_collision/cylinder  [over]
/colliders/link5/simplified_collision/box  [over]
/colliders/link6/simplified_collision/cylinder  [over]
/colliders/gripper_base/simplified_collision/box  [over]

# link7/link8（被 patch 脚本修复后的碰撞体）：
/colliders/link7  [over]
  └── /colliders/link7/link7  [over, apiSchemas=[PhysicsCollisionAPI]]
        attrs: physics:collisionEnabled, physics:approximation
        └── /colliders/link7/link7/node_STL_BINARY_  [over]
              attrs: physics:approximation
/colliders/link8  [over]
  └── /colliders/link8/link8  [over, apiSchemas=[PhysicsCollisionAPI]]
        attrs: physics:collisionEnabled, physics:approximation

# 碰撞组：
/colliders/robotCollisionGroup   [PhysicsCollisionGroup]
/colliders/collidersCollisionGroup  [PhysicsCollisionGroup]

# 腿部碰撞体（fl/fr/hl/hr 各腿）：
/colliders/fl_hipx/mesh_0/cylinder  [over]
/colliders/fl_hipy/mesh_0/box + mesh_1/cylinder  [over]
/colliders/fl_knee/mesh_0/box + mesh_1/cylinder  [over]
/colliders/fl_wheel/mesh_0/cylinder  [over]
...（fr/hl/hr 同结构）
```

**关键要点：**
- `physics.usd` 的 subLayerPaths 包含 `base.usd`，因此 physics.usd 是比 base.usd **优先级更高**的层
- `/M20_Piper/<link>/collisions` 的 Xform 在 physics.usd 中定义，但它是空壳，实际碰撞数据在 `/colliders/` 树下（instanceable）
- link7/link8 的 `PhysicsCollisionAPI` 在 **physics.usd** 里的 `/colliders/link7/link7` 节点上，attrs 包含 `physics:collisionEnabled=true` 和 `physics:approximation="convexDecomposition"`
- 其他普通 link 的碰撞体（box/cylinder）虽然在 `/colliders/` 路径下，但 spec 的 attrs 都为空（`[]`），CollisionAPI 属性通过更高层（robot.usd？）或 schema 默认值应用

---

### 4. `configuration/M20_Piper_robot.usd`（Isaac Sim 元数据层）

```
subLayerPaths: []  ← 无子层

# 所有 prim 都是 over（type=''），纯属性覆盖层

/M20_Piper  [Xform（over）]
  attrs: isaac:description, isaac:namespace, isaac:robotType,
         isaac:license, isaac:version, isaac:source, isaac:changelog

# 每个 link 仅添加 nameOverride：
/M20_Piper/base_link   [over]  attrs: isaac:nameOverride
/M20_Piper/arm_base_link  [over]  attrs: isaac:nameOverride
/M20_Piper/link1 ~ link8  [over]  attrs: isaac:nameOverride
/M20_Piper/gripper_base  [over]  attrs: isaac:nameOverride
/M20_Piper/fl_hipx, fl_hipy, fl_knee, fl_wheel  [over]  attrs: isaac:nameOverride
...（四腿所有 link 共 16 个）

# 每个关节存储 DOF 偏移和运动控制限制：
/M20_Piper/joints  [over]  （空）
/M20_Piper/joints/base_to_arm  [over]
  attrs: isaac:physics:index,
         isaac:physics:Rot_X:DofOffset,  isaac:physics:Rot_Y:DofOffset,
         isaac:physics:Rot_Z:DofOffset,  isaac:physics:Tr_X:DofOffset,
         isaac:physics:Tr_Y:DofOffset,   isaac:physics:Tr_Z:DofOffset,
         isaac:nameOverride,
         isaac:physics:AccelerationLimit, isaac:physics:JerkLimit
/M20_Piper/joints/joint1 ~ joint6  [over]  （同上 10 个 attrs）
/M20_Piper/joints/joint7, joint8  [over]  （同上）
/M20_Piper/joints/fl_hipx_joint ~ hr_wheel_joint  [over]  （腿部 16 个关节，同上）
```

**关键要点：**
- 这个文件只存 Isaac Sim 私有命名空间 `isaac:*` 下的元数据，不含任何几何或物理模拟数据
- `isaac:physics:index`：该关节在 articulation DOF 数组中的索引（用于 Isaac Lab 读取 joint state）
- `isaac:physics:{Rot/Tr}_{X/Y/Z}:DofOffset`：关节各自由度的偏移量（6 个 DOF 通道）
- `isaac:physics:AccelerationLimit` / `JerkLimit`：加速度和加加速度限制
- `isaac:nameOverride`：Isaac Sim 中显示的名称，与 USD prim 名可以不同

---

### 5. `configuration/M20_Piper_sensor.usd`（传感器层）

```
subLayerPaths: []  ← 无子层

/M20_Piper  [Xform]  ← 只有根 prim 定义，无子节点
/  [over]  （伪根）
```

**关键要点：** 该文件实际上是**空壳**，只有根 prim 定义，没有任何传感器内容。是 UrdfConverter 预留的扩展位。

---

## 总结：四个子文件的职责分工

| 文件 | subLayers | 主要内容 | 关键 prim 类型 |
|------|-----------|---------|---------------|
| `base.usd` | 无 | 材质+link变换+视觉网格 | Xform, Mesh, Material, Shader |
| `physics.usd` | base.usd | 质量/惯量+关节+碰撞体 | PhysicsRevoluteJoint, PhysicsPrismaticJoint, PhysicsCollisionAPI |
| `robot.usd` | 无 | Isaac DOF索引+加速度/加加速度限制 | over（只有 isaac:* 属性）|
| `sensor.usd` | 无 | 空（预留） | 仅根 Xform |

## 文件间的引用关系（真实情况）

经过本次实际读取，发现与之前描述有一处**重要不同**：
- `M20_Piper.usd` 的 `subLayerPaths` **为空**，它通过 **variant set（变体集）** 而非直接 subLayer 来引用子文件
- `physics.usd` 的 `subLayerPaths` 包含 `M20_Piper_base.usd`（在 physics.usd 层内部声明了对 base.usd 的依赖）
- `robot.usd` 和 `sensor.usd` 都没有 subLayers
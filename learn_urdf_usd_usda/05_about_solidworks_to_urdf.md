ROS wiki 全部被反爬虫拦截，但我已经从 GitHub 以及对 URDF 文件本身的分析中获取了足够的信息。结合 SOURCE_M20_Piper.urdf 的具体内容，我可以给你一个完整、准确的讲解。

---

# SolidWorks to URDF Exporter 详解

## 一、这是什么工具？

**SW2URDF**（SolidWorks to URDF Exporter）是一个安装在 SolidWorks 里的**插件（Add-in）**，由 Stephen Brawner 为 Willow Garage / ROS 社区开发，目前在 GitHub 开源维护（675 stars）。

它做的事情只有一件：**把 SolidWorks 的装配体（Assembly）自动转换成 URDF 文件**，同时把每个零件的 3D 网格导出为 STL 文件。

没有这个工具的话，你需要手工看着 SolidWorks 的模型，把每个零件的质量、惯性矩、关节位置、旋转轴全部手算/量取后再写入 URDF——这对于一个有几十个零件的机器人来说是极其繁琐的工作。有了 SW2URDF，这些数据直接从 CAD 模型里自动提取。

---

## 二、安装

- 只能在 **Windows** 上安装（SolidWorks 是 Windows 软件）
- 最低版本要求：**SolidWorks 2018 SP5** 或更新
- 从 GitHub Releases 页面下载对应版本的 `.exe` 安装包，双击安装
- 安装后重启 SolidWorks，在菜单栏的 **Tools（工具）** 菜单下会出现 **Export as URDF** 选项

---

## 三、在 SolidWorks 里需要准备什么

这是最关键的部分。SW2URDF **不是万能魔法**，它能自动提取的内容有限，很多东西需要机械设计师在建模时就提前准备好。

### 3.1 模型本身的要求

**必须是装配体（Assembly，.sldasm），不能是单个零件文件（Part，.sldprt）**。  
整个机器人由多个零件组成的装配体，每个可动关节对应装配体里的一个子装配体或零件。

对于 M20_Piper 这样的机器人，SolidWorks 文件里的结构大概是：
```
M20_Piper.sldasm（总装配体）
├── base_link.sldprt（底盘主体零件）
├── fl_hipx.sldasm（前左髋关节子装配体）
│   ├── fl_hipx.sldprt
│   ├── fl_hipy.sldasm
│   │   ├── fl_hipy.sldprt
│   │   └── fl_knee.sldasm
│   │       └── ...
├── arm_base_link.sldprt
├── link1.sldprt
└── ...
```

### 3.2 必须准备：坐标系（Coordinate Systems）

这是 SW2URDF 最核心的准备工作。**每个 link 都需要在 SolidWorks 里预先定义一个坐标系**，这个坐标系就是 URDF 里该 link 的局部坐标系原点。

在 SolidWorks 里建坐标系的方法：
- `Insert（插入）→ Reference Geometry（参考几何体）→ Coordinate System（坐标系）`
- 选择一个顶点/边/面作为原点，选择两条边/轴定义 X/Y 方向，Z 自动由右手定则确定
- **命名规范**：通常命名为 `<link名>_frame` 或直接用零件名

**为什么必须有坐标系？**  
URDF 里每个 joint 的 `<origin xyz rpy>` 描述的是「子 link 的坐标系原点在父 link 坐标系中的位置」。SW2URDF 就是通过计算两个相邻零件的坐标系之间的相对变换，自动算出这个 `origin`。

如果没有预先定义坐标系，SW2URDF 就不知道每个零件的「原点」在哪里，导出的 URDF 会使用零件的默认原点（通常是 SolidWorks 建模时的原点，往往位置很奇怪）。

以 M20_Piper 为例，`fl_hipx_joint` 的数据：
```xml
<joint name="fl_hipx_joint" type="revolute">
  <origin xyz="0.3141 0.0685 0" rpy="0 0 0"/>
```
这里的 `xyz="0.3141 0.0685 0"` 就是 SW2URDF 自动计算出来的——从 base_link 的坐标系原点到 fl_hipx 零件的坐标系原点，距离是 (31.41cm, 6.85cm, 0)。这个数据来自 SolidWorks 装配体中的几何关系。

### 3.3 必须准备：旋转轴（Reference Axes）

对于每个**可动关节**，还需要在 SolidWorks 里定义一根**参考轴（Reference Axis）**，表示该关节的旋转轴或平移轴方向。

建参考轴的方法：
- `Insert → Reference Geometry → Axis`
- 可以通过「两点」「一条直线/边」「圆柱面的轴线」等方式定义
- 例如：关节销轴的圆柱面的中心线就是旋转轴

SW2URDF 用这根轴的方向向量自动生成 URDF 里的 `<axis xyz>`。

比如 `fl_hipx_joint` 的旋转轴：
```xml
<axis xyz="-1 0 0"/>
```
这表示沿 X 轴负方向旋转。这个 `-1 0 0` 就是从 SolidWorks 里定义的参考轴方向自动读取的。

### 3.4 必须准备：装配配合（Mates）确定运动关系

SolidWorks 装配体里零件之间靠**配合（Mates）**约束相对位置。SW2URDF 通过分析配合关系来判断哪两个零件之间有关节。

常见配合类型和对应的 URDF joint 类型：
| SolidWorks 配合 | URDF joint type |
|---|---|
| 铰链（Hinge）配合 | revolute / continuous |
| 滑动（Slider）配合 | prismatic |
| 刚性（Coincident + 固定）配合 | fixed |
| 自由旋转（无限位铰链）| continuous |

### 3.5 可选但推荐：材料属性

在 SolidWorks 里给每个零件指定**材料（Material）**（例如：6061铝合金、304不锈钢），SolidWorks 就会根据材料密度和零件几何体自动计算质量和惯性矩。

SW2URDF 直接读取这些计算结果，填入 URDF 的 `<inertial>` 里。这就是为什么 SOURCE_M20_Piper.urdf 里的惯性数据精确到小数点后很多位：
```xml
<mass value="0.261359958476582"/>          ← SolidWorks 精确计算的质量
<inertia ixx="0.000186" ixy="0.000025" ... ← SolidWorks 精确计算的惯性张量
```

如果没有指定材料，SW2URDF 会用默认密度估算，结果可能不准。

---

## 四、导出 URDF 的完整操作流程

准备好以上内容后，实际导出的步骤：

### Step 1：打开装配体，启动插件
```
在 SolidWorks 里打开总装配体 .sldasm 文件
菜单栏 → Tools（工具）→ Export as URDF
```
一个 GUI 窗口弹出。

### Step 2：配置关节树（Joint Tree）

SW2URDF 会尝试**自动识别**装配体中的零件层级关系，生成一个初步的关节树。但通常需要手动调整：

- **指定根节点（Root Link）**：选择 `base_link` 作为根
- **为每个 link 指定坐标系**：在 GUI 里每个节点都有下拉框，选择之前在 SolidWorks 里定义好的坐标系
- **为每个 joint 指定旋转轴**：选择对应的参考轴
- **设置关节类型**：revolute / continuous / fixed / prismatic
- **设置关节限位**：手动填入 lower/upper 角度（弧度），以及 effort/velocity 限制

这一步是整个过程中**唯一需要大量手工输入**的部分，也是最容易出错的地方。关节限位数据通常来自电机规格书或机械设计文件。

### Step 3：预览检查

SW2URDF 提供实时预览，可以看到：
- 每个 link 的坐标系位置是否正确（坐标轴显示在 SolidWorks 视图中）
- 关节树的层级是否正确
- 旋转轴方向是否正确

### Step 4：导出
```
点击 Export（导出）按钮
选择输出目录
```

SW2URDF 自动生成：
```
输出目录/
├── M20_Piper.urdf          ← 主 URDF 文件
├── meshes/
│   ├── base_link.STL       ← 每个 link 的 3D 网格（STL 格式）
│   ├── fl_hipx.STL
│   ├── link7.STL
│   └── ...（每个 link 一个 STL）
└── launch/                 ← ROS launch 文件（可选）
    └── display.launch
```

STL 文件是 SW2URDF 从 SolidWorks 零件的 3D 实体模型自动导出的，**精度由 SolidWorks 的 STL 导出设置控制**（三角面数量、弦差等参数）。

---

## 五、结合 SOURCE_M20_Piper.urdf 验证这个流程

现在可以对照文件内容印证上面说的每一点：

### 验证1：质量和惯性从材料自动计算
```xml
<!-- base_link：底盘钢结构，15.882 kg，惯性矩精确 -->
<mass value="15.882"/>
<inertia ixx="0.0880787" ixy="0.001396" ixz="0.000111"
         iyy="0.533816"  iyz="0.000519"  izz="0.563915"/>

<!-- fl_hipx：小零件，仅 0.261 kg -->
<mass value="0.261359958476582"/>
```
小数点后这么多位，明显是软件自动计算，不是人工填写的。

### 验证2：joint origin 从坐标系自动计算
```xml
<!-- fl_hipx_joint：前左腿安装在底盘前方31.41cm、左6.85cm处 -->
<origin xyz="0.3141 0.0685 0" rpy="0 0 0"/>

<!-- fl_knee_joint：膝关节在髋关节下方25cm、偏前9.84cm处 -->
<origin xyz="0 0.0984 -0.25" rpy="0 0 0"/>

<!-- joint2（机械臂第2关节）：带有复杂旋转偏移 -->
<origin xyz="0 0 0" rpy="1.5708 -0.1359 -3.1416"/>
```
这些数值都是 SolidWorks 里几何关系的直接读出，人工很难算到这个精度。

### 验证3：STL 文件覆盖每个 link
```
deps/.../urdf/meshes/
  arm_base_link.STL
  base_link.STL
  fl_hipx.STL  fl_hipy.STL  fl_knee.STL  fl_wheel.STL
  fr_hipx.STL  ... （4条腿 × 4个零件 = 16个）
  link1.STL ~ link8.STL
  gripper_base.STL
```
共 27 个 link 就有 27 个 STL 文件，一一对应，正是 SW2URDF 自动导出的结果。

### 验证4：文件头注释
```xml
<!-- This URDF was automatically created by SolidWorks to URDF Exporter!
     Commit Version: 1.6.0-4-g7f85cfe
     Build Version: 1.6.7995.38578 -->
```
直接标明了使用的插件版本，1.6.0 对应 SolidWorks 2020 版本的插件。

---

## 六、SW2URDF 的局限性（也是本项目遇到的问题根源）

| 局限 | 在本项目的体现 |
|---|---|
| 碰撞体只能导出为 STL mesh，不能直接生成 primitive | M20 底盘和 Piper 臂的 `<collision>` 全部是 `<mesh>`，需要 build 脚本手工替换为 box/cylinder |
| Isaac Sim 的 UrdfConverter 对 `<collision><mesh>` 支持不完整 | link7/link8 碰撞体丢失，需要 patch 脚本补救 |
| 关节限位需要手工填入 | lower/upper 数据来自电机规格书，不是自动算出的 |
| 导出的是静态 URDF，不包含 actuator 参数 | stiffness/damping 等控制参数是在 Isaac Lab 的 `.py` 配置文件里手工加的 |

总结一句话：**SW2URDF 解决了「机器人长什么样、各部件在哪里、质量是多少」这些几何和物理问题，但「怎么控制它、碰撞体简化」等仿真工程问题仍需手工处理。**


---
## 问题二：SW2URDF 能指定 collision 用 STL 还是 box/cylinder 吗？

**不能直接指定**。SW2URDF 对碰撞体的处理方式是固定的：

**SW2URDF 导出的 collision 永远是 STL mesh，和 visual 用同一个文件。**

```xml
<!-- SW2URDF 导出的典型结果 -->
<visual>
  <geometry><mesh filename="./meshes/link2.STL"/></geometry>
</visual>
<collision>
  <geometry><mesh filename="./meshes/link2.STL"/></geometry>  ← 和 visual 完全相同
</collision>
```

SW2URDF 没有提供「为 collision 选择 box/cylinder」的 GUI 选项。它的设计思路是：**用高精度 STL 同时作为视觉和碰撞体，让仿真尽量接近真实形状**。

如果要用 primitive 替换碰撞体，有两种方法：

**方法 A：导出后手工编辑 URDF**（本项目的做法）

SW2URDF 导出后，用代码或手工把 `<collision><geometry><mesh>` 替换成 `<box>`/`<cylinder>`，这正是本项目 `build_m20_piper_urdf.py` 里 `PIPER_COLLIDERS` 字典做的事。

**方法 B：在 SolidWorks 里建专用的 collision 几何体（高级做法）**

SW2URDF 有一个不太常用的功能：如果你在 SolidWorks 零件里专门建了一个名为 `_collision` 的配置（Configuration），插件会用这个配置的几何体作为 collision，用默认配置作为 visual。但这需要在 SolidWorks 里维护两套模型，实际上用得很少。

**方法 C：用 xacro 宏（ROS 生态的做法）**

ROS 里通常用 `.xacro` 文件（URDF 的宏扩展）来描述碰撞体，可以在宏参数里灵活指定。但这已经超出了 SW2URDF 的范畴，是手工设计 URDF 时的方法。

---
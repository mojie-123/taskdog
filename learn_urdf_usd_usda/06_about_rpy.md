# URDF 中 rpy 参数的完整讲解

---

## 一、rpy 是什么

**rpy = roll, pitch, yaw**，是描述「旋转」的一种方式，三个数字分别代表绕三根轴的旋转角度，单位是**弧度（rad）**。

```
rpy="roll  pitch  yaw"
      ↓      ↓     ↓
    绕X轴  绕Y轴  绕Z轴
    滚转   俯仰   偏航
```

旋转顺序是固定的：**先 roll（绕X）→ 再 pitch（绕Y）→ 再 yaw（绕Z）**，这三次旋转合在一起描述一个完整的姿态变换。

常用的特殊值：
- `0`：不旋转
- `1.5708` ≈ π/2 = **90°**
- `3.1416` ≈ π = **180°**
- `-1.5708` ≈ -π/2 = **-90°**

URDF 里的坐标系约定（右手系）：
```
     Z（上）
     ↑
     │
     └────→ Y（左）
    ╱
   X（前）
```

---

## 二、rpy 在不同标签中的含义（分别讲）

---

### 2.1 `<inertial>` 里的 `<origin rpy>`

**含义：质心坐标系相对于 link 坐标系的旋转**

```xml
<inertial>
  <origin xyz="0.00355 0.00165 -0.00346" rpy="0 0 0"/>
  <mass value="15.882"/>
  <inertia ixx="0.088" ixy="0.001" .../>
</inertial>
```

物理背景：惯性张量（ixx/iyy/izz/ixy/ixz/iyz）描述物体绕某个坐标系的旋转阻力。这个坐标系**不一定**和 link 坐标系方向相同——如果质量分布不对称，惯性张量的「主轴」方向可能是倾斜的。

`rpy` 在这里描述的是：**惯性张量是在哪个坐标系下计算出来的**，该坐标系相对于 link 坐标系有多少旋转。

**实际情况**：SolidWorks 导出时，对于大多数零件，质心坐标系和 link 坐标系方向相同（`rpy="0 0 0"`），所以本文件里几乎所有 `<inertial>` 的 rpy 都是零，直接读即可，不需要深究。

如果 rpy 不是零，意思是：「我给你的 ixx/iyy/izz 这些数字，是在一个相对于 link 坐标系旋转了 (roll,pitch,yaw) 的坐标系下定义的，你物理引擎在计算时要先把这个旋转考虑进去。」

---

### 2.2 `<visual>` 里的 `<origin rpy>`

**含义：视觉模型（STL 网格）相对于 link 坐标系的旋转**

```xml
<visual>
  <origin xyz="0 0 0" rpy="0 0 0"/>
  <geometry>
    <mesh filename="./meshes/base_link.STL"/>
  </geometry>
</visual>
```

这是最直观的：**STL 文件里的几何体，在渲染时要旋转多少角度才能对齐到 link 坐标系的正确朝向。**

SW2URDF 导出时，STL 的坐标系和 link 坐标系通常已经对齐（因为它们都从 SolidWorks 同一个零件导出），所以大多数情况下是 `rpy="0 0 0"`——STL 文件里的模型原点和朝向，就是 link 坐标系的原点和朝向，不需要额外旋转。

**什么时候不是零？**  
如果设计师建模时 STL 文件的朝向和希望的 link 坐标系不一致（例如 STL 里圆柱是竖立的，但 link 需要它横躺），就要在 rpy 里旋转来纠正。

---

### 2.3 `<collision>` 里的 `<origin rpy>`

**含义：碰撞几何体相对于 link 坐标系的旋转**

这里 rpy 最有实际意义，也最容易看到非零值。来看几个具体例子：

#### 例子 A：base_link 的侧面圆柱碰撞体

```xml
<collision>
  <origin xyz="0 0.045 0" rpy="0 1.57079632 0"/>
  <geometry>
    <cylinder length="0.75" radius="0.07"/>
  </geometry>
</collision>
```

SolidWorks 里，`<cylinder>` 的默认方向是**沿 Z 轴竖立**：
```
  Z
  ↑
  │  ╔═╗
  │  ║ ║  ← 默认竖立圆柱
  │  ╚═╝
  └──────→ Y
```

但机器人底盘的侧面保护体需要**横躺，沿 X 方向延伸 75cm**：
```
  Z
  ↑
  │
  │  ══════════════════  ← 需要横卧的圆柱
  └──────────────────→ X
```

`rpy="0 1.57079632 0"` = `rpy="0 π/2 0"` = **绕 Y 轴旋转 90°**，把竖立的圆柱「推倒」变成横卧，从而沿 X 方向延伸。这样就实现了保护机器人侧面轮子的效果。

#### 例子 B：fl_hipx 髋关节的圆柱碰撞体

```xml
<!-- fl_hipx 的碰撞体 -->
<collision>
  <origin xyz="0 0.055 0" rpy="1.57079632 0 0"/>
  <geometry>
    <cylinder length="0.14" radius="0.048"/>
  </geometry>
</collision>
```

`rpy="1.57079632 0 0"` = **绕 X 轴旋转 90°**，同样是把竖立圆柱推倒，但这次是**沿 Y 轴方向横卧**（关节销轴沿 Y 方向穿过）。

```
  Z             Z
  ↑             ↑
  │  ╔═╗        │
  │  ║ ║  →绕X旋转90°→  ══╦══  ← 沿Y方向横卧
  │  ╚═╝        │
  └──→ Y        └──→ Y
  （默认竖立）         （绕X转90°后横卧）
```

#### 例子 C：fl_hipy 大腿的组合碰撞体

```xml
<!-- 碰撞体1：大腿主杆（box，不旋转）-->
<collision>
  <origin xyz="-0.01 0.11 -0.125" rpy="0 0 0"/>
  <geometry>
    <box size="0.07 0.02 0.25"/>
  </geometry>
</collision>

<!-- 碰撞体2：膝关节处圆柱（旋转后横卧）-->
<collision>
  <origin xyz="0 0.11 -0.25" rpy="1.57079632 0 0"/>
  <geometry>
    <cylinder length="0.09" radius="0.05"/>
  </geometry>
</collision>
```

box 不需要旋转（`rpy="0 0 0"`），因为大腿杆本来就沿 Z 轴方向（向下）延伸，box 的默认朝向就是对的。  
膝盖处圆柱需要绕 X 轴旋转 90°，变成沿 Y 轴方向横卧（模拟关节销轴）。

---

### 2.4 `<joint>` 里的 `<origin rpy>`

**含义：子 link 坐标系相对于父 link 坐标系的旋转**

这里的 rpy 最复杂，但也最重要——它决定了子 link 坐标系的初始朝向。

#### 例子 D：大多数腿部关节 rpy=0

```xml
<joint name="fl_hipx_joint" type="revolute">
  <origin xyz="0.3141 0.0685 0" rpy="0 0 0"/>
  <parent link="base_link"/>
  <child link="fl_hipx"/>
</joint>
```

`rpy="0 0 0"` 说明：`fl_hipx` 的坐标系朝向和 `base_link` 完全相同，只是位置偏移了（xyz 部分）。腿部关节的坐标轴方向和底盘保持一致，这很自然。

#### 例子 E：机械臂 joint2 的复杂旋转

```xml
<joint name="joint2" type="revolute">
  <origin xyz="0 0 0" rpy="1.5708 -0.1359 -3.1416"/>
  <parent link="link1"/>
  <child link="link2"/>
  <axis xyz="0 0 1"/>
</joint>
```

`rpy="1.5708 -0.1359 -3.1416"` = **roll=90°, pitch=-7.78°, yaw=-180°**。这非常复杂，是 SolidWorks 里 Piper 臂的关节坐标系之间的真实旋转关系，来自机械设计的角度约束。

直觉理解：机械臂的各个关节坐标系不是和底盘对齐的，每个关节都有自己的朝向（这样 `<axis xyz="0 0 1">` 才能正确描述旋转轴在局部坐标系里的方向）。

#### 例子 F：夹爪手指关节

```xml
<!-- joint7（左手指）-->
<joint name="joint7" type="prismatic">
  <origin xyz="0 0 0.1358" rpy="1.5708 0 0"/>
  <parent link="gripper_base"/>
  <child link="link7"/>
  <axis xyz="0 0 1"/>
</joint>

<!-- joint8（右手指）-->
<joint name="joint8" type="prismatic">
  <origin xyz="0 0 0.1358" rpy="1.5708 0 -3.1416"/>
  <parent link="gripper_base"/>
  <child link="link8"/>
  <axis xyz="0 0 -1"/>
</joint>
```

两个手指都在 gripper_base 前方 13.58cm 处（`xyz="0 0 0.1358"`），都需要绕 X 轴旋转 90°（`roll=1.5708`）来改变手指坐标系的朝向。

joint8 多了一个 `yaw=-3.1416`（= **yaw=-180°**），这相当于把右手指绕 Z 轴翻转 180°，让它朝向与 link7 **相反的方向**。配合 `<axis xyz="0 0 -1">`（负Z方向运动），两根手指就能**向中间对称夹合**。

用图示意：
```
gripper_base 坐标系：

        Z（前）
        ↑
   link8← ─ → link7
   (yaw=-180°)  (yaw=0°)
   axis=-Z      axis=+Z

关节运动：link7 沿+Z滑动（向右张开），link8 沿-Z滑动（向左张开）
         → 两指同时向外张，同时向内夹
```

---

## 三、总结对照表

| 标签位置 | rpy 的含义 | 本文件常见值 |
|---|---|---|
| `<inertial><origin rpy>` | 惯性张量所在坐标系相对 link 坐标系的旋转 | 几乎全是 `0 0 0` |
| `<visual><origin rpy>` | STL 网格相对 link 坐标系的旋转（显示用） | 几乎全是 `0 0 0` |
| `<collision><origin rpy>` | 碰撞形状相对 link 坐标系的旋转（物理用） | 经常有 `1.5708 0 0` 等，用来把 cylinder 横躺 |
| `<joint><origin rpy>` | 子 link 坐标系相对父 link 坐标系的旋转 | 简单关节是 `0 0 0`，复杂臂关节有复杂值 |

**一个记忆口诀**：
- `rpy="0 0 0"` → 不旋转，坐标轴方向继承父坐标系
- `rpy="1.5708 0 0"` → 绕X旋转90°，把「竖」变「沿Y横卧」
- `rpy="0 1.5708 0"` → 绕Y旋转90°，把「竖」变「沿X横卧」
- `rpy="0 0 1.5708"` → 绕Z旋转90°，在水平面内旋转朝向
- `rpy="0 0 3.1416"` → 绕Z旋转180°，反向（相当于翻转）


---



逐一回答你的四个问题：

---

## 问题一：URDF 的 rpy 是 fixed angles 还是 euler angles？

**URDF 的 rpy 是 Fixed Angles（也叫 extrinsic/静止坐标系旋转）**，具体是：

> 绕**全局固定坐标系**（即父坐标系）的 X、Y、Z 轴，**依次**旋转 roll、pitch、yaw。

数学上等价于：
```
R = Rz(yaw) · Ry(pitch) · Rx(roll)
```
（矩阵从右往左应用，先 roll，再 pitch，再 yaw，但三根轴始终是父坐标系的轴，不随旋转移动）

这和 Euler Angles（intrinsic）的区别：

| | Fixed Angles（URDF 用的） | Euler Angles（intrinsic）|
|---|---|---|
| 旋转轴 | 始终是**父坐标系**的固定轴 | 每次旋转后轴**随着物体转动** |
| 旋转顺序 | 先 X 后 Y 后 Z（轴不变） | 先绕自身X后绕自身Y后绕自身Z |
| 数学等价 | `Rz·Ry·Rx`（右乘顺序） | `Rx·Ry'·Rz''`（每次用新轴）|
| 别名 | extrinsic XYZ | intrinsic XYZ |

**重要等价关系**：

固定坐标系下先 X 后 Y 后 Z = 绕自身坐标系先 Z 后 Y 后 X

也就是说，`rpy="r p y"` 这个 Fixed Angles 表示，和 intrinsic ZYX Euler Angles（先绕自身Z旋转y、再绕自身Y旋转p、再绕自身X旋转r）的结果完全相同。

**用本文件的例子验证**：`joint2` 的 `rpy="1.5708 -0.1359 -3.1416"`
- 第一步：绕**固定X轴**转 1.5708 rad（90°）
- 第二步：绕**固定Y轴**（仍是原始父坐标系的Y轴，没有跟着第一步动）转 -0.1359 rad（-7.78°）
- 第三步：绕**固定Z轴**（仍是原始父坐标系的Z轴）转 -3.1416 rad（-180°）

---

问题二见05（solidworks to urdf有关的问题）
---

## 问题三：四种 joint 的 origin 含义是否相同？旋转轴位置怎么确定？

### 含义完全相同

四种 joint 类型（revolute / continuous / fixed / prismatic）的 `<origin>` 含义**完全一致**：

> **`<origin xyz rpy>`** 描述的是：**child link 坐标系的原点和朝向，在 parent link 坐标系中的位置和旋转。**

用变换矩阵表示：`T_parent_to_child = Translation(xyz) · Rotation(rpy)`

这个变换定义了 child 坐标系的「零位姿」（关节在默认位置 0 时的状态）。

### 旋转轴/平移轴的位置如何确定？

这是理解 URDF 的关键点，**旋转轴/平移轴永远穿过 child link 坐标系的原点**。

也就是说：
- `<origin xyz>` 决定了轴穿过哪个点（child 坐标系原点在 parent 坐标系中的位置）
- `<axis xyz>` 决定了轴的方向（在 child 坐标系中）
- 两者合在一起，才完整确定旋转轴在空间中的位置和方向

用 `fl_hipx_joint` 举例：

```xml
<joint name="fl_hipx_joint" type="revolute">
  <origin xyz="0.3141 0.0685 0" rpy="0 0 0"/>
  <!-- child坐标系原点在parent坐标系的 (0.3141, 0.0685, 0) 处 -->
  <parent link="base_link"/>
  <child link="fl_hipx"/>
  <axis xyz="-1 0 0"/>
  <!-- 旋转轴方向是child坐标系的 -X 方向 -->
  <!-- 因为rpy=0，child坐标系和parent方向相同，所以就是世界-X方向 -->
</joint>
```

综合解读：
- 旋转轴的**穿过点**：parent坐标系中的 (0.3141, 0.0685, 0)（即前左腿安装位置）
- 旋转轴的**方向**：-X 轴（腿向内/外侧摆动的轴）
- 所以这根轴就是：过点 (31.4cm, 6.85cm, 0)，方向沿 -X，这根无穷长的线

**对比 prismatic joint（joint7 夹爪手指）：**

```xml
<joint name="joint7" type="prismatic">
  <origin xyz="0 0 0.1358" rpy="1.5708 0 0"/>
  <!-- child坐标系原点在parent的正前方13.58cm处，并绕X旋转90° -->
  <parent link="gripper_base"/>
  <child link="link7"/>
  <axis xyz="0 0 1"/>
  <!-- 平移方向是child坐标系的+Z方向 -->
  <!-- 但child坐标系已绕X转了90°，所以child的+Z = parent的+Y方向 -->
</joint>
```

prismatic 的 origin 含义完全一样，只不过 `<axis>` 描述的不是旋转轴而是平移方向。**轴同样穿过 child 坐标系原点**（对平移关节来说，「轴穿过哪个点」其实无所谓，因为平移轴是一个方向而不是一根直线，但惯例上仍以 child 原点为起点描述方向）。

---

## 问题四：prismatic joint 的 velocity 是线速度还是角速度？

**是线速度，单位是 m/s**，不是角速度。

URDF 规范对 `<limit>` 里各参数的单位定义如下：

| joint 类型 | `lower`/`upper` 单位 | `effort` 单位 | `velocity` 单位 |
|---|---|---|---|
| revolute | rad（弧度） | N·m（牛顿·米，力矩） | rad/s（角速度） |
| continuous | 无限位，N/A | N·m | rad/s |
| prismatic | m（米） | N（牛顿，力） | **m/s（线速度）** |
| fixed | N/A | N/A | N/A |

所以 `joint7` 的 limit：

```xml
<limit lower="0" upper="0.035" effort="10" velocity="1"/>
```

- `lower="0"`：最小位移 0 m（完全闭合）
- `upper="0.035"`：最大位移 0.035 m = 3.5 cm（最大张开）
- `effort="10"`：最大力 **10 N**（不是力矩，是线性力）
- `velocity="1"`：最大线速度 **1 m/s**（手指张合的速度）

看起来 `velocity="1"` 像是角速度，其实完全是合理的线速度——夹爪手指以最快 1 m/s 的速度滑动（3.5cm 行程在 35ms 内走完），对于一个小型夹爪来说是合理的运动速度。

这里容易混淆是因为 revolute 和 prismatic 的 `effort`/`velocity` 参数名相同，但物理单位和含义完全不同，URDF 规范约定了根据 joint 类型来解读。


---

# 在isaacsim界面中有没有办法看到这里的origin定义的坐标系
## 方法三：使用 Transform Gizmo 直接看坐标系

1. 在 Stage 面板里点击选中某个 link（例如 `link2`）
2. 按键盘 __W__（移动 gizmo）或 __E__（旋转 gizmo）
3. 视图里会出现三色箭头/圆弧，这个 gizmo 的原点和方向就是该 link 坐标系（即 joint origin 定义的坐标系）
4. 可以和其他 link 的 gizmo 对比，直观看出旋转关系

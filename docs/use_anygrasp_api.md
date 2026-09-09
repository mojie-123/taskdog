## 项目中 AnyGrasp 的完整使用解析


### 一、AnyGrasp 是什么？

AnyGrasp 是上海交大发布的一个**抓取姿态检测系统**，核心论文是《AnyGrasp: Robust and Efficient Grasp Perception in Spatial and Temporal Domains》。

它的输入是一个三维点云（N×3 的 XYZ 坐标数组），输出是若干个候选抓取姿态，每个姿态包含：

- `translation`：抓取点的 3D 位置（相机坐标系下，单位：米）
- `rotation_matrix`：3×3 旋转矩阵（夹爪的朝向）
- `width`：夹爪张开宽度
- `score`：置信度分数

---

### 二、安装流程

整个安装分为五个步骤：

#### 步骤一：克隆 SDK 仓库

```bash
cd /home/mojie
git clone https://github.com/graspnet/anygrasp_sdk.git
cd anygrasp_sdk
```

克隆后的目录结构：
```
/home/mojie/anygrasp_sdk/
├── grasp_detection/
│   ├── gsnet_versions/          ← 各 Python 版本的 gsnet.so 二进制文件
│   │   ├── gsnet.cpython-310-x86_64-linux-gnu.so
│   │   ├── gsnet.cpython-311-x86_64-linux-gnu.so  ← 本项目使用
│   │   ├── gsnet.cpython-312-x86_64-linux-gnu.so
│   │   └── ...（支持 3.6 ~ 3.14）
│   ├── gsnet.so                 ← 从 gsnet_versions/ 复制过来的，实际使用的文件
│   ├── license/                 ← License 文件夹放这里
│   ├── checkpoint_detection.tar ← 模型权重
│   ├── demo.py
│   └── demo.sh
├── license_registration/
│   ├── gsnet.so                 ← 用于获取 feature_id 的临时副本
│   └── license_JieMo/          ← 申请到的 License 文件
│       ├── licenseCfg.json
│       ├── JieMo.public_key
│       ├── JieMo.signature
│       └── JieMo.lic
├── pointnet2/
└── requirements.txt
```

#### 步骤二：编译安装 MinkowskiEngine（最关键、最复杂）

MinkowskiEngine 是 AnyGrasp 的核心神经网络依赖，它是一个**稀疏卷积库**，用 C++/CUDA 编写，需要从源码编译。

```bash
conda activate env_isaaclab
conda install openblas-devel -c anaconda -y   # 编译依赖的数学库

cd /home/mojie/anygrasp_sdk
mkdir -p dependencies && cd dependencies
git clone https://github.com/chenxi-wang/MinkowskiEngine.git  # 官方修改版
cd MinkowskiEngine

# 切换到 CUDA 12.x 专用分支（本项目 PyTorch 用 CUDA 12.8）
git checkout cuda-12-1

# CUDA 12.8 需要的系统头文件补丁（修复 C++ 标准库兼容问题）
sed -i 's/\bauto __raw = __to_address(__r.get());/auto __raw = std::__to_address(__r.get());/' \
    /usr/include/c++/11/bits/shared_ptr_base.h

# ⚠️ 关键：CUDA_HOME 必须指向 conda 里的 CUDA 12.8，不能用系统的 /usr/local/cuda（11.8）
export CUDA_HOME=/home/mojie/anaconda3/envs/env_isaaclab

# 编译并安装（需要几分钟）
python setup.py install \
    --blas_include_dirs=${CONDA_PREFIX}/include \
    --blas_library_dirs=${CONDA_PREFIX}/lib \
    --blas=openblas
```

**为什么需要这个补丁？** CUDA 12.8 使用的 C++ 标准库中，`__to_address` 的命名空间发生了变化，直接用 `auto __raw = __to_address(...)` 会导致编译器找不到该函数，加上 `std::` 前缀就能解决。

#### 步骤三：安装其余 Python 依赖

```bash
cd /home/mojie/anygrasp_sdk
pip install -r requirements.txt   # numpy, Pillow, scipy, tqdm, open3d

cd pointnet2
python setup.py install           # PointNet++ 点云处理模块

pip install graspnetAPI           # GraspNet 数据格式 API
```

#### 步骤四：复制对应 Python 版本的 gsnet.so

```bash
cd /home/mojie/anygrasp_sdk/grasp_detection

# 本项目用 Python 3.11，复制对应版本
cp gsnet_versions/gsnet.cpython-311-x86_64-linux-gnu.so gsnet.so
```

**gsnet.so 是什么？** 这是 AnyGrasp 核心推理引擎的预编译二进制文件（`.so` 是 Linux 下的动态链接库，相当于 Windows 的 `.dll`）。它包含了神经网络推理代码，但源码不公开——这就是为什么需要 License 才能使用。不同 Python 版本的 ABI（应用二进制接口）不同，所以每个 Python 版本都有一个对应的 `.so` 文件。

---

### 三、License 申请与验证

#### 为什么需要 License？

AnyGrasp 的代码不开源，核心逻辑全在 `gsnet.so` 这个预编译二进制里。为了防止随意转发和商业滥用，官方采用**硬件指纹绑定**的 License 机制：License 与申请机器的硬件特征绑定，复制到其他机器无法使用。

#### License 申请流程（4步）

**第1步：获取本机硬件指纹（feature_id）**

```bash
conda activate env_isaaclab
cd /home/mojie/anygrasp_sdk/license_registration

# 复制 Python 3.11 对应的 gsnet.so（获取 feature_id 时也需要用）
cp ../grasp_detection/gsnet_versions/gsnet.cpython-311-x86_64-linux-gnu.so gsnet.so

# 调用 gsnet 内置函数读取本机硬件指纹
python -c "from gsnet import get_feature_id; print(get_feature_id())"
# 输出示例：N51790452892269025015
```

`feature_id` 是 `gsnet.so` 内部读取 CPU、主板等硬件信息生成的唯一字符串，格式是 `N` 开头的 20 位数字。

**第2步：填写申请表单**

访问 `https://forms.gle/XVV3Eip8njTYJEBo6`，填写邮箱、姓名、机构、用途说明和 feature_id，等待约 5 个工作日。

**第3步：收到 License zip 文件，解压**

收到的 zip 解压后得到（以本项目为例，申请人名叫 JieMo）：
```
license_JieMo/
├── licenseCfg.json        ← 配置文件，记录 feature_id 和各文件名
├── JieMo.public_key       ← RSA 公钥
├── JieMo.signature        ← 数字签名
└── JieMo.lic              ← License 文件本身
```

`licenseCfg.json` 内容：
```json
{
    "version": "3.1.0",
    "feature_id": "N51790452892269025015",
    "public_key": "JieMo.public_key",
    "signature": "JieMo.signature",
    "license": "JieMo.lic",
    "toolbox": [
        {"name": "Basic", "type": "PERMANENT", "data": ""}
    ]
}
```
`PERMANENT` 表示永久 License，不会过期。

**第4步：把 License 放到正确位置并验证**

```bash
# 把 license 文件夹放到 grasp_detection/ 下，命名为 license
cp -r license_JieMo /home/mojie/anygrasp_sdk/grasp_detection/license

# 验证 License 是否有效
cd /home/mojie/anygrasp_sdk/grasp_detection
python -c "from gsnet import check_license; check_license('license')"
```

**License 的验证原理**：`gsnet.so` 启动时读取当前目录下的 `license/` 文件夹，用 `licenseCfg.json` 找到各文件，然后：
1. 重新计算本机 feature_id
2. 用公钥验证签名，确认 License 是官方颁发的
3. 验证 feature_id 是否与 License 中记录的一致

三者都通过才允许运行，否则 `create_detector()` 返回 `None`。

**关键注意点**：`gsnet.so` 在加载 License 时，会在**当前工作目录**下查找 `license/` 子文件夹。所以 `grasp_worker.py` 第 30 行有这样一行代码：

```python
os.chdir(SDK_DETECTION_DIR)  # 切换到 /home/mojie/anygrasp_sdk/grasp_detection/
```

如果不切换工作目录，即使 License 文件存在，`gsnet.so` 也找不到它。

---

### 四、在项目中的实际使用方式

#### 整体架构：子进程隔离

AnyGrasp **不能**直接在 `navigate_to_goal.py`（父进程，运行着 Isaac Sim）里调用，原因是：
- Isaac Sim 已经占用了大量 GPU 显存和 CUDA 上下文
- MinkowskiEngine 是针对 numpy 2.4.6 编译的，而 Isaac Sim 环境用的是 numpy 1.26.0，如果直接 import 会导致 C ABI 不兼容崩溃

因此采用**子进程隔离**方案：

```
navigate_to_goal.py（父进程）
    │
    │  1. 把点云写入 /tmp/pointcloud.npz
    │  2. subprocess.run() 启动子进程，阻塞等待（最多120秒）
    ▼
grasp_worker.py（子进程，全新的独立 Python 进程）
    │  - 完全独立的 GPU 上下文，不受 Isaac Sim 影响
    │  - import gsnet → License 验证 → 加载模型
    │  - 读取 /tmp/pointcloud.npz
    │  - 推理，NMS，排序
    │  - 结果写入 /tmp/grasp_result.npz
    │  - 进程退出
    ▼
navigate_to_goal.py（父进程恢复）
    │  3. 读取 /tmp/grasp_result.npz
    │  4. 解析最佳抓取姿态，执行 IK + 手臂控制
```

#### GRASP_PLAN 状态的完整执行过程

在状态机里，`GRASP_PLAN` 状态做了以下事情：

**① 从 30 帧深度图构建点云**

```python
depth_med = np.median(np.stack(depth_accum, axis=0), axis=0)  # 30帧取中值，去噪
# 用针孔相机模型反投影：像素坐标 → 相机3D坐标
z    = depth_med
pts  = np.stack([
    (u - cx) * z / fx,   # X = (像素u - 光心cx) * 深度 / 焦距fx
    (v - cy) * z / fy,   # Y
    z,                    # Z = 深度
], axis=-1).astype(np.float32)
```

**② RANSAC 去除桌面**

用 RANSAC 算法自动找到点云中最大的平面（即桌面），将其剔除，只保留桌面上突出的物体点云。这是通用的做法，不依赖物体颜色。

**③ 把点云写入临时文件**

```python
np.savez("/tmp/pointcloud.npz", points=pts, colors=cols)
```

**④ 启动 grasp_worker.py 子进程**

```python
import subprocess, sys as _sys

worker = os.path.join(os.path.dirname(__file__), "grasp_worker.py")
res = subprocess.run(
    [_sys.executable,           # /home/mojie/anaconda3/envs/env_isaaclab/bin/python3
     worker,                    # grasp_worker.py 的绝对路径
     "--checkpoint", args.grasp_checkpoint,   # AnyGrasp 模型权重路径
     "--topk", str(50)],        # 最多返回 50 个候选抓取
    timeout=120,                # 最多等待 120 秒
)
```

**⑤ grasp_worker.py 内部（子进程里）**

```python
# 1. 切换工作目录（让 gsnet.so 能找到 license/）
os.chdir("/home/mojie/anygrasp_sdk/grasp_detection")

# 2. 先 import gsnet（必须在 import numpy 之前！）
from gsnet import create_detector

# 3. 再 import numpy（避免 ABI 冲突）
import numpy as np

# 4. 初始化检测器（内部进行 License 验证）
config = Namespace(
    checkpoint_path=args.checkpoint,   # checkpoint_detection.tar
    max_gripper_width=0.08,            # Piper 夹爪最大张开宽度 8cm
    gripper_height=0.06,               # 手指高度 6cm（用于碰撞检测）
)
detector = create_detector(config)    # License 验证失败则返回 None

# 5. 读点云、推理
points = np.load("/tmp/pointcloud.npz")["points"]
gg = detector.get_grasp(points, {
    "collision_detection": True,      # 过滤与点云碰撞的抓取
    "dense_grasp": False,             # 标准模式（非密集预测）
})

# 6. NMS + 排序 + 截取 topk
gg = gg.nms()
gg = gg.sort_by_score()
gg = gg[:50]

# 7. 结果写入临时文件
np.savez("/tmp/grasp_result.npz",
    translations=gg.translations,       # (50, 3)
    rotations=gg.rotation_matrices,     # (50, 3, 3)
    widths=gg.widths,                   # (50,)
    scores=gg.scores,                   # (50,)
)
```

**⑥ 父进程读取结果，执行颜色过滤**

子进程退出后，父进程读取 `/tmp/grasp_result.npz`，再用 HSV 颜色过滤（找香蕉的黄色像素对应的点云）进一步筛选最佳抓取，最终选出一个抓取姿态交给 IK 求解模块执行手臂控制。

#### 两种 checkpoint 的切换

`navigate_to_goal.py` 的 `--grasp_checkpoint` 参数支持两种模型：

```bash
# 使用 graspnet-baseline（开源版，用于开发期间）
--grasp_checkpoint /home/mojie/graspnet-baseline/logs/checkpoint-rs.tar
# 对应子进程：grasp_worker_graspnet_baseline.py

# 使用 AnyGrasp（性能更好，需要 License）
--grasp_checkpoint /home/mojie/anygrasp_sdk/grasp_detection/log/checkpoint_detection.tar
# 对应子进程：grasp_worker.py（当前版本）
```

两个版本的子进程与父进程之间的文件通信协议完全相同（输入 `/tmp/pointcloud.npz`，输出 `/tmp/grasp_result.npz`），所以切换时父进程代码零改动，只改子进程脚本约 15 行。
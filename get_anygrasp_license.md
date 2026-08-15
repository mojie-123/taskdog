# AnyGrasp License 申请完整指南

> 适用于本项目环境：Python 3.11、PyTorch 2.7.1+cu128、RTX 4060 Laptop 8GB

---

## 环境现状确认

```
Python:          3.11.15  ✅ AnyGrasp 支持
PyTorch:         2.7.1+cu128  ✅
CUDA（nvcc）:    11.8（系统）⚠️  但 PyTorch 用的是 CUDA 12.8
open3d:          0.19.0  ✅
MinkowskiEngine: 未安装  ❌ 需要安装（AnyGrasp 的核心依赖）
graspnetAPI:     未安装  ❌ 需要安装
```

> ⚠️ MinkowskiEngine 编译时 `CUDA_HOME` 必须指向 CUDA 12.8（与 PyTorch 匹配），
> 不是系统的 nvcc 11.8，安装时要格外注意。

---

## 申请流程总览

```
第1步：克隆 SDK 仓库              （现在就做，约 10 分钟）
   ↓
第2步：复制 gsnet.so → 获取 feature_id  （现在就做，约 5 分钟）
   ↓
第3步：填写申请表单，提交 feature_id    （现在就做，约 5 分钟）
   ↓
       等待约 5 个工作日（注意查看垃圾邮件）
   ↓
第4步：收到 .zip 邮件 → 安装依赖 → 验证 License
```

---

## 第1步：克隆 SDK 仓库

```bash
cd /home/mojie
git clone https://github.com/graspnet/anygrasp_sdk.git
cd anygrasp_sdk
```

克隆后的仓库结构：

```
anygrasp_sdk/
├── grasp_detection/
│   ├── gsnet_versions/          ← 各 Python 版本的 gsnet.so 二进制
│   │   ├── gsnet.cpython-311-x86_64-linux-gnu.so   ← 本项目使用此版本
│   │   └── ...
│   ├── demo.py / demo.sh
│   └── log/                     ← 模型权重放这里（License 批准后下载）
├── license_registration/
│   ├── README.md
│   └── sample_license/          ← license 格式示例
├── grasp_tracking/
├── pointnet2/
└── requirements.txt
```

---

## 第2步：获取 feature_id

feature_id 是绑定本机硬件的指纹，格式为 `N12345678900987654321`，申请表单必填。

```bash
conda activate env_isaaclab

cd /home/mojie/anygrasp_sdk/license_registration

# 复制 Python 3.11 对应的 gsnet.so
cp ../grasp_detection/gsnet_versions/gsnet.cpython-311-x86_64-linux-gnu.so gsnet.so

# 获取 feature_id
python -c "from gsnet import get_feature_id; print(get_feature_id())"
```

输出示例：`N12345678900987654321`

> ⚠️ 如果输出末尾有 `%`，填表时必须手动删掉。

---

## 第3步：填写申请表单

**申请地址（Google Form）：**
```
https://forms.gle/XVV3Eip8njTYJEBo6
```

| 字段 | 填写内容 |
|------|----------|
| 邮箱 | 用来接收 license .zip 文件 |
| 姓名 | 会出现在 license 文件名中 |
| 机构/学校 | 所在单位或学校 |
| 用途说明 | 例如：robot grasping research in Isaac Sim simulation |
| feature_id | 上一步的 N 开头字符串（去掉末尾 % 如果有） |

提交后等待约 **5 个工作日**，注意检查垃圾邮件。

---

## 第4步：收到 License 后的安装步骤

### 4.1 安装 MinkowskiEngine（最复杂的一步）

```bash
conda activate env_isaaclab

# 安装编译依赖
conda install openblas-devel -c anaconda -y

# 克隆修改版 MinkowskiEngine
cd /home/mojie/anygrasp_sdk
mkdir -p dependencies && cd dependencies
git clone https://github.com/chenxi-wang/MinkowskiEngine.git
cd MinkowskiEngine

# 切换到 CUDA 12.x 专用分支
git checkout cuda-12-1

# CUDA 12.8 专用补丁（先查实际路径）
find /usr/include -name 'shared_ptr_base.h' 2>/dev/null
# 通常在 /usr/include/c++/11/bits/ 或 /usr/include/c++/13/bits/
# 将路径替换成实际路径后执行：
sed -i 's/\bauto __raw = __to_address(__r.get());/auto __raw = std::__to_address(__r.get());/' \
    /usr/include/c++/11/bits/shared_ptr_base.h

# ⚠️ 关键：CUDA_HOME 必须指向 CUDA 12.8，不是系统的 /usr/local/cuda（11.8）
export CUDA_HOME=/home/mojie/anaconda3/envs/env_isaaclab

# 编译安装（需要几分钟）
python setup.py install \
    --blas_include_dirs=${CONDA_PREFIX}/include \
    --blas_library_dirs=${CONDA_PREFIX}/lib \
    --blas=openblas
```

### 4.2 安装其余依赖

```bash
cd /home/mojie/anygrasp_sdk
pip install -r requirements.txt

# 安装 pointnet2
cd pointnet2 && python setup.py install && cd ..

# 安装 graspnetAPI（graspnet-baseline 和 AnyGrasp 共用，装一次就够）
git clone https://github.com/graspnet/graspnetAPI.git
cd graspnetAPI && pip install . && cd ..
```

### 4.3 部署 License 文件

```bash
# 解压收到的 zip
unzip your_license.zip -d /home/mojie/anygrasp_sdk/license_registration/license
```

解压后目录结构：
```
license/
├── licenseCfg.json
├── yourname.public_key
├── yourname.signature
└── yourname.lic
```

```bash
# 验证 license 是否有效
cd /home/mojie/anygrasp_sdk/license_registration
python -c "from gsnet import check_license; check_license('license')"
# 输出 'license is valid' 表示成功

# 分别复制到 grasp_detection 和 grasp_tracking
cp -r license ../grasp_detection/license
cp -r license ../grasp_tracking/license
```

### 4.4 下载模型权重

AnyGrasp 模型权重**不包含在 GitHub 仓库里**，有两种获取方式：

1. **邮件附带**：审批邮件通常会附带权重下载链接
2. **GitHub Issue**：在 [anygrasp_sdk Issues](https://github.com/graspnet/anygrasp_sdk/issues) 搜索 `checkpoint` 或 `weights`

下载后放到：
```
/home/mojie/anygrasp_sdk/grasp_detection/log/
```

### 4.5 验证 Demo 运行

```bash
cd /home/mojie/anygrasp_sdk/grasp_detection
cp gsnet_versions/gsnet.cpython-311-x86_64-linux-gnu.so gsnet.so
bash demo.sh
```

---

## 等待 License 期间并行推进的工作

License 需要约 5 个工作日，期间可同步完成以下工作，不会被阻塞：

```
① 用 graspnet-baseline 先跑通完整流水线
   - 安装 graspnetAPI（pip install graspnetAPI）
   - 下载 graspnet-baseline 权重（约 60 MB .tar 文件）
   - 实现 grasp_worker.py 子进程脚本（graspnet-baseline 版）

② 在 single_piper_env_cfg.py 中添加 CameraCfg（腕部深度相机）

③ 在 navigate_to_goal.py 中实现 Phase 2 状态机
   （到达目标 → 暂停 Physics → 启动子进程 → 读取结果 → 执行 IK）

④ 实现 IK 求解模块（用 ikpy + Piper URDF）

⑤ License 到了 → 替换 grasp_worker.py 约 15 行代码即可切换到 AnyGrasp
```

---

## 关键注意事项

| 注意点 | 说明 |
|--------|------|
| **feature_id 不能换机器** | License 绑定硬件指纹，申请后不能迁移到其他机器 |
| **末尾的 % 要删掉** | 复制 feature_id 填表时，如有 `%` 需手动删除 |
| **MinkowskiEngine CUDA 版本** | 必须用 `cuda-12-1` 分支 + CUDA 12.8 补丁，不能用系统 nvcc 11.8 |
| **gsnet.so 版本匹配** | Python 3.11 → 必须用 `gsnet.cpython-311-x86_64-linux-gnu.so` |
| **邮件检查垃圾箱** | 官方明确提示：回复可能进垃圾邮件文件夹 |
| **模型权重单独获取** | 权重不在 GitHub 仓库中，需从邮件或 Issue 获取 |
| **License 不可共享** | 一个 feature_id 对应一个 License，不能给其他机器用 |
| **必须用新版 license 工具** | 2026年7月4日后，旧的 `lib_cxx.so` 和 `license_checker` 已移除，按本文档流程操作 |

---

## graspnet-baseline → AnyGrasp 迁移代码对比

License 批准后切换 AnyGrasp，只需修改子进程脚本约 **15 行**，主进程 `navigate_to_goal.py` **零改动**。

```python
# ===== graspnet-baseline 版（当前）=====
from graspnet import GraspNet, pred_decode
net = GraspNet(input_feature_dim=0, num_view=300, num_angle=12, num_depth=4,
               cylinder_radius=0.05, hmin=-0.02,
               hmax_list=[0.01,0.02,0.03,0.04], is_training=False)
checkpoint = torch.load('/path/to/checkpoint.tar')
net.load_state_dict(checkpoint['model_state_dict'])
net.eval()
end_points = net({'point_clouds': cloud_tensor})
ggarray = pred_decode(end_points)[0].cpu().numpy()
best = ggarray[0]
np.savez('/tmp/grasp_result.npz',
         translation=best[13:16], rotation=best[4:13].reshape(3,3),
         width=best[1], score=best[0])

# ===== AnyGrasp 版（License 到了之后替换上面内容）=====
from gsnet import AnyGrasp
cfgs = type('cfg', (), {
    'checkpoint_path': '/home/mojie/anygrasp_sdk/grasp_detection/log/checkpoint.tar',
    'max_gripper_width': 0.08,
    'gripper_height': 0.06,
    'top_down_grasp': False,
    'debug': False
})()
net = AnyGrasp(cfgs)
net.load_net(cfgs.checkpoint_path)
ggarray, _ = net.get_grasp(points, colors,
                            lims=[-0.5, 0.5, -0.5, 0.5, 0.0, 0.8])
best = ggarray[0]
np.savez('/tmp/grasp_result.npz',
         translation=best.translation, rotation=best.rotation_matrix,
         width=best.width, score=best.score)
```

数据通信文件格式（两个版本完全相同，主进程读取代码无需改动）：
```
输入：/tmp/pointcloud.npz   → points (N,3), colors (N,3)
输出：/tmp/grasp_result.npz → translation (3,), rotation (3,3), width, score
```

---

## 参考链接

- AnyGrasp SDK：https://github.com/graspnet/anygrasp_sdk
- License 申请表单：https://forms.gle/XVV4SBu4TKHY3vMP9
- MinkowskiEngine（修改版）：https://github.com/chenxi-wang/MinkowskiEngine
- graspnetAPI：https://github.com/graspnet/graspnetAPI
- AnyGrasp 论文：https://arxiv.org/abs/2212.08333

# 对于我们环境中安装的rsl_rl，我们是不是没有办法修改它里面的内容，只能直接使用？那我们是不是可以在具体项目中继承rsl_rl中的某个模块进行改写？有没有什么常见的用法？对于pip install -e .这种方式来说，pyproject.toml 或 setup.py记录了怎样的包的信息？当我们在同一个conda环境（如仍用env_isaaclab）但是rsl_rl要使用其他版本时，是不是可以用到刚刚讲过的PYTHONPATH来指定？



## 一、pip install 安装的 rsl_rl 能修改吗？怎么用继承改写？

### 技术上可以直接改，但不推荐

pip 安装的 rsl_rl 代码文件就在 site-packages 里，是普通的 `.py` 文件，用编辑器直接打开修改是可以的：

```
/home/mojie/anaconda3/envs/env_isaaclab/lib/python3.11/site-packages/rsl_rl/
    runners/on_policy_runner.py   ← 可以直接改这个文件
```

但这样做有很大的问题：
- `pip install --upgrade rsl-rl-lib` 升级时会把你的修改全部覆盖掉
- 团队里其他人的环境不会有你的修改，行为不一致
- 没有版本控制，改坏了很难回溯

**推荐做法是用 Python 继承机制在项目里改写**，我们项目里就是这样做的。

### 我们项目里的实际用法：继承配置类

从 `rsl_rl_ppo_cfg.py` 可以看到，我们用的是**继承 `RslRlOnPolicyRunnerCfg`** 来定制训练配置：

```python
# isaaclab_rl 提供的基类（对 rsl_rl 配置的封装）
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg

@configclass
class DeeproboticsM20ProRoughPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """PPO runner config for M20 Pro rough terrain."""
    num_steps_per_env = 24          # 覆盖父类默认值
    max_iterations = 20000          # 覆盖
    experiment_name = "deeprobotics_m20pro_rough"  # 覆盖
    policy = RslRlPpoActorCriticCfg(
        actor_hidden_dims=[512, 256, 128],   # 定制网络结构
    )
    algorithm = RslRlPpoAlgorithmCfg(
        learning_rate=1.0e-3,       # 定制超参数
    )
```

然后还有第二层继承——从 Rough 配置继承出 Flat 配置：

```python
@configclass
class DeeproboticsM20ProFlatPPORunnerCfg(DeeproboticsM20ProRoughPPORunnerCfg):
    """PPO runner config for M20 Pro flat terrain."""
    def __post_init__(self):
        super().__post_init__()          # 先执行父类初始化
        self.max_iterations = 5000       # 只改需要改的字段
        self.experiment_name = "deeprobotics_m20pro_flat"
```

这样 `FlatPPORunnerCfg` 继承了 `RoughPPORunnerCfg` 的所有配置，只改了两个字段，其余（网络结构、学习率等）完全不用重复写。

### 如果要修改训练逻辑本身（不只是配置）

继承 `OnPolicyRunner` 类，覆盖你想改的方法：

```python
from rsl_rl.runners import OnPolicyRunner

class MyCustomRunner(OnPolicyRunner):
    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        # 在父类训练循环基础上加入自己的逻辑
        # 比如加入课程学习、自定义日志等
        super().learn(num_learning_iterations, init_at_random_ep_len)
        # 或者完全重写
```

然后在 `train.py` 里把 `OnPolicyRunner` 换成 `MyCustomRunner` 即可，rsl_rl 本身的代码完全不用动。

---

## 二、`setup.py` / `pyproject.toml` 记录了什么信息？

从我们找到的 `train_unitree_IRMV-master/rsl_rl/setup.py` 来看：

```python
from setuptools import setup, find_packages

setup(
    name='rsl_rl',              # ← pip 安装名（pip install rsl_rl）
    version='1.0.2',            # ← 版本号（metadata.version() 读的就是这里）
    author='Nikita Rudin',      # ← 作者信息
    author_email='rudinn@ethz.ch',
    license="BSD-3-Clause",     # ← 开源协议
    packages=find_packages(),   # ← 自动扫描所有含 __init__.py 的目录作为包
    description='Fast and simple RL algorithms implemented in pytorch',
    python_requires='>=3.6',    # ← Python 版本要求
    install_requires=[          # ← 依赖列表（pip 会自动安装这些）
        "torch>=1.4.0",
        "torchvision>=0.5.0",
        "numpy>=1.16.4"
    ],
)
```

`pip install -e .` 执行时，pip 读取这个文件，了解：

| 字段 | pip 用它来做什么 |
|------|----------------|
| `name` | 注册包名，`pip show rsl_rl` 显示的名字 |
| `version` | 记入 dist-info/METADATA，供 `metadata.version()` 查询 |
| `packages=find_packages()` | 知道哪些目录是要安装的包（扫描含 `__init__.py` 的目录） |
| `install_requires` | 自动安装依赖，版本不满足时报错或升级 |
| `python_requires` | 检查当前 Python 版本是否兼容 |

`pyproject.toml` 是更现代的格式，内容类似但写法不同，本质上记录的是同样的信息。新版 rsl_rl（3.0.1）就用的是 `pyproject.toml`。

---

## 三、同一 conda 环境中使用不同版本的 rsl_rl，能用 PYTHONPATH 吗？

**可以，这正是 PYTHONPATH 的典型用途之一。** 步骤如下：

### 前提：准备好另一个版本的 rsl_rl 源码

```bash
cd /home/mojie
git clone https://github.com/leggedrobotics/rsl_rl.git rsl_rl_v2
cd rsl_rl_v2
git checkout v2.x.x   # 切换到你想用的版本
# 不 pip install，只是把源码放在这里
```

### 临时使用旧版本（只对这次运行生效）

```bash
PYTHONPATH=/home/mojie/rsl_rl_v2 python3 train.py --task Rough-M20-v0 --headless
```

Python 搜索路径变成：
```
[0] 当前脚本目录
[1] rsl_rl_v2/          ← PYTHONPATH 插在这里，优先级高于 site-packages
[2] 标准库
[3] site-packages/      ← 这里的 rsl_rl 3.0.1 被遮蔽了
```

`import rsl_rl` 时，Python 先在 `rsl_rl_v2/` 里找到了 `rsl_rl/` 目录，直接用它，site-packages 里的 3.0.1 版本被完全跳过。

### 与 `pip install -e` 的区别

| 方式 | 效果 | 适用场景 |
|------|------|--------|
| `PYTHONPATH=/path/to/rsl_rl python3 ...` | 只对这一条命令生效，不改动环境 | 临时测试某个版本 |
| `export PYTHONPATH=/path/to/rsl_rl` | 当前终端会话都生效 | 调试阶段频繁切换 |
| `pip install -e /path/to/rsl_rl` | 永久改变环境里的 rsl_rl | 长期使用某个修改版 |
| 新建 conda 环境 | 完全隔离 | 不同项目用不同版本 |

**注意一个陷阱**：PYTHONPATH 只是让 Python 找到代码，不会改变 `metadata.version("rsl-rl-lib")` 的返回值。`metadata.version()` 读的是 dist-info 元数据，和 PYTHONPATH 无关。所以如果 `train.py` 做了版本检测：

```python
installed_version = metadata.version("rsl-rl-lib")   # 仍然返回 "3.0.1"
```

即使你用 PYTHONPATH 指向了旧版本的代码，版本检测这里也不会察觉，它还是读 site-packages 里的元数据。这是用 PYTHONPATH 切换版本时需要注意的地方。


# isaaclab/isaacsim是怎么安装的，和rsl_rl一样吗？（我应该没有删除相关的文件，可以搜索一下，以及可以查找相关网页）详细讲解一下


## Isaac Sim / Isaac Lab 的安装方式详解

### 一、两者都是用 pip 安装的，和 rsl_rl 一样

从 dist-info 元数据可以确认：

```
isaacsim-5.1.0.0.dist-info/INSTALLER    → pip
isaaclab-2.3.2.post1.dist-info/INSTALLER → pip
```

两者都是标准 pip 安装，和 rsl_rl 的安装机制完全相同。但有几个关键的不同点，下面逐一解释。

---

### 二、Isaac Sim 的安装：metapackage + 私有 PyPI 源

Isaac Sim 的 METADATA 里写着：
```
Summary: A metapackage to manage the Isaac Sim installation
```

**`isaacsim` 本身是一个「元包（metapackage）」**，它自身几乎没有代码，作用是把几十个子包用依赖关系串联起来，一条命令安装全部：

```
isaacsim（元包，只有依赖声明）
  └── Requires-Dist: isaacsim-kernel==5.1.0.0
  └── [all] isaacsim-app==5.1.0.0
  └── [all] isaacsim-core==5.1.0.0
  └── [all] isaacsim-rl==5.1.0.0
  └── [all] isaacsim-robot==5.1.0.0
  └── [all] isaacsim-sensor==5.1.0.0
  └── [all] isaacsim-gui==5.1.0.0
  └── [extscache] isaacsim-extscache-kit==5.1.0.0   ← 扩展缓存（omni.usd.libs 在这里）
  └── ... 共十几个子包
```

你运行一条命令：
```bash
pip install isaacsim[all,extscache]==5.1.0.0 --extra-index-url https://pypi.nvidia.com
```

pip 实际上会安装十几个子包，每个子包里才是真正的代码。`[all,extscache]` 叫做**「可选依赖组（extras）」**，方括号里指定想安装哪些组。

**为什么需要 `--extra-index-url https://pypi.nvidia.com`？**

Isaac Sim 不在公开的 PyPI 上（`pypi.org`），而是在 NVIDIA 自己的私有 PyPI 服务器上。pip 默认只去 `pypi.org` 找包，找不到就报错。`--extra-index-url` 告诉 pip「除了官方 PyPI，还要去这个地址找」：

```
默认：pypi.org              → 找到 numpy、torch 等公开包
额外：pypi.nvidia.com       → 找到 isaacsim、isaacsim-core 等 NVIDIA 私有包
```

---

### 三、Isaac Lab 的安装：依赖 isaacsim，并声明了可选 RL 框架

Isaac Lab 的 METADATA 关键部分：

```
Name: isaaclab
Version: 2.3.2.post1
Requires-Python: ==3.11.*          ← 严格要求 Python 3.11，其他版本不行
Requires-Dist: numpy<2             ← 必须是 numpy 1.x，不能用 2.x
Requires-Dist: torch>=2.7
Requires-Dist: gymnasium==1.2.0
Requires-Dist: hydra-core         ← train.py 里的 @hydra_task_config 就来自这里
Requires-Dist: warp-lang          ← NVIDIA Warp，GPU 加速的物理计算库
...

Provides-Extra: isaacsim
Requires-Dist: isaacsim[all,extscache]==5.1.0.*; extra == "isaacsim"

Provides-Extra: rsl-rl
Requires-Dist: rsl-rl-lib==3.0.1; extra == "rsl-rl"   ← 指定了 rsl_rl 的版本

Provides-Extra: all
Requires-Dist: stable-baselines3>=2.6; extra == "all"
Requires-Dist: skrl>=1.4.3; extra == "all"
Requires-Dist: rsl-rl-lib==3.0.1; extra == "all"
```

可以看到 `rsl-rl-lib==3.0.1` 正是 Isaac Lab 的可选依赖之一，这也解释了为什么 `train.py` 要检测 rsl_rl 版本必须是 3.0.1——因为 Isaac Lab 2.3.2 就是专门配套这个版本的。

安装命令有两种写法：

```bash
# 方式A：分开安装（先装 isaacsim，再装 isaaclab）
pip install isaacsim[all,extscache]==5.1.0.0 --extra-index-url https://pypi.nvidia.com
pip install isaaclab==2.3.2

# 方式B：一条命令，让 isaaclab 的依赖声明自动触发 isaacsim 安装
pip install isaaclab[isaacsim,rsl-rl]==2.3.2 --extra-index-url https://pypi.nvidia.com
# 等价于同时安装 isaaclab 本体 + isaacsim[all,extscache] + rsl-rl-lib==3.0.1
```

---

### 四、我们环境的实际安装流程还原

根据 INSTALLER 文件（两者都是 `pip`）、无 `direct_url.json`（说明不是本地 `-e` 安装）、以及 `basic_work.md` 里的记录，完整流程是：

**第1步：创建 conda 环境**
```bash
conda create -n env_isaaclab python=3.11 -y
conda activate env_isaaclab
```

为什么用 conda 而不是系统 Python？
- conda 提供了完全隔离的环境，不影响系统 Python
- conda 能同时管理 Python 包和 C 库（如 CUDA 相关的 `.so` 文件）
- Isaac Sim 严格要求 Python 3.11，conda 可以精确指定版本

**第2步：安装 Isaac Sim**
```bash
pip install isaacsim[all,extscache]==5.1.0.0 \
    --extra-index-url https://pypi.nvidia.com
```

pip 做了以下事情：
1. 去 `pypi.nvidia.com` 查询 `isaacsim` 元包
2. 解析元包的依赖，得到十几个子包的列表
3. 逐个下载这些子包的 `.whl` 文件（总大小约 10-20 GB）
4. 解压到 site-packages：
   ```
   site-packages/
     isaacsim/            ← 元包本身（很小）
     isaacsim/extscache/  ← 扩展缓存，包括 omni.usd.libs
     omni/                ← Omniverse 核心模块
     carb/                ← Carbonite 运行时框架
     ...
   ```

**第3步：安装 Isaac Lab**
```bash
pip install isaaclab==2.3.2
```

pip 安装 `isaaclab` 本体及其所有 `Requires-Dist` 依赖（numpy、torch、gymnasium、hydra-core 等）。

**第4步：安装 rl_training（项目专属，用 `-e` 可编辑安装）**
```bash
cd /home/mojie/taskdog/deps/rl_training
python -m pip install -e source/rl_training
```

这里用了 `-e` 可编辑安装，因为 `rl_training` 是我们项目自己的代码（包含任务定义、环境配置等），需要随时修改调试，改完立即生效，不用重新安装。

---

### 五、和 rsl_rl 安装方式的对比总结

| 对比项 | rsl_rl | Isaac Sim | Isaac Lab | rl_training |
|--------|--------|-----------|-----------|-------------|
| 安装方式 | `pip install rsl-rl-lib` | `pip install isaacsim[all,extscache]` | `pip install isaaclab` | `pip install -e source/rl_training` |
| PyPI 来源 | 公开 pypi.org | NVIDIA 私有 pypi.nvidia.com | 公开 pypi.org | 本地源码 |
| 包类型 | 普通包 | 元包（metapackage） | 普通包（含可选依赖组） | 可编辑安装 |
| 安装后大小 | 几 MB | 10~20 GB | 几十 MB | 指针，不复制文件 |
| 能直接改代码吗 | 不推荐 | 不推荐 | 不推荐 | ✅ 可以，改完立即生效 |

**核心机制完全相同**：都是 pip 把文件放进 site-packages，创建 dist-info 元数据目录。区别只在于包的来源（官方 PyPI vs NVIDIA PyPI vs 本地）和是否用 `-e` 模式。
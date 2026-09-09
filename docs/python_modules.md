# Python 查找模块的完整顺序

Python 执行 `import xxx` 时，按照以下**固定顺序**依次查找，找到就立刻停止：

**第 0 步（最先）：检查缓存 `sys.modules`**

`sys.modules` 是一个字典，记录了所有已经加载过的模块。如果 `xxx` 之前已经 import 过了，直接从缓存里取，根本不去磁盘找。这也是为什么同一个模块 import 两次，第二次几乎没有开销。

**第 1 步：内置模块（built-in modules）**

Python 有一批用 C 语言写死在解释器里的模块，比如 `sys`、`os`、`builtins`。这些模块不在磁盘上，直接从解释器内存里取。

python解释器：一个可执行文件。如/home/mojie/anaconda3/envs/env_isaaclab/bin/python3

解释器内存：内置模块（如 `sys`、`math`）的 C 代码在编译时就被直接打包进了 `python3` 这个可执行文件里。当操作系统把 `python3` 加载进内存时，这些 C 代码随之进入内存。所以叫「解释器内存」——它们和解释器本身共享同一块内存，不需要从磁盘上单独加载文件。

__激活不同的 conda 环境，本质上就是切换了 `python3` 命令指向哪个可执行文件__。


```python
import sys   # 不需要找任何文件，解释器内置的
```

**第 2 步：按顺序搜索 `sys.path` 列表里的每个目录**

`sys.path` 是一个列表，Python 从列表第 0 项开始，逐个目录查找 `xxx.py` 或 `xxx/` 包目录，找到就停。

`sys.path` 的**初始内容**（按顺序）大致是：

| 顺序 | 内容 | 来源 |
|------|------|------|
| 第 0 项 | 当前脚本所在目录（或空字符串 `""` 表示当前工作目录） | Python 自动添加 |
| 第 1~N 项 | `PYTHONPATH` 环境变量里的目录 | 用户设置的环境变量 |
| 后面几项 | Python 标准库目录 | Python 安装时确定 |
| 最后几项 | site-packages（第三方包目录） | pip install 安装的包放在这里 |

**所以「本脚本所在目录」就是 `sys.path` 的第 0 项**，它本来就在 `sys.path` 里，并不是独立于 `sys.path` 之外的特殊地方。`sys.path.append(某目录)` 是把新目录加到列表**末尾**，所以自定义添加的目录优先级比脚本所在目录**低**。

用一张图来表示完整的查找顺序：

```
import cli_args
    ↓
① sys.modules 缓存里有吗？ → 有：直接用，结束
    ↓ 没有
② 是内置模块吗（sys/os等）？ → 是：直接用，结束
    ↓ 不是
③ 遍历 sys.path 列表：
    [0] 脚本所在目录（rsl_rl/）         → 有 cli_args.py 吗？没有，继续
    [1] PYTHONPATH 里的目录（若有）      → 有 cli_args.py 吗？没有，继续
    [2] 标准库目录                       → 有 cli_args.py 吗？没有，继续
    [3] site-packages                    → 有 cli_args.py 吗？没有，继续
    [4] reinforcement_learning/（append的）→ 有 cli_args.py 吗？有！加载它，结束
    ↓ 所有目录都没找到
④ 抛出 ModuleNotFoundError
```

**关键结论**：「本脚本所在目录」不是什么特殊的独立搜索位置，它就是 `sys.path[0]`，是 `sys.path` 的第一项，所以它的优先级最高（内置模块除外）。`sys.path.append()` 加到末尾，优先级最低。


---


# 一、PYTHONPATH 环境变量是什么？

`PYTHONPATH` 是操作系统级别的环境变量（不是 Python 专属的语法），专门用来告诉 Python 解释器「除了默认的搜索路径之外，还要去哪些额外的目录里找模块」。

你之前说的 `export PYTHONPATH=xxx` 就是在 shell 里设置这个环境变量。Python 启动时会自动读取它，并把里面的路径插入到 `sys.path` 的前部（排在脚本目录后面，标准库前面）。

### 具体例子（来自本项目）

在 `docs/add_collision.md` 里有这样一条命令：

```bash
PXR_PATH=$CONDA_PREFIX/lib/python3.11/site-packages/isaacsim/extscache/omni.usd.libs-xxx

LD_LIBRARY_PATH=$PXR_PATH/bin:$CONDA_PREFIX/lib:$LD_LIBRARY_PATH \     # 冒号`:` 是目录分隔符（Windows 上是分号 `;`），代表「依次在这些目录里找」。
PYTHONPATH=$PXR_PATH \
python3 custom_envs/scripts/assets/patch_gripper_finger_collision.py
```

这里$CONDA_PREFIX取决于激活的conda环境（如/home/mojie/anaconda3/envs/env_isaaclab）

这里 `PYTHONPATH=$PXR_PATH` 的意思是：**只在这一条命令执行期间**，临时把 `$PXR_PATH` 加入 Python 的搜索路径，让脚本能找到 `pxr`（Pixar USD 库）这个模块。

这种写法（`KEY=VALUE 命令`）不会永久修改环境变量，只对这一次 `python3 ...` 的执行生效。

`LD_LIBRARY_PATH` 是 Linux 下的环境变量，作用与 `PYTHONPATH` 类似，但针对的不是 Python 模块，而是 __C/C++ 动态链接库（`.so` 文件）__。

当一个程序运行时，如果它依赖某个 `.so` 库文件（比如 `libusd_ms.so`），Linux 的动态链接器会按顺序在以下位置查找：

1. `LD_LIBRARY_PATH` 里的目录
2. 系统默认库目录（`/usr/lib`、`/usr/local/lib` 等）
3. `/etc/ld.so.conf` 里配置的目录

如果找不到就报错：`error while loading shared libraries: libXXX.so: cannot open shared object file`


### `export PYTHONPATH=xxx` 和上面有什么区别？

```bash
# 方式1：只影响当前这一条命令
PYTHONPATH=/some/path python3 train.py

# 方式2：影响当前 shell 会话里所有后续命令
export PYTHONPATH=/some/path
python3 train.py      # 受影响
python3 play.py       # 也受影响
# 关闭终端后失效

# 方式3：写进 ~/.bashrc，每次打开终端都生效
echo 'export PYTHONPATH=/some/path' >> ~/.bashrc
```

本项目里 `export PYTHONPATH` 搜索结果为空，说明这个项目**没有用 PYTHONPATH 来管理模块路径**，而是用 `sys.path.append()` 或 pip 安装的方式。

### 什么时候使用PYTHONPATH
#### 场景一：模块在一个「奇怪的位置」，不是 pip 安装的

比如 Isaac Sim 把 USD 相关的 Python 绑定放在一个很深的子目录里：

```javascript
/home/mojie/anaconda3/envs/env_isaaclab/lib/python3.11/site-packages/
  isaacsim/
    extscache/
      omni.usd.libs-1.0.1+xxx.lx64.r.cp311/   ← pxr 模块在这里
        pxr/
          Usd.so
          Sdf.so
```

这个目录太深，pip 不认识它，Python 默认也不会搜索它。你想 `import pxr` 就必须手动告诉 Python 去这里找：

```bash
PYTHONPATH=/path/to/omni.usd.libs-xxx python3 my_script.py
```

这就是本项目里用到 PYTHONPATH 的原因。

#### 场景二：临时测试，不想 pip install

你写了一个库放在 `/home/mojie/mylib/`，想快速测试，不想正式安装：

```bash
export PYTHONPATH=/home/mojie/mylib
python3 test.py   # 可以 import mylib 了
```

#### 场景三：CI/CD 或 Docker 环境，无法修改代码

在自动化流水线里，修改脚本代码很麻烦，但设置环境变量很简单，PYTHONPATH 是最轻量的方案。

---

# 二、Python 标准库目录是什么？安装在哪里？

**标准库**是 Python 官方随解释器一起发布的内置模块集合，不需要 `pip install`，装完 Python 就有了。包括我们常用的 `os`、`sys`、`argparse`、`json`、`datetime`、`re` 等等。

在你的 `env_isaaclab` 环境里，标准库实际存放在：

```
/home/mojie/anaconda3/envs/env_isaaclab/lib/python3.11/
```

我们前面已经查到了这个目录的内容，里面全是 `.py` 文件：

```
abc.py         argparse.py    datetime.py
asyncio/       json/          os.py
random.py      re/            sys.py（实际是内置的）
logging/       pathlib.py     socket.py
...（共数百个）
```

这些就是标准库。你写 `import argparse` 时，Python 就是从这个目录里找到 `argparse.py` 并加载它。

### 和 PYTHONPATH 有什么区别？

| 对比项 | 标准库 | PYTHONPATH |
|--------|--------|-----------|
| 内容 | Python 官方自带的模块 | 你手动指定的额外目录 |
| 需要安装吗 | 不需要，装 Python 就有 | 不需要安装，只是指定路径 |
| 搜索顺序 | 排在 PYTHONPATH 后面 | 排在标准库前面 |
| 典型用途 | `os`、`argparse`、`json` | 临时让 Python 找到某个特殊目录里的模块 |

---

# 三、site-packages 是怎么安装的？

**site-packages 是所有 `pip install` 安装的第三方包的存放地**。在你的环境里，它的位置是：

```
/home/mojie/anaconda3/envs/env_isaaclab/lib/python3.11/site-packages/
```

我们已经查到这个目录里有什么（以你的 env_isaaclab 为例）：

```
isaaclab/                        ← isaaclab 的 Python 代码
isaaclab-2.3.2.post1.dist-info/  ← 安装元数据（版本、文件列表等）
isaacsim/                        ← Isaac Sim 的 Python 绑定
isaacsim-5.1.0.0.dist-info/
rsl_rl/                          ← RSL-RL 强化学习库
rsl_rl_lib-3.0.1.dist-info/
torch/                           ← PyTorch
torch-2.7.1+cu128.dist-info/
numpy/
gymnasium/
hydra/
...（还有很多）
```

### 以 Isaac Sim 的安装流程为具体例子

从我们查到的 `RECORD` 文件内容可以看出，`isaaclab` 的安装方式是标准的 pip 安装。整个 Isaac Sim + Isaac Lab 的安装大致流程是：

**第一步：创建 conda 环境，指定 Python 版本**
```bash
conda create -n env_isaaclab python=3.11
conda activate env_isaaclab
```
这会在 `/home/mojie/anaconda3/envs/env_isaaclab/` 下创建一套独立的 Python 环境。

**第二步：pip install 安装 Isaac Sim**
```bash
pip install isaacsim==5.1.0.0 --extra-index-url https://pypi.nvidia.com
```
pip 会把 Isaac Sim 的所有 Python 文件复制到：
```
/home/mojie/anaconda3/envs/env_isaaclab/lib/python3.11/site-packages/isaacsim/
```
同时创建元数据目录 `isaacsim-5.1.0.0.dist-info/`，记录版本号和文件列表，供 `pip show`、`pip uninstall` 等命令使用。

**第三步：pip install 安装 Isaac Lab**
```bash
pip install isaaclab==2.3.2
# 或者从源码安装：
pip install -e /path/to/isaaclab/source
```
结果是把 `isaaclab/` 目录放进 site-packages，并创建 `isaaclab-2.3.2.post1.dist-info/`。

**第四步：pip install 安装 RSL-RL**
```bash
pip install rsl-rl-lib==3.0.1
```
结果是 `rsl_rl/` 目录进入 site-packages，`rsl_rl_lib-3.0.1.dist-info/` 记录元数据。

### 为什么 `import isaaclab` 能直接找到？

因为 `isaaclab/` 目录就坐落在 site-packages 里，而 site-packages 在 Python 启动时会自动被加入 `sys.path`，所以任何时候写 `import isaaclab` Python 都能找到它，不需要 `sys.path.append` 也不需要 `PYTHONPATH`。


## pip是干什么的？它的具体行为是怎样的？能否结合rsl_rl的完整安装流程讲讲pip在其中发挥的作用（rsl_rl安装时需要pip install -e .）

---

### 一、pip 是什么？它具体做了什么？

**pip（Pip Installs Packages）是 Python 的包管理工具**，负责从网络下载、安装、卸载、升级 Python 包。它的核心工作就是把别人写好的 Python 库放到你的 site-packages 目录里，让你能 `import` 使用。

---

### 二、rsl_rl 的两种安装方式对比

METADATA 文件里明确写了两种安装方式：

```bash
# 方式一：从 PyPI 直接安装（我们的环境用的是这种）
pip install rsl-rl-lib

# 方式二：从源码安装（开发者用这种）
git clone https://github.com/leggedrobotics/rsl_rl
cd rsl_rl
pip install -e .
```

---

### 三、方式一：`pip install rsl-rl-lib` 的完整行为

我们环境里的 `rsl_rl` 就是这样安装的（INSTALLER 文件内容是 `pip`，且没有 `direct_url.json`，说明是从 PyPI 下载的）。

pip 执行了以下步骤：

#### 第1步：去 PyPI 查询包信息

PyPI（Python Package Index）是官方的包托管网站 `https://pypi.org`。pip 访问它的 API 查询 `rsl-rl-lib` 的最新版本和下载地址。

#### 第2步：下载 wheel 文件

pip 下载了一个 `.whl` 文件，比如：
```
rsl_rl_lib-3.0.1-py3-none-any.whl
```

wheel（`.whl`）文件本质上是一个 zip 压缩包，里面就是打包好的 Python 文件。`py3-none-any` 表示「纯 Python，不依赖特定平台」，解压就能用，不需要编译。

#### 第3步：检查并安装依赖

METADATA 文件里有 `Requires-Dist` 字段，记录了 rsl_rl 需要的依赖：

```
Requires-Dist: torch>=2.6.0
Requires-Dist: torchvision>=0.5.0
Requires-Dist: tensordict>=0.7.0
Requires-Dist: numpy>=1.16.4
Requires-Dist: GitPython
Requires-Dist: onnx
```

pip 检查这些依赖是否已经安装、版本是否满足，如果没有就自动递归安装它们。

#### 第4步：把文件解压复制到 site-packages

RECORD 文件完整记录了安装的所有文件，对应到磁盘上就是：

```
/home/mojie/anaconda3/envs/env_isaaclab/lib/python3.11/site-packages/
    rsl_rl/                          ← 实际可用的 Python 代码
        __init__.py
        algorithms/
            ppo.py                   ← PPO 算法实现
            distillation.py
        runners/
            on_policy_runner.py      ← OnPolicyRunner，train.py 里用到的
        modules/
            actor_critic.py          ← 神经网络定义
        storage/
            rollout_storage.py       ← 经验回放缓冲区
        utils/
            utils.py
    rsl_rl_lib-3.0.1.dist-info/      ← 安装元数据（pip 管理用）
        INSTALLER                    ← 内容："pip"
        METADATA                     ← 包的描述、依赖、版本等
        RECORD                       ← 所有已安装文件的清单 + 哈希值
        WHEEL                        ← wheel 格式信息
        top_level.txt                ← 内容："rsl_rl"（import 时用的包名）
        licenses/LICENSE
```

`top_level.txt` 的内容是 `rsl_rl`，这解释了一个关键问题：
- **pip 安装名**：`rsl-rl-lib`（连字符，PyPI 上的名字）
- **import 名**：`rsl_rl`（下划线，Python 代码里用的名字）

两个名字不同，但通过 `top_level.txt` 关联起来，`metadata.version("rsl-rl-lib")` 就是去找 `rsl_rl_lib-3.0.1.dist-info/METADATA` 里的 Version 字段。

#### 第5步：生成 `.pyc` 缓存文件

RECORD 里还有大量 `.pyc` 文件：
```
rsl_rl/algorithms/__pycache__/ppo.cpython-311.pyc
```

`.pyc` 是 Python 把 `.py` 源码预编译成字节码的缓存文件，下次 import 时直接加载字节码，不需要重新解析源码，速度更快。

---

### 四、方式二：`pip install -e .` 的完整行为（可编辑安装）

`-e` 是 `--editable` 的缩写，意思是「可编辑模式安装」。

假设你克隆了 rsl_rl 源码到 `/home/mojie/rsl_rl/`，然后：

```bash
cd /home/mojie/rsl_rl
pip install -e .
```

`.` 表示「当前目录」，pip 会读取当前目录下的 `pyproject.toml` 或 `setup.py` 来了解这个包的信息。

#### 与普通安装的关键区别

**普通安装**：把文件**复制**到 site-packages，之后修改原始源码对已安装的包没有任何影响。

**`-e` 安装**：不复制文件，而是在 site-packages 里创建一个**指针文件**（`.pth` 文件或 `direct_url.json`），让 Python 直接到你的源码目录里加载：

```
site-packages/
    rsl_rl.egg-link          ← 或者 __editable__.rsl_rl.pth
    内容：/home/mojie/rsl_rl  ← 指向源码目录
```

效果是：`import rsl_rl` 时，Python 实际加载的是 `/home/mojie/rsl_rl/rsl_rl/` 下的源文件。

**好处**：你修改 `/home/mojie/rsl_rl/rsl_rl/algorithms/ppo.py` 后，不需要重新安装，下次 import 就能看到修改效果。

**使用场景对比**：

| 场景 | 推荐方式 | 原因 |
|------|---------|------|
| 只是使用 rsl_rl，不改源码 | `pip install rsl-rl-lib` | 简单，从 PyPI 直接装 |
| 要修改 rsl_rl 源码调试 | `pip install -e .` | 改完立即生效，不用反复重装 |
| 某个功能 PyPI 版本没有，需要用最新 commit | `git clone + pip install -e .` | 用上 GitHub 最新代码 |

我们的环境用的是方式一（INSTALLER 文件内容是 `pip`，无 `direct_url.json`），说明直接从 PyPI 下载安装，不打算修改 rsl_rl 源码。

---

# 四、完整的 sys.path 结构总结（以 env_isaaclab 为例）

```
sys.path = [
  # [0] 当前脚本所在目录（Python 自动添加）
  "/home/mojie/taskdog/deps/rl_training/scripts/reinforcement_learning/rsl_rl",

  # [1] train.py 手动 append 的（上一级目录）
  "/home/mojie/taskdog/deps/rl_training/scripts/reinforcement_learning",

  # [2] PYTHONPATH 环境变量（本项目没有设，所以这一段为空）

  # [3] Python 标准库
  "/home/mojie/anaconda3/envs/env_isaaclab/lib/python3.11",
  "/home/mojie/anaconda3/envs/env_isaaclab/lib/python3.11/lib-dynload",

  # [4] site-packages（pip 安装的第三方包都在这里）
  "/home/mojie/anaconda3/envs/env_isaaclab/lib/python3.11/site-packages",
]
```

对应到 `train.py` 里的各种 import：

| import 语句 | 从哪里找到的 |
|------------|------------|
| `import cli_args` | `[1]` 手动 append 的父目录 |
| `import argparse` | `[3]` 标准库目录 |
| `import torch` | `[4]` site-packages |
| `from isaaclab.app import AppLauncher` | `[4]` site-packages |
| `from rsl_rl.runners import OnPolicyRunner` | `[4]` site-packages |
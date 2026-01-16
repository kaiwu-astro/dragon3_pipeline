# Dragon3 Pipelines - 开发者指南

本文档为 AI 助手提供 Dragon3 N-body 模拟分析工具的完整上下文信息。

## 项目概述

Dragon3 Pipelines 是一个用于分析和可视化 Dragon3 N-body 模拟数据的模块化 Python 包。该项目专注于处理星团模拟中的粒子、双星系统、合并事件等数据，并提供丰富的可视化功能。

**核心功能：**
- HDF5 格式的模拟数据读取与处理
- 粒子轨迹跟踪与演化分析
- 双星系统特性分析（轨道参数、质量比、引力波合并时标等）
- Lagrangian 半径演化分析
- 多进程并行处理支持
- 自动化绘图与数据可视化

## 技术栈与依赖

### Python 版本要求
- Python >= 3.11

### 核心依赖库
- **pandas >= 2.3.0** - 数据处理与 DataFrame 操作
- **numpy >= 2.2.6** - 数值计算
- **h5py >= 3.13.0** - HDF5 文件读写
- **matplotlib >= 3.10.3** - 绘图与可视化
- **astropy >= 7.0.1** - 天文学计算与单位转换
- **scipy >= 1.15.3** - 科学计算
- **seaborn >= 0.13.2** - 统计可视化
- **PyYAML >= 6.0** - YAML 配置文件解析
- **tqdm >= 4.67.1** - 进度条显示
- **colour >= 0.1.5** - 颜色处理

### 开发工具依赖
- **pytest >= 7.0** - 单元测试框架
- **pytest-cov >= 4.0** - 测试覆盖率
- **black >= 23.0** - 代码格式化
- **ruff >= 0.1.0** - 代码检查
- **mypy >= 1.0** - 类型检查

## 项目结构说明

```
dragon3_pipelines/
├── config/          # 配置管理模块
│   ├── manager.py       # ConfigManager 类，管理所有配置项
│   └── default_config.yaml  # 默认配置文件（包含路径、物理常数、绘图限制等）
├── io/              # 数据输入输出模块
│   ├── hdf5_processor.py    # HDF5FileProcessor，读取 .h5part 文件
│   ├── lagr_processor.py    # LagrFileProcessor，读取 Lagrangian 半径文件
│   └── text_parsers.py      # 文本文件解析函数
├── analysis/        # 数据分析模块
│   ├── particle_tracker.py  # ParticleTracker，跟踪粒子演化
│   └── tau_gw.py            # 引力波合并时标计算
├── visualization/   # 可视化模块
│   ├── base.py              # BaseVisualizer 基类
│   ├── single_star.py       # SingleStarVisualizer，单星可视化
│   ├── binary_star.py       # BinaryStarVisualizer，双星可视化
│   ├── lagr.py              # LagrVisualizer，Lagrangian 半径可视化
│   └── coll_coal.py         # CollCoalVisualizer，碰撞/合并事件可视化
├── utils/           # 工具函数模块
│   ├── serialization.py     # save()/read() 序列化函数
│   ├── logging.py           # @log_time 装饰器，日志工具
│   ├── shell.py             # get_output() 执行 shell 命令
│   ├── color.py             # BlackbodyColorConverter 颜色转换
│   └── misc.py              # 其他辅助函数
└── scripts/         # Shell 脚本（如视频生成脚本）
```

## 代码规范

### 代码格式化
使用 **Black** 进行代码格式化：
```bash
black --line-length=100 dragon3_pipelines/
```

配置（已在 `pyproject.toml` 中定义）：
```toml
[tool.black]
line-length = 100
target-version = ['py311', 'py312', 'py313']
```

### 代码检查
使用 **Ruff** 进行代码检查：
```bash
ruff check dragon3_pipelines/ --fix
```

配置：
```toml
[tool.ruff]
line-length = 100
target-version = "py311"
```

### 类型注解
使用 **mypy** 进行类型检查：
```bash
mypy dragon3_pipelines/
```

配置：
```toml
[tool.mypy]
python_version = "3.11"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = false
```

**类型注解要求：**
- 所有公共函数和方法必须添加类型注解
- 函数参数和返回值都需要类型标注
- 使用 `from typing import` 导入类型提示工具

**示例：**
```python
from typing import Dict, List, Optional
import pandas as pd

def process_data(
    df: pd.DataFrame, 
    threshold: float, 
    columns: Optional[List[str]] = None
) -> Dict[str, pd.DataFrame]:
    """处理数据并返回结果字典"""
    ...
```

### Docstring 风格
使用 **Google Style** docstring：

```python
def function_name(param1: str, param2: int) -> bool:
    """
    简短的一行功能描述。
    
    更详细的功能说明（可选）。可以分多段描述。
    
    Args:
        param1: 参数1的描述
        param2: 参数2的描述
        
    Returns:
        返回值的描述
        
    Raises:
        ValueError: 什么情况下抛出此异常
        
    Example:
        >>> result = function_name("test", 42)
        >>> print(result)
        True
    """
    ...
```

## 关键术语

> 详见 `docs/terminology.md`

### Snapshot vs HDF5 File

**Snapshot（快照）**：
- 指模拟在**某一特定时刻**的数据状态
- 包含单个 `TTOT` 时间点的所有粒子、双星、标量数据
- 从 HDF5 文件中提取：`get_snapshot_at_t(df_dict, ttot)`
- 变量命名示例：`single_df_at_t`、`binary_df_at_t`

**HDF5 File（HDF5 文件）**：
- 物理文件：`.h5part` 格式
- 包含**多个快照**的数据容器（默认每个文件 8 个快照）
- 包含多个 `Step#` 组，每组对应不同的 `TTOT` 值
- 文件命名：`snap.40_<TIME>.h5part`
- 变量命名示例：`hdf5_file_path`、`df_dict`

**使用指南：**
- 提到"特定时刻的数据"时使用 "snapshot"
- 提到"物理文件"或"文件 I/O"时使用 "HDF5 file"

### TTOT 时间单位
`TTOT` 是模拟时间，单位为 N-body 时间单位（通常转换为 Myr）。`TTOT` 是所有快照数据的时间索引。

### KW 恒星类型码
恒星类型使用 KW 码表示（见 `default_config.yaml`）：

```yaml
stellar_types:
  -1: "PMS"      # Pre-main sequence
  0: "LMS"       # Low-mass star
  1: "MS"        # Main sequence
  2: "HG"        # Hertzsprung gap
  3: "GB"        # Giant branch
  7: "HeMS"      # Helium main sequence
  10: "HeWD"     # Helium white dwarf
  11: "COWD"     # Carbon-oxygen white dwarf
  12: "ONeWD"    # Oxygen-neon white dwarf
  13: "NS"       # Neutron star
  14: "BH"       # Black hole
  15: "MLR"      # Massless remnant
```

致密天体 KW 码：`[10, 11, 12, 13, 14]`

## API 兼容性约束

> 详见 `docs/api.md`

### 公共接口必须保持向后兼容

以下类和函数构成公共 API，**不得破坏现有签名**：

**配置模块：**
- `ConfigManager` 类
- `load_config()` 函数

**I/O 模块：**
- `HDF5FileProcessor` 类及其方法
- `LagrFileProcessor` 类及其方法
- `text_parsers` 模块中的解析函数

**分析模块：**
- `ParticleTracker` 类及其方法
- `tau_gw` 模块中的计算函数

**可视化模块：**
- `BaseVisualizer` 基类
- `BaseHDF5Visualizer` 基类
- `SingleStarVisualizer` 类
- `BinaryStarVisualizer` 类
- `LagrVisualizer` 类
- `CollCoalVisualizer` 类

**工具模块：**
- `save()`, `read()` 序列化函数
- `get_output()` shell 命令执行
- `log_time` 装饰器
- `BlackbodyColorConverter` 类

### 扩展原则
- ✅ **可以**添加新的可选参数（必须有默认值）
- ✅ **可以**添加新的方法和类
- ✅ **可以**添加新的返回值字段（在字典或对象中）
- ❌ **不可以**修改现有参数的类型或含义
- ❌ **不可以**删除公共方法或参数
- ❌ **不可以**更改函数返回值的结构

## 开发工作流

### 测试命令
运行所有测试：
```bash
pytest tests/ -v
```

运行测试并生成覆盖率报告：
```bash
pytest tests/ -v --cov=dragon3_pipelines --cov-report=xml --cov-report=term
```

运行特定测试文件：
```bash
pytest tests/test_config.py -v
```

### CI 流程
项目使用 GitHub Actions 进行持续集成（见 `.github/workflows/ci.yml`）：

**触发条件：**
- 推送到 `main`、`develop`、`copilot/**` 分支
- 向 `main`、`develop` 发起 Pull Request

**测试矩阵：**
- Python 3.11, 3.12, 3.13

**CI 步骤：**
1. 安装依赖：`pip install -e ".[dev]"`
2. 运行测试：`pytest tests/ -v --cov=dragon3_pipelines`
3. 上传覆盖率到 Codecov

**Pre-commit Hooks：**
项目配置了 pre-commit（见 `.pre-commit-config.yaml`）：
- `black` - 代码格式化
- `ruff` - 代码检查和自动修复
- `mypy` - 类型检查
- 各类文件检查（trailing whitespace, YAML 语法, 大文件, merge conflict 等）

安装 pre-commit hooks：
```bash
pip install pre-commit
pre-commit install
```

### 并行处理注意事项

**使用 forkserver 上下文：**
```python
import multiprocessing
ctx = multiprocessing.get_context('forkserver')
with ctx.Pool(processes=config.processes_count) as pool:
    results = pool.map(process_func, items)
```

**设置 OpenBLAS 线程数：**
在主程序开始时设置（避免 forkserver 问题）：
```python
import os
os.environ["OPENBLAS_NUM_THREADS"] = "4"
```

**内存管理：**
处理大量数据时，在循环中使用 `gc.collect()` 主动回收内存：
```python
import gc

for hdf5_file in hdf5_files:
    process_file(hdf5_file)
    gc.collect()  # 手动触发垃圾回收
```

**典型并行处理模式：**
```python
def process_single_file(args):
    hdf5_path, simu_name, config = args
    # 处理单个文件
    return result

# 使用 forkserver 上下文
ctx = multiprocessing.get_context('forkserver')
args_list = [(path, name, config) for path in hdf5_files]
with ctx.Pool(processes=config.processes_count) as pool:
    results = pool.map(process_single_file, args_list)
```

## 常见模式

### DataFrame 列名约定

**单星数据 DataFrame（singles）：**
- `NAME` - 粒子名称/ID
- `M` - 质量 [M☉]
- `TTOT` - 模拟时间
- `X1`, `X2`, `X3` - 位置坐标 [pc]
- `V1`, `V2`, `V3` - 速度分量 [km/s]
- `Teff*` - 有效温度 [K]
- `L*` - 光度 [L☉]
- `Distance_to_cluster_center[pc]` - 到星团中心的距离
- `KW` - 恒星类型码

**双星数据 DataFrame（binaries）：**
- `NAME(1)`, `NAME(2)` - 主星和伴星名称
- `Bin A[au]` - 半长轴
- `Bin ECC` - 偏心率
- `mass_ratio` - 质量比（次星质量/主星质量）
- `primary_mass[solar]`, `secondary_mass[solar]` - 主星/次星质量
- `total_mass[solar]` - 双星总质量
- `Bin cm V1`, `Bin cm V2`, `Bin cm V3` - 质心速度
- `tau_gw[Myr]` - 引力波合并时标
- `Ebind/kT` - 结合能（归一化）

**标量数据 DataFrame（scalars）：**
- `TTOT` - 时间
- 各种全局统计量

**合并事件 DataFrame（mergers）：**
- `TTOT` - 合并时间
- `NAME(OUT)` - 合并产物名称
- `NAME(1)`, `NAME(2)` - 参与合并的两个天体

### 缓存机制

**Feather 格式缓存：**
使用 Apache Arrow Feather 格式缓存 DataFrame，速度快且保留类型信息：

```python
import pandas as pd
from pathlib import Path

# 保存缓存
cache_path = Path(config.cache_dir) / "particle_data.feather"
df.to_feather(cache_path)

# 读取缓存
if cache_path.exists():
    df = pd.read_feather(cache_path)
```

**典型缓存模式：**
```python
def get_data_with_cache(cache_path: Path) -> pd.DataFrame:
    """带缓存的数据读取"""
    if cache_path.exists():
        logger.info(f"Loading from cache: {cache_path}")
        return pd.read_feather(cache_path)
    
    # 计算数据
    df = compute_expensive_data()
    
    # 保存缓存
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_feather(cache_path)
    return df
```

### 日志使用

**获取 logger：**
```python
import logging
logger = logging.getLogger(__name__)
```

**使用 @log_time 装饰器：**
```python
from dragon3_pipelines.utils.logging import log_time
import logging

logger = logging.getLogger(__name__)

@log_time(logger)
def expensive_function():
    """耗时函数会自动记录开始/结束时间和执行时长"""
    # 复杂计算
    pass
```

**日志级别：**
```python
logger.debug("调试信息")      # 详细的调试信息
logger.info("正常信息")       # 一般流程信息
logger.warning("警告信息")    # 警告但不影响运行
logger.error("错误信息")      # 错误信息
```

**典型日志模式：**
```python
import logging
from dragon3_pipelines.utils.logging import log_time

logger = logging.getLogger(__name__)

class DataProcessor:
    def __init__(self, config):
        self.config = config
        logger.info(f"Initialized DataProcessor with config: {config}")
    
    @log_time(logger)
    def process_all_files(self, file_list):
        """处理所有文件，自动记录时间"""
        logger.info(f"Processing {len(file_list)} files")
        for i, file_path in enumerate(file_list):
            logger.debug(f"Processing file {i+1}/{len(file_list)}: {file_path}")
            self._process_single_file(file_path)
        logger.info("All files processed")
```

## 禁止事项

### ❌ 禁止修改 default_config.yaml 中的硬编码路径

**原因：**
`default_config.yaml` 中的路径（`paths.simulations`、`paths.plot_dir` 等）是特定于开发环境的硬编码路径。这些路径在不同的机器或用户环境中无效。

**正确做法：**
- 用户应创建自己的 `config.yaml` 覆盖默认路径
- 代码应支持通过命令行参数或环境变量指定配置文件
- 文档应明确说明如何自定义配置

**示例 - 用户自定义配置：**
```yaml
# my_config.yaml
paths:
  simulations:
    my_sim: "/path/to/my/simulation"
  plot_dir: "/path/to/my/plots"

processing:
  processes_count: 20
```

加载自定义配置：
```python
from dragon3_pipelines.config import load_config
config = load_config("my_config.yaml")
```

### ❌ 其他禁止事项

- **禁止删除或重命名公共 API**（见"API 兼容性约束"章节）
- **禁止在代码中硬编码绝对路径**（应使用配置或相对路径）
- **禁止在测试中依赖外部数据文件**（使用 mock 或生成测试数据）
- **禁止提交包含敏感信息的文件**（如 API 密钥、私有路径等）
- **禁止跳过类型注解**（所有公共函数必须有类型标注）
- **禁止使用未经测试的新依赖库**（添加新依赖前应评估必要性和兼容性）

# 步数检测系统 - 基于IMU传感器的计步器

~~源码将在出分后放出~~

## 项目概述

这是一个基于惯性测量单元(IMU)传感器数据的综合步数检测系统。本项目提供离线批量处理和实时在线步数检测功能,支持可配置的信号处理算法和可选的LSTM深度学习模型。

## 主要特性

- **双模式运行**: 支持离线CSV文件处理和实时在线数据流处理
- **高级信号处理**: 多种滤波和分析算法,包括:
  - 巴特沃斯低通滤波器(LPF)
  - Hampel滤波器用于异常值去除
  - Savitzky-Golay滤波器用于平滑处理
  - 质心加速度(CMA)计算
  - 自相关函数(ACF)分析
  - 有限状态机(FSM)用于步态检测
- **深度学习集成**: 可选的LSTM模型以提高步数检测精度
- **可配置参数**: 丰富的参数调节选项,适应不同应用场景和传感器配置
- **消融研究支持**: 内置评估框架,用于算法组件分析
- **图形用户界面**: 基于PyQt5的GUI,便于配置和可视化
- **实时可视化**: 传感器数据和检测到的步数的实时绘图

## 项目结构

```
github/
├── demo.py              # 主GUI应用程序,包含离线、在线和评估三种模式
├── step_counter.py      # 核心计步器实现
├── train.py             # LSTM模型训练脚本
├── README.md            # 英文文档
└── README-zh.md         # 中文文档(本文件)
```

## 环境要求

### Python依赖包

```bash
pip install numpy pandas matplotlib scipy scikit-learn tensorflow PyQt5 hampel requests
```
或者
```bash
pip install -r requirements.txt
```

### 系统要求

- Python 3.7 或更高版本
- PyQt5 (用于GUI功能)
- TensorFlow/Keras (用于LSTM模型支持,可选)

## 使用方法

### 1. 离线模式

处理包含IMU传感器数据的CSV文件:

```bash
python demo.py
```

然后在GUI中选择"Offline Step Counter":
1. 浏览并选择CSV文件
2. 根据需要配置参数
3. 点击OK开始处理

### 2. 在线模式

连接到实时数据流:

```bash
python demo.py
```

然后在GUI中选择"Online Step Counter":
1. 输入数据源的IP地址和端口
2. 配置参数
3. 点击OK开始实时检测

### 3. 训练LSTM模型

训练自定义的LSTM模型用于步数检测:

```bash
python train.py
```

运行前在`train.py`中配置训练参数:
- `INPUT_ROOT`: 原始训练数据路径
- `OUTPUT_ROOT`: 保存训练模型的目录
- `MODEL_NAME`: 输出模型文件名

### 4. 消融研究

评估不同的算法配置:

```bash
python demo.py
```

选择"Evaluation"模式运行带有多个参数组合的消融研究。

## 配置参数说明

### 信号处理参数

| 参数名 | 默认值 | 说明 |
|--------|--------|------|
| `window_sec` | 2.0 | 分析窗口大小(秒) |
| `threshold_factor` | 0.8 | 步数检测阈值倍数 |
| `refractory_sec` | 0.75 | 连续步之间的最小时间间隔 |
| `filter_len` | 5 | 平滑滤波器长度 |
| `cutoff_low` | 0.5 | 带通滤波器低截止频率(Hz) |
| `cutoff_high` | 2.0 | 带通滤波器高截止频率(Hz) |
| `fs` | 50.0 | 采样频率(Hz) |
| `order` | 3 | 滤波器阶数 |
| `n_sigma` | 3.0 | 异常值检测的标准差倍数 |
| `polyorder` | 3 | Savitzky-Golay滤波器的多项式阶数 |

### 算法标志

- **gravity**: 启用重力补偿
- **CMA**: 启用心加速度计算
- **hampel**: 启用Hampel滤波器进行异常值去除
- **savgol**: 启用Savitzky-Golay平滑滤波器
- **LPF**: 启用低通滤波器
- **ACF**: 启用自相关函数分析
- **FSM**: 启用有限状态机进行步数检测

### 高级参数

| 参数名 | 默认值 | 说明 |
|--------|--------|------|
| `gyro_weight` | 0.3 | 陀螺仪数据融合权重 |
| `noise_threshold` | 0.2 | 信号质量评估的噪声阈值 |
| `threshold_absolute` | 0.6 | 步数检测的绝对阈值 |
| `gyro_energy_min` | 0.05 | 最小陀螺仪能量阈值 |
| `mag_var_min` | 0.05 | 最小磁力计方差阈值 |
| `penalty` | 0.5 | 减少误报的惩罚因子 |
| `acf_dominance` | 0.7 | ACF主导性阈值 |

### FSM参数(启用FSM时)

| 参数名 | 默认值 | 说明 |
|--------|--------|------|
| `peak_prominence` | 0.1 | 最小峰值显著度 |
| `valley_prominence` | 0.05 | 最小谷值显著度 |
| `min_step_period` | 0.3 | 最小步周期(秒) |
| `max_step_period` | 3.0 | 最大步周期(秒) |
| `state_timeout` | 1.5 | 状态超时持续时间(秒) |
| `peak_valley_min_dist` | 0.2 | 峰谷之间的最小距离 |

## 输入数据格式

CSV文件应包含以下列:
- 时间戳信息
- 加速度计数据(ax, ay, az)
- 陀螺仪数据(gx, gy, gz)
- 磁力计数据(mx, my, mz) - 可选

示例格式:
```csv
timestamp,ax,ay,az,gx,gy,gz,mx,my,mz
2024-01-01 10:00:00.000,0.1,0.2,9.8,0.01,0.02,0.03,0.1,0.2,0.3
...
```

## 输出结果

系统提供:
- 总步数统计
- 步数时间戳
- 标记了检测到的步数的传感器数据可视化图表
- 实时步数更新(在线模式)
- 评估指标(消融研究模式)

## 系统架构

### 核心组件

1. **StepCounter类** (`step_counter.py`)
   - 主步数检测引擎
   - 实现所有信号处理算法
   - 支持批处理和流处理两种模式

2. **DataPreprocessor类** (`step_counter.py`)
   - 数据加载和预处理
   - 传感器数据归一化
   - 时间序列插值

3. **Analysis类** (`step_counter.py`)
   - 性能评估和指标计算
   - 结果可视化
   - 统计分析

4. **GUI应用程序** (`demo.py`)
   - 基于PyQt5的用户界面
   - 三种运行模式:离线、在线、评估
   - 实时数据可视化

5. **训练流水线** (`train.py`)
   - LSTM模型训练
   - 数据准备和增强
   - 模型评估和导出

## 自定义扩展

### 添加新算法

添加新的信号处理算法:
1. 在`StepCounter`类中实现算法
2. 在参数字段中添加相应参数
3. 在`step_counter.py`中更新处理流程

### 修改LSTM模型

编辑`train.py`以自定义LSTM架构:
- 更改LSTM单元数量
- 修改网络深度
- 调整训练超参数

## 常见问题

### 常见问题及解决方案

1. **PyQt5导入错误**
   ```bash
   pip install PyQt5
   ```

2. **找不到TensorFlow**(使用LSTM时)
   ```bash
   pip install tensorflow
   ```

3. **CSV文件格式错误**
   - 确保CSV文件有正确的列标题
   - 检查时间戳格式一致性
   - 验证传感器读数的数值数据类型

4. **连接被拒绝**(在线模式)
   - 验证IP地址和端口设置
   - 确保数据服务器正在运行
   - 检查防火墙设置

## 性能优化建议

- 根据应用选择合适的`window_sec`(实时应用用较短窗口,精度优先用较长窗口)
- 仅启用必要的算法标志以减少计算开销
- 根据传感器噪声特性调整`threshold_factor`
- 在噪声环境中,增加`n_sigma`并启用所有滤波器
- 仅在传统方法不足时使用LSTM模型

## 许可证

本项目用于教育和研究目的。

## 致谢

- 香港大学课程 7310
- 项目贡献者和维护者

## 联系方式

如有问题或疑问,请参考课程材料或联系项目维护者。

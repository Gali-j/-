# 机器学习期末课程项目 - 风速测试试验

## 项目简介
本项目为机器学习课程期末作业，以气象风速观测数据为研究对象，完成从数据探索、预处理、特征工程到模型构建、评估与可视化的全流程机器学习实验。项目通过传统机器学习算法与深度学习模型对比，实现风速相关特征分析与预测任务，验证不同模型在气象时序数据上的泛化效果。

## 数据集说明
本实验采用结构化气象观测数据集，已按标准比例划分为训练集、验证集与测试集，数据集基础信息如下：
- 总样本量：10573 条
  - 训练集（train）：8458 条
  - 验证集（val）：1057 条
  - 测试集（test）：1058 条
- 数据总大小：约257kb


### 数据字段详情
| 字段名 | 数据类型 | 字段说明 |
| :--- | :--- | :--- |
| Date & Time Stamp | string | 观测时间戳 |
| SpeedAvg | float64 | 平均风速 |
| SpeedMax | float64 | 最大风速 |
| DirectionAvg | int64 | 平均风向 |
| TemperatureAvg | float64 | 平均温度 |
| TemperatureMax | float64 | 最高温度 |
| PressureAvg | float64 | 平均气压 |
| PressureMax | float64 | 最高气压 |
| HumidtyAvg | float64 | 平均湿度 |
| HumityMax | float64 | 最大湿度 |
| height | int64 | 观测高度 |

## 环境与依赖
### 核心依赖库
- 数据处理：`pandas`、`numpy`、`scipy`、`pyarrow`
- 机器学习：`scikit-learn`
- 深度学习框架：`torch`
- 可视化工具：`matplotlib`、`seaborn`

### 依赖安装
执行以下命令一键安装所有依赖：
```bash
pip install pandas numpy scikit-learn torch matplotlib seaborn scipy pyarrow

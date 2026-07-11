# ==========================================================
# 商学院机器学习课程项目 - 风速序列预测 完整版（本地Parquet数据）
# 依赖安装：pip install pandas numpy scikit-learn torch matplotlib seaborn scipy pyarrow
# ==========================================================
# -*- coding: utf-8 -*-

# ==========================================================
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import warnings
warnings.filterwarnings("ignore")

# ====================== 1. 本地数据路径配置 ======================
# 已适配你桌面的文件路径，使用 r"" 原始字符串避免Windows路径转义报错
DATA_PATHS = {
    "10m": {
        "train": r"C:\Users\LEGION\Desktop\train-00000-of-00001 10m.parquet",
        "val": r"C:\Users\LEGION\Desktop\val-00000-of-00001 10m.parquet",
        "test": r"C:\Users\LEGION\Desktop\test-00000-of-00001 10m.parquet"
    },
    "50m": {
        "train": r"C:\Users\LEGION\Desktop\train-00000-of-00001 50m.parquet",
        "val": r"C:\Users\LEGION\Desktop\val-00000-of-00001 50m.parquet",
        "test": r"C:\Users\LEGION\Desktop\test-00000-of-00001 50m.parquet"
    },
    "100m": {
        "train": r"C:\Users\LEGION\Desktop\train-00000-of-00001 100m.parquet",
        "val": r"C:\Users\LEGION\Desktop\val-00000-of-00001 100m.parquet",
        "test": r"C:\Users\LEGION\Desktop\test-00000-of-00001 100m.parquet"
    }
}

# ====================== 2. 全局任务参数配置 ======================
# 切换预测任务只需修改 PRED_LEN：
# 单步预测: PRED_LEN = 1
# 多步预测A(8h→1h): PRED_LEN = 6
# 多步预测B(8h→16h): PRED_LEN = 96
HISTORY_LEN = 48        # 历史窗口：8小时 = 48个10分钟步长
PRED_LEN = 1            # 预测步长
TARGET_COL = "SpeedAvg_100m"  # 预测目标：100米平均风速
BATCH_SIZE = 64
EPOCHS = 100
LEARNING_RATE = 1e-3
PATIENCE = 10           # 早停耐心值
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_DIR = "./saved_models"
os.makedirs(SAVE_DIR, exist_ok=True)

# ====================== 3. 数据加载与多高度整合 ======================
def load_single_height(file_paths, height_suffix):
    """加载单个高度的train/val/test parquet文件，合并为全量数据并做基础清洗"""
    # 读取三个拆分文件
    df_train = pd.read_parquet(file_paths["train"])
    df_val = pd.read_parquet(file_paths["val"])
    df_test = pd.read_parquet(file_paths["test"])
    
    # 合并为完整数据集
    df = pd.concat([df_train, df_val, df_test], ignore_index=True)
    
    # 统一时间戳列名并转为标准datetime格式
    df.rename(columns={"Date & Time Stamp": "timestamp"}, inplace=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    
    # 去除列名中的空格，避免后续索引报错
    df.columns = df.columns.str.replace(" ", "_")
    
    # 给所有非时间戳列添加高度后缀，避免不同高度列名冲突
    for col in df.columns:
        if col != "timestamp":
            df.rename(columns={col: f"{col}_{height_suffix}"}, inplace=True)
    
    # 修正原始数据中湿度字段的拼写误差
    df.columns = df.columns.str.replace("Humidty", "Humidity")
    df.columns = df.columns.str.replace("Humity", "HumidityMax")
    
    # 删除50m数据集中冗余的height列（已通过后缀区分高度）
    height_col = f"height_{height_suffix}"
    if height_col in df.columns:
        df.drop(height_col, axis=1, inplace=True)
    
    # 按时间排序、去重，保证时序严格唯一
    df = df.sort_values("timestamp").drop_duplicates(subset="timestamp").reset_index(drop=True)
    print(f"✅ {height_suffix} 高度加载完成：共 {len(df)} 条样本，{len(df.columns)-1} 个特征")
    return df

print("="*60)
print("步骤1：加载并整合三个高度数据集")
print("="*60)
df_10m = load_single_height(DATA_PATHS["10m"], "10m")
df_50m = load_single_height(DATA_PATHS["50m"], "50m")
df_100m = load_single_height(DATA_PATHS["100m"], "100m")

# 按时间戳内连接拼接，仅保留三个高度都存在的时间点
df_merged = df_10m.merge(df_50m, on="timestamp", how="inner")
df_merged = df_merged.merge(df_100m, on="timestamp", how="inner")
df_merged = df_merged.sort_values("timestamp").reset_index(drop=True)

print(f"\n三高度整合完成，总样本数：{len(df_merged)}，总字段数：{len(df_merged.columns)}")
print(f"时间范围：{df_merged['timestamp'].min()} 至 {df_merged['timestamp'].max()}")

# ====================== 4. 数据清洗 ======================
def clean_data(df):
    """10分钟重采样 + 缺失值处理 + 3σ异常值截断"""
    # 设置时间索引并重采样为10分钟固定间隔
    df = df.set_index("timestamp").sort_index()
    df = df.resample("10T").mean()
    print(f"\n10分钟重采样后样本数：{len(df)}")
    
    # 缺失值分级处理
    df = df.interpolate(method="linear")  # 线性插值填充短缺失
    df = df.ffill().bfill()               # 前后填充边缘缺失
    df = df.dropna()                      # 删除长缺失行
    print(f"缺失值处理后样本数：{len(df)}")
    
    # 3σ原则处理异常值
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        mean = df[col].mean()
        std = df[col].std()
        lower = mean - 3 * std
        upper = mean + 3 * std
        df[col] = np.clip(df[col], lower, upper)
    
    print("✅ 数据清洗完成")
    return df

print("\n" + "="*60)
print("步骤2：数据清洗")
print("="*60)
df_clean = clean_data(df_merged)

# ====================== 5. 探索性可视化（作业要求） ======================
print("\n" + "="*60)
print("步骤3：生成可视化图表")
print("="*60)

# 图1：数据集分布图（风速分布 + 多高度箱线图）
def plot_data_distribution(df, target_col):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 目标风速分布直方图+核密度
    sns.histplot(df[target_col], kde=True, ax=axes[0], color="#1f77b4", bins=30)
    axes[0].set_title("Wind Speed Distribution (100m Height)", fontsize=12)
    axes[0].set_xlabel("Wind Speed (m/s)")
    axes[0].set_ylabel("Frequency")
    
    # 多高度风速箱线图对比
    speed_cols = [col for col in df.columns if "SpeedAvg_" in col]
    sns.boxplot(data=df[speed_cols], ax=axes[1], palette="Set2")
    axes[1].set_title("Wind Speed Comparison by Height", fontsize=12)
    axes[1].set_ylabel("Wind Speed (m/s)")
    
    plt.tight_layout()
    plt.savefig("data_distribution.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("✅ 已生成：data_distribution.png（数据集分布图）")

# 图2：特征相关性热力图
def plot_correlation(df):
    plt.figure(figsize=(14, 12))
    corr = df.corr()
    sns.heatmap(corr, cmap="RdBu_r", center=0, vmin=-1, vmax=1,
                square=True, linewidths=0.3, annot=False)
    plt.title("Feature Correlation Heatmap", fontsize=14)
    plt.tight_layout()
    plt.savefig("feature_correlation.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("✅ 已生成：feature_correlation.png（特征相关性图）")

plot_data_distribution(df_clean, TARGET_COL)
plot_correlation(df_clean)

# ====================== 6. 特征工程 ======================
def feature_engineering(df):
    """时间周期编码 + 风向角度编码"""
    df = df.copy()
    
    # 时间周期特征：小时、月份的正弦余弦编码
    df["hour"] = df.index.hour
    df["month"] = df.index.month
    df["hour_sin"] = np.sin(df["hour"] * 2 * np.pi / 24)
    df["hour_cos"] = np.cos(df["hour"] * 2 * np.pi / 24)
    df["month_sin"] = np.sin((df["month"] - 1) * 2 * np.pi / 12)
    df["month_cos"] = np.cos((df["month"] - 1) * 2 * np.pi / 12)
    df.drop(["hour", "month"], axis=1, inplace=True)
    
    # 所有高度风向的正弦余弦编码
    dir_cols = [col for col in df.columns if "DirectionAvg" in col]
    for col in dir_cols:
        df[f"{col}_sin"] = np.sin(df[col] * np.pi / 180)
        df[f"{col}_cos"] = np.cos(df[col] * np.pi / 180)
        df.drop(col, axis=1, inplace=True)
    
    print(f"✅ 特征工程完成，最终特征数：{len(df.columns)}")
    return df

print("\n" + "="*60)
print("步骤4：特征工程")
print("="*60)
df_feat = feature_engineering(df_clean)

# ====================== 7. 时序7:2:1划分 + 标准化 ======================
def split_and_scale(df, target_col, train_ratio=0.7, val_ratio=0.2):
    """纯时序划分，仅训练集拟合适配器，杜绝数据泄露"""
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:]

    feature_cols = df.columns.tolist()
    
    # 全特征标准化
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_df[feature_cols])
    val_scaled = scaler.transform(val_df[feature_cols])
    test_scaled = scaler.transform(test_df[feature_cols])

    # 提取目标列缩放参数，用于反归一化
    target_idx = feature_cols.index(target_col)
    target_scaler = StandardScaler()
    target_scaler.mean_ = scaler.mean_[target_idx]
    target_scaler.scale_ = scaler.scale_[target_idx]

    # 转回DataFrame保留索引
    train_scaled = pd.DataFrame(train_scaled, columns=feature_cols, index=train_df.index)
    val_scaled = pd.DataFrame(val_scaled, columns=feature_cols, index=val_df.index)
    test_scaled = pd.DataFrame(test_scaled, columns=feature_cols, index=test_df.index)

    print(f"训练集：{len(train_scaled)} | 验证集：{len(val_scaled)} | 测试集：{len(test_scaled)}")
    return train_scaled, val_scaled, test_scaled, target_scaler, feature_cols

print("\n" + "="*60)
print("步骤5：数据集7:2:1时序划分与标准化")
print("="*60)
train_scaled, val_scaled, test_scaled, target_scaler, feature_cols = split_and_scale(df_feat, TARGET_COL)

# ====================== 8. 滑动窗口构造时序样本 ======================
def create_sliding_windows(data_array, history_len, pred_len, target_idx):
    """生成 (历史窗口) → (未来预测) 样本对"""
    X, y = [], []
    for i in range(history_len, len(data_array) - pred_len + 1):
        X.append(data_array[i-history_len:i, :])
        y.append(data_array[i:i+pred_len, target_idx])
    return np.array(X), np.array(y)

target_idx = feature_cols.index(TARGET_COL)
train_X, train_y = create_sliding_windows(train_scaled.values, HISTORY_LEN, PRED_LEN, target_idx)
val_X, val_y = create_sliding_windows(val_scaled.values, HISTORY_LEN, PRED_LEN, target_idx)
test_X, test_y = create_sliding_windows(test_scaled.values, HISTORY_LEN, PRED_LEN, target_idx)

print(f"\n样本构造完成：")
print(f"训练集：{len(train_X)} 个，输入形状{train_X.shape[1:]}，输出形状{train_y.shape[1:]}")
print(f"验证集：{len(val_X)} 个")
print(f"测试集：{len(test_X)} 个")

# 构建PyTorch数据集与加载器
class WindDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_dataset = WindDataset(train_X, train_y)
val_dataset = WindDataset(val_X, val_y)
test_dataset = WindDataset(test_X, test_y)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ====================== 9. 三类模型定义 ======================
# 模型1：线性回归（传统基线模型）
class LinearModel(nn.Module):
    def __init__(self, history_len, feature_dim, pred_len):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(history_len * feature_dim, pred_len)
    def forward(self, x):
        x = self.flatten(x)
        return self.linear(x)

# 模型2：LSTM（循环神经网络）
class LSTMModel(nn.Module):
    def __init__(self, history_len, feature_dim, pred_len, hidden_dim=128, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=feature_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )
        self.fc = nn.Linear(hidden_dim, pred_len)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_out = lstm_out[:, -1, :]
        return self.fc(self.dropout(last_out))

# 模型3：Transformer（自注意力前沿架构）
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)
    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return x

class TransformerModel(nn.Module):
    def __init__(self, history_len, feature_dim, pred_len, d_model=128, nhead=4, num_layers=2, dropout=0.2):
        super().__init__()
        self.embedding = nn.Linear(feature_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, history_len)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model*4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, pred_len)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        x = self.embedding(x)
        x = self.pos_encoder(x)
        enc_out = self.transformer_encoder(x)
        pooled = enc_out.mean(dim=1)
        return self.fc(self.dropout(pooled))

# ====================== 10. 训练与评估函数 ======================
def train_model(model, train_loader, val_loader, save_path):
    """训练+早停+保存最优.pth模型"""
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(EPOCHS):
        # 训练阶段
        model.train()
        train_loss = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
            optimizer.zero_grad()
            pred = model(batch_x)
            loss = criterion(pred, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch_x.size(0)
        train_loss /= len(train_loader.dataset)

        # 验证阶段
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
                pred = model(batch_x)
                val_loss += criterion(pred, batch_y).item() * batch_x.size(0)
        val_loss /= len(val_loader.dataset)

        # 早停与保存最优模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)
            patience_counter = 0
        else:
            patience_counter += 1

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

        if patience_counter >= PATIENCE:
            print(f"早停触发，共训练 {epoch+1} 轮，最优验证损失: {best_val_loss:.6f}")
            break

    # 加载最优权重
    model.load_state_dict(torch.load(save_path, map_location=DEVICE, weights_only=True))
    return model

def evaluate_model(model, test_loader, target_scaler):
    """测试集评估，计算MSE/RMSE/MAE/R²，返回原始量纲结果"""
    model.eval()
    preds = []
    trues = []
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(DEVICE)
            pred = model(batch_x).cpu().numpy()
            preds.append(pred)
            trues.append(batch_y.numpy())
    
    preds = np.concatenate(preds, axis=0)
    trues = np.concatenate(trues, axis=0)
    
    # 反归一化
    preds_original = preds * target_scaler.scale_ + target_scaler.mean_
    trues_original = trues * target_scaler.scale_ + target_scaler.mean_
    
    # 计算四项指标
    mse = mean_squared_error(trues_original.flatten(), preds_original.flatten())
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(trues_original.flatten(), preds_original.flatten())
    r2 = r2_score(trues_original.flatten(), preds_original.flatten())
    
    metrics = {
        "MSE": round(mse, 4),
        "RMSE": round(rmse, 4),
        "MAE": round(mae, 4),
        "R²": round(r2, 4)
    }
    return metrics, trues_original, preds_original

# ====================== 11. 主流程：训练+评估+对比 ======================
print("\n" + "="*60)
print("步骤6：三类模型训练与评估")
print("="*60)
print(f"运行设备：{DEVICE}")
print(f"任务配置：历史{HISTORY_LEN}步 → 预测{PRED_LEN}步\n")

feature_dim = len(feature_cols)
models = {
    "Linear Regression": LinearModel(HISTORY_LEN, feature_dim, PRED_LEN).to(DEVICE),
    "LSTM": LSTMModel(HISTORY_LEN, feature_dim, PRED_LEN).to(DEVICE),
    "Transformer": TransformerModel(HISTORY_LEN, feature_dim, PRED_LEN).to(DEVICE)
}

all_metrics = {}
all_preds = []
model_names = list(models.keys())

for name, model in models.items():
    print(f"\n--- 正在训练 {name} ---")
    save_path = os.path.join(SAVE_DIR, f"{name.replace(' ', '_').lower()}.pth")
    model = train_model(model, train_loader, val_loader, save_path)
    
    metrics, trues, preds = evaluate_model(model, test_loader, target_scaler)
    all_metrics[name] = metrics
    all_preds.append(preds)
    
    print(f"{name} 测试集性能：")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

# 输出性能对比表
print("\n" + "="*60)
print("📊 所有模型性能对比汇总（测试集，原始量纲）")
print("="*60)
metrics_df = pd.DataFrame(all_metrics).T
print(metrics_df)
metrics_df.to_csv("model_metrics.csv")
print("\n✅ 已保存：model_metrics.csv（指标对比表）")

# ====================== 12. 预测结果可视化（作业要求） ======================
def plot_prediction_comparison(trues, preds, model_names):
    plt.figure(figsize=(14, 6))
    plt.plot(trues[:200, 0], label="True Wind Speed", color="black", linewidth=1.5)
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    for i, pred in enumerate(preds):
        plt.plot(pred[:200, 0], label=model_names[i], color=colors[i], alpha=0.85)
    
    plt.title("Wind Speed Prediction Comparison (Test Set)", fontsize=14)
    plt.xlabel("Time Step (10 minutes)", fontsize=12)
    plt.ylabel("Wind Speed (m/s)", fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("prediction_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("✅ 已生成：prediction_comparison.png（预测结果对比图）")

plot_prediction_comparison(trues, all_preds, model_names)

# ====================== 最终总结 ======================
print("\n" + "="*60)
print("🎉 项目全部运行完成！")
print("="*60)
print("生成的文件清单：")
print("1. 模型文件（saved_models/目录）：")
print("   - linear_regression.pth")
print("   - lstm.pth")
print("   - transformer.pth")
print("2. 可视化图表：")
print("   - data_distribution.png（数据集分布图）")
print("   - feature_correlation.png（特征相关性图）")
print("   - prediction_comparison.png（预测结果对比图）")
print("3. 指标文件：model_metrics.csv（四项指标对比表）")
print("\n所有文件均在当前代码运行目录下，可直接插入实验报告。")
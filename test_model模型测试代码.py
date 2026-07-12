import torch
import torch.nn as nn
import numpy as np

HISTORY_LEN = 48
FEATURE_DIM = 36
PRED_LEN = 1
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class LinearModel(nn.Module):
    def __init__(self, history_len, feature_dim, pred_len):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(history_len * feature_dim, pred_len)
    def forward(self, x):
        x = self.flatten(x)
        return self.linear(x)

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

def load_model(model_name):
    if model_name == "Linear Regression":
        model = LinearModel(HISTORY_LEN, FEATURE_DIM, PRED_LEN)
        path = "./saved_models/linear_regression.pth"
    elif model_name == "LSTM":
        model = LSTMModel(HISTORY_LEN, FEATURE_DIM, PRED_LEN)
        path = "./saved_models/lstm.pth"
    elif model_name == "Transformer":
        model = TransformerModel(HISTORY_LEN, FEATURE_DIM, PRED_LEN)
        path = "./saved_models/transformer.pth"
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    model.load_state_dict(torch.load(path, map_location=DEVICE, weights_only=True))
    model.to(DEVICE)
    model.eval()
    return model

print("="*60)
print("📦 加载训练好的模型")
print("="*60)

models = {}
for name in ["Linear Regression", "LSTM", "Transformer"]:
    models[name] = load_model(name)
    print(f"✅ {name} 模型加载成功")

print(f"\n运行设备：{DEVICE}")

print("\n" + "="*60)
print("🧪 生成测试数据并进行预测")
print("="*60)

np.random.seed(42)
test_data = np.random.randn(3, HISTORY_LEN, FEATURE_DIM)
test_tensor = torch.tensor(test_data, dtype=torch.float32).to(DEVICE)

print(f"输入数据形状：{test_tensor.shape}")
print(f"  - 样本数：{test_tensor.shape[0]}")
print(f"  - 历史窗口：{test_tensor.shape[1]} 步")
print(f"  - 特征数：{test_tensor.shape[2]}")

print("\n预测结果（标准化后的风速值）：")
print("-" * 50)

with torch.no_grad():
    for name, model in models.items():
        pred = model(test_tensor).cpu().numpy()
        print(f"\n{name}:")
        for i in range(len(pred)):
            print(f"  样本{i+1}: {pred[i][0]:.4f}")

print("\n" + "="*60)
print("💡 使用说明：")
print("="*60)
print("1. 真实数据需要先进行标准化处理，再输入模型")
print("2. 模型输出需要使用 target_scaler 进行反归一化")
print("3. 反归一化公式：原始风速 = 预测值 * scale + mean")
print("4. scale 和 mean 参数在训练脚本中已保存")
print("\n示例（假设scale=3.5, mean=7.0）：")
print("  原始风速 = 预测值 * 3.5 + 7.0")

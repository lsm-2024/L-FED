import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 更换为 Agg 后端（仅支持保存图片，不显示窗口）
import matplotlib.pyplot as plt
from ta.trend import SMAIndicator, EMAIndicator
from ta.momentum import RSIIndicator
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

def load_btc_eth_data(file_path):
    """加载`btc.docx`/`eth.docx`对应的CSV数据，输出8个特征"""
    # 数据字段必须与BTC/ETH交易数据一致：date/open/high/low/close/volume
    df = pd.read_csv(file_path, parse_dates=["candle_begin_time"], index_col="candle_begin_time")
    df = df.sort_index()  # 时间序列按升序排列（BTC/ETH数据时序性要求）
    df = df.dropna(subset=["open", "high", "low", "close", "volume"])  # 删除核心数据缺失值

    # 计算3个技术指标（与前期逻辑一致，确保8特征完整性）
    df["sma_14"] = SMAIndicator(close=df["close"], window=14).sma_indicator()
    df["ema_14"] = EMAIndicator(close=df["close"], window=14).ema_indicator()
    df["rsi_14"] = RSIIndicator(close=df["close"], window=14).rsi()

    # 固定8个特征（严格匹配BTC/ETH数据）
    feature_cols = ["open", "high", "low", "close", "volume", "sma_14", "ema_14", "rsi_14"]
    df = df[feature_cols].dropna()  # 删除指标计算产生的缺失值

    print(f"数据加载完成（{file_path}）：时间范围{df.index.min()}至{df.index.max()}，特征数={df.shape[1]}（8个，符合要求）")
    return df

def create_btc_eth_timeseries(df, seq_len=16, target_col="close"):
    """构建BTC/ETH时序数据：16天历史→1天收盘价，6:2:2划分"""
    # 归一化（特征与目标共用scaler，确保BTC/ETH价格反归一化正确）
    scaler = MinMaxScaler(feature_range=(0, 1))
    df_scaled = scaler.fit_transform(df)

    # 构建输入输出窗口（8特征×16天）
    X, y = [], []
    for i in range(seq_len, len(df_scaled)):
        X.append(df_scaled[i-seq_len:i, :])  # (16, 8)
        y.append(df_scaled[i, df.columns.get_loc(target_col)])  # 1天收盘价
    X = np.array(X)  # (样本数, 16, 8)
    y = np.array(y).reshape(-1, 1)  # (样本数, 1)

    # 6:2:2划分（时序不打乱，避免BTC/ETH数据泄露）
    total = len(X)
    train_size = int(total * 0.6)
    val_size = int(total * 0.2)
    X_train, y_train = X[:train_size], y[:train_size]
    X_val, y_val = X[train_size:train_size+val_size], y[train_size:train_size+val_size]
    X_test, y_test = X[train_size+val_size:], y[train_size+val_size:]

    # 转为PyTorch张量（适配GPU）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tensors = [torch.tensor(data, dtype=torch.float32).to(device)
               for data in [X_train, X_val, X_test, y_train, y_val, y_test]]
    X_train, X_val, X_test, y_train, y_val, y_test = tensors

    print(f"\nBTC/ETH数据划分（总样本数={total}）：")
    print(f"训练集：X={X_train.shape}, y={y_train.shape} | 验证集：X={X_val.shape}, y={y_val.shape} | 测试集：X={X_test.shape}, y={y_test.shape}")
    return X_train, X_val, X_test, y_train, y_val, y_test, scaler, device

class PositionalEncoding(nn.Module):
    """FEDFormer分支位置编码：适配BTC/ETH 16天窗口"""
    def __init__(self, d_model, max_len=16):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, 16, d_model)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class FEDFormerBlock(nn.Module):
    """FEDFormer分支基础块：频率分解适配BTC/ETH价格波动"""
    def __init__(self, d_model, nhead=4, dropout=0.1):
        super().__init__()
        # 频率分解：分离BTC/ETH价格高低频
        self.freq_decomp = lambda x: (torch.fft.rfft(x, dim=1).real, torch.fft.rfft(x, dim=1).imag)
        self.freq_recon = lambda r, i: torch.fft.irfft(torch.complex(r, i), dim=1, n=r.size(1))

        # 多头注意力（处理高频波动）
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.drop1 = nn.Dropout(dropout)

        # 前馈网络（处理低频趋势）
        self.ffn = nn.Sequential(nn.Linear(d_model, d_model*2), nn.ReLU(), nn.Dropout(dropout), nn.Linear(d_model*2, d_model))
        self.norm2 = nn.LayerNorm(d_model)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x):
        # 高低频分离与处理
        x_real, x_imag = self.freq_decomp(x)
        x_low = self.freq_recon(x_real, torch.zeros_like(x_imag))  # 低频趋势
        x_high = self.freq_recon(torch.zeros_like(x_real), x_imag)  # 高频波动

        # 高频分支：注意力
        attn_out, _ = self.attn(x_high, x_high, x_high)
        x_high = self.norm1(x_high + self.drop1(attn_out))

        # 低频分支：前馈网络
        ffn_out = self.ffn(x_low)
        x_low = self.norm2(x_low + self.drop2(ffn_out))

        return x_low + x_high  # 高低频融合

class LSTM_FEDFormer_Concat(nn.Module):
    """并行混合模型：LSTM分支 + FEDFormer分支 + Concat融合（适配BTC/ETH 8特征）"""
    def __init__(self, input_dim=8, seq_len=16, d_model=64, lstm_hidden=64, num_layers=2, dropout=0.1):
        super().__init__()
        self.input_dim = input_dim  # 固定8（BTC/ETH特征数）
        self.seq_len = seq_len      # 固定16（BTC/ETH历史窗口）
        self.d_model = d_model      # FEDFormer特征维度
        self.lstm_hidden = lstm_hidden  # LSTM隐藏层维度

        # ---------------------- 1. LSTM分支（捕捉时序依赖）----------------------
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=lstm_hidden,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False,
            dropout=dropout if num_layers>1 else 0
        )
        self.lstm_proj = nn.Linear(lstm_hidden, d_model)  # LSTM输出→d_model维度（与FEDFormer对齐）

        # ---------------------- 2. FEDFormer分支（捕捉高低频波动）----------------------
        self.fed_proj = nn.Linear(input_dim, d_model)  # 8特征→d_model维度
        self.pos_enc = PositionalEncoding(d_model, max_len=seq_len)
        self.fed_blocks = nn.ModuleList([FEDFormerBlock(d_model) for _ in range(num_layers)])
        self.fed_proj = nn.Linear(input_dim, d_model)
        self.fed_pool = nn.AdaptiveAvgPool1d(1)  # FEDFormer输出池化→(batch, d_model, 1)

        # ---------------------- 3. Concat融合层 + 预测头（适配BTC/ETH收盘价预测）----------------------
        self.concat_dim = d_model + d_model  # LSTM输出(d_model) + FEDFormer输出(d_model)
        self.predict_head = nn.Sequential(
            nn.Linear(self.concat_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1)  # 输出1天收盘价
        )

    def forward(self, x):
        # x: (batch_size, 16, 8) → BTC/ETH输入格式
        batch_size = x.size(0)

        # ---------------------- LSTM分支前向传播 ----------------------
        lstm_out, _ = self.lstm(x)  # (batch, 16, lstm_hidden=64)
        lstm_out = lstm_out[:, -1, :]  # 取最后一步输出（时序依赖汇总）→ (batch, 64)
        lstm_feat = self.lstm_proj(lstm_out)  # (batch, d_model=64)

        # ---------------------- FEDFormer分支前向传播 ----------------------
        fed_out = self.fed_proj(x)  # (batch, 16, 8) → (batch, 16, d_model=64)
        fed_out = self.pos_enc(fed_out)  # 加入位置信息
        for block in self.fed_blocks:
            fed_out = block(fed_out)  # (batch, 16, 64)
        fed_out = fed_out.permute(0, 2, 1)  # (batch, 64, 16) → 适配池化
        fed_out = self.fed_pool(fed_out).squeeze(-1)  # (batch, 64) → 池化后汇总
        fed_feat = fed_out  # (batch, d_model=64)

        # ---------------------- Concat融合 + 预测 ----------------------
        concat_feat = torch.cat([lstm_feat, fed_feat], dim=1)  # (batch, 64+64=128) → Concat核心
        pred = self.predict_head(concat_feat)  # (batch, 1)

        return pred

# 自定义数据集（BTC/ETH时序数据）
class BTCETH_Dataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# 训练函数（含验证早停，适配BTC/ETH模型）
def train_btc_eth_model(model, train_loader, val_loader, criterion, optimizer, epochs=100, patience=10):
    best_val_loss = float("inf")
    early_stop_cnt = 0
    train_losses, val_losses = [], []

    model.train()
    for epoch in range(epochs):
        # 训练集
        train_loss = 0.0
        for X, y in train_loader:
            y_pred = model(X)
            loss = criterion(y_pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * X.size(0)
        train_avg_loss = train_loss / len(train_loader.dataset)
        train_losses.append(train_avg_loss)

        # 验证集
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X, y in val_loader:
                y_pred = model(X)
                val_loss += criterion(y_pred, y).item() * X.size(0)
        val_avg_loss = val_loss / len(val_loader.dataset)
        val_losses.append(val_avg_loss)

        # 日志与早停
        if (epoch+1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}] | Train MSE: {train_avg_loss:.6f} | Val MSE: {val_avg_loss:.6f}")
        if val_avg_loss < best_val_loss:
            best_val_loss = val_avg_loss
            early_stop_cnt = 0
            torch.save(model.state_dict(), "lstm_fedformer_btc_eth.pth")  # 保存最优模型
        else:
            early_stop_cnt += 1
            if early_stop_cnt >= patience:
                print(f"\n早停触发：验证损失连续{patience}轮未降，最优Val MSE={best_val_loss:.6f}")
                break

    # 绘制损失曲线
    plt.figure(figsize=(10,4))
    plt.plot(train_losses, label="Train MSE", color="blue")
    plt.plot(val_losses, label="Val MSE", color="orange")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("LSTM-FEDFormer (Concat) Training Loss (BTC/ETH)")
    plt.legend()
    plt.show()

    return model


def evaluate_btc_eth_model(model, test_loader, scaler, device):
    """评估模型：输出MSE/RMSE/R²/MAPE（与`btc.docx`/`eth.docx`指标一致）"""
    model.load_state_dict(torch.load("lstm_fedformer_btc_eth.pth"))  # 加载最优模型
    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for X, y in test_loader:
            y_batch_pred = model(X)
            # 反归一化（恢复BTC/ETH真实价格）
            y_true_scaled = np.zeros((len(y), 8))
            y_true_scaled[:, df.columns.get_loc("close")] = y.cpu().numpy().reshape(-1)
            y_true.extend(scaler.inverse_transform(y_true_scaled)[:, df.columns.get_loc("close")])

            y_pred_scaled = np.zeros((len(y_batch_pred), 8))
            y_pred_scaled[:, df.columns.get_loc("close")] = y_batch_pred.cpu().numpy().reshape(-1)
            y_pred.extend(scaler.inverse_transform(y_pred_scaled)[:, df.columns.get_loc("close")])

    # 计算指标
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100

    # 输出结果（与`btc.docx`/`eth.docx`对比）
    print(f"\nLSTM-FEDFormer (Concat) 测试集性能（BTC/ETH）：")
    print(f"MSE: {mse:.2f} | RMSE: {rmse:.2f} | R²: {r2:.4f} | MAPE: {mape:.2f}%")

    # 绘制真实值vs预测值（BTC/ETH价格对比）
    plt.figure(figsize=(12, 4))
    plt.plot(y_true, label="True Close Price", color="blue")
    plt.plot(y_pred, label="Predicted Close Price", color="red", alpha=0.8)
    plt.xlabel("Time Step (Test Set)")
    plt.ylabel("Close Price (USD)")
    plt.title("BTC/ETH True vs Predicted Close Price (LSTM-FEDFormer Concat)")
    plt.legend()
    plt.show()

    return mse, rmse, r2, mape



# 主程序 --------------------------------------------------
if __name__ == "__main__":
    # 加载BTC/ETH数据（二选一，与`btc.docx`/`eth.docx`对应）
    df_btc = load_btc_eth_data("BTC-USDT.csv")  # 对应btc.docx
    # df_eth = load_btc_eth_data("eth_historical_data.csv")  # 对应eth.docx
    df = df_btc  # 当前使用BTC数据（切换ETH需改为df_eth）

    # 生成时序数据（固定16天窗口，匹配BTC/ETH预测任务）
    X_train, X_val, X_test, y_train, y_val, y_test, scaler, device = create_btc_eth_timeseries(df)

    # 数据加载器（batch_size=32，适配BTC/ETH数据量）
    batch_size = 32
    train_ds = BTCETH_Dataset(X_train, y_train)
    val_ds = BTCETH_Dataset(X_val, y_val)
    test_ds = BTCETH_Dataset(X_test, y_test)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    # 模型初始化（固定8特征参数）
    model = LSTM_FEDFormer_Concat(
        input_dim=8,
        seq_len=16,
        d_model=64,
        lstm_hidden=64,
        num_layers=2,
        dropout=0.1
    ).to(device)

    # 损失函数与优化器（MSE与`btc.docx`/`eth.docx`指标一致）
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)

    # 启动训练（适配BTC/ETH数据）
    trained_model = train_btc_eth_model(model, train_loader, val_loader, criterion, optimizer, epochs=100, patience=10)
    # 执行评估（与`btc.docx`/`eth.docx`中LSTM、FEDFormer指标对比）
    test_mse, test_rmse, test_r2, test_mape = evaluate_btc_eth_model(trained_model, test_loader, scaler, device)
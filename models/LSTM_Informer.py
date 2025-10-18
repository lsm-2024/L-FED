import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import os
from datetime import datetime

# 设置随机种子以确保结果可复现
torch.manual_seed(42)
np.random.seed(42)


# 计算技术指标
def calculate_technical_indicators(df):
    """计算技术指标：SMA、EMA、布林带"""
    # 简单移动平均线 (SMA) - 14天
    df['SMA'] = df['close'].rolling(window=14).mean()

    # 指数移动平均线 (EMA) - 14天
    df['EMA'] = df['close'].ewm(span=14, adjust=False).mean()

    # 布林带 (Bollinger Bands) - 20天周期
    df['BB_Middle'] = df['close'].rolling(window=20).mean()
    df['BB_Std'] = df['close'].rolling(window=20).std()
    df['BB_Upper'] = df['BB_Middle'] + 2 * df['BB_Std']  # 上轨
    df['BB_Lower'] = df['BB_Middle'] - 2 * df['BB_Std']  # 下轨

    # 删除包含NaN值的行（由于滚动窗口计算）
    df = df.dropna()

    return df


# 从CSV文件读取数据并添加技术指标
def load_data(file_path, seq_length=30):
    """从CSV文件加载比特币价格数据，计算技术指标并创建创建序列"""
    # 检查文件是否存在
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"数据文件不存在: {file_path}")

    # 读取CSV文件
    df = pd.read_csv(file_path)

    # 确保数据按日期排序
    if 'candle_begin_time' in df.columns:
        df['candle_begin_time'] = pd.to_datetime(df['candle_begin_time'])
        df = df.sort_values('candle_begin_time')

    # 计算技术指标
    print("计算技术指标...")
    df = calculate_technical_indicators(df)

    # 选择所需特征（原始特征 + 技术指标）
    features = [
        'open', 'high', 'low', 'close', 'volume',
        'SMA', 'EMA', 'BB_Upper', 'BB_Middle', 'BB_Lower'
    ]

    if not set(features).issubset(df.columns):
        missing = set(features) - set(df.columns)
        raise ValueError(f"数据文件缺少必要的特征: {missing}")

    data = df[features].values

    # 创建序列和目标
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])  # 序列数据 [seq_length, n_features]
        y.append(data[i + seq_length, 3])  # 目标值为close（在原始特征中索引为3）

    return np.array(X), np.array(y), df['candle_begin_time'].values[seq_length:], features  # 返回日期和特征名用于可视化


# 数据预处理
def prepare_data(X, y, test_size=0.2, batch_size=32):
    """预处理数据并创建数据加载器"""
    # 划分训练集和测试集（时间序列不打乱）
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, shuffle=False
    )

    # 数据标准化 - 对每个特征单独标准化
    n_features = X_train.shape[2]
    feature_scalers = []

    # 重塑数据以便标准化 (samples * seq_length, features)
    X_train_reshaped = X_train.reshape(-1, n_features)
    X_test_reshaped = X_test.reshape(-1, n_features)

    # 标准化每个特征
    for i in range(n_features):
        scaler = StandardScaler()
        X_train_reshaped[:, i] = scaler.fit_transform(X_train_reshaped[:, i].reshape(-1, 1)).flatten()
        X_test_reshaped[:, i] = scaler.transform(X_test_reshaped[:, i].reshape(-1, 1)).flatten()
        feature_scalers.append(scaler)

    # 恢复原始形状
    X_train = X_train_reshaped.reshape(X_train.shape)
    X_test = X_test_reshaped.reshape(X_test.shape)

    # 标准化目标值（close价格）
    target_scaler = StandardScaler()
    y_train_scaled = target_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
    y_test_scaled = target_scaler.transform(y_test.reshape(-1, 1)).flatten()

    # 转换为PyTorch张量 [batch, seq_len, features]
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train_scaled)
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.FloatTensor(y_test_scaled)

    # 创建数据加载器
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return (train_loader, test_loader, X_train, X_test, y_train, y_test,
            target_scaler, feature_scalers)


# Informer的ProbSparse自注意力机制
class ProbSparseAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1):
        super(ProbSparseAttention, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / np.sqrt(E)

        # 计算稀疏注意力
        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        if self.mask_flag:
            mask = torch.ones(L, S, device=queries.device)
            mask = torch.triu(mask, diagonal=1).bool()
            scores = scores.masked_fill(mask[None, None, :, :], -np.inf)

        # 概率稀疏化
        top_k = max(1, int(L / self.factor))
        index = torch.topk(torch.sum(scores, dim=-1), top_k, dim=-1)[1]
        scores = self._mask_with_index(scores, index, top_k)

        attn = torch.softmax(scale * scores, dim=-1)
        attn = self.dropout(attn)
        output = torch.einsum("bhls,bshd->blhd", attn, values)

        return output.contiguous()

    def _mask_with_index(self, scores, index, top_k):
        B, H, L, _ = scores.shape
        mask = torch.zeros_like(scores, dtype=torch.bool)
        for b in range(B):
            for h in range(H):
                mask[b, h, :, index[b, h, :]] = True
        return scores.masked_fill(~mask, -np.inf)


# Informer编码器
class InformerEncoder(nn.Module):
    def __init__(self, input_size, d_model=64, n_heads=2, dropout=0.1):
        super(InformerEncoder, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads

        # 输入投影
        self.input_proj = nn.Linear(input_size, d_model)

        # 自注意力层
        self.attention = ProbSparseAttention(
            mask_flag=True,
            factor=5,
            attention_dropout=dropout
        )

        # 前馈网络
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model)
        )

        # 层归一化
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x shape: [batch_size, seq_len, input_size]

        # 输入投影
        x = self.input_proj(x)  # [batch_size, seq_len, d_model]

        # 自注意力
        residual = x
        attn_output = self.attention(
            x.unsqueeze(2),  # [batch_size, seq_len, 1, d_model]
            x.unsqueeze(2),
            x.unsqueeze(2)
        ).squeeze(2)  # [batch_size, seq_len, d_model]

        x = residual + self.dropout(attn_output)
        x = self.norm1(x)

        # 前馈网络
        residual = x
        x = residual + self.dropout(self.ffn(x))
        x = self.norm2(x)

        return x


# LSTM-Informer混合串行模型
class LSTMInformerModel(nn.Module):
    def __init__(self, input_size, lstm_hidden_size, lstm_num_layers,
                 informer_d_model, informer_n_heads, output_size, dropout=0.2):
        super(LSTMInformerModel, self).__init__()

        # LSTM层
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True,
            dropout=dropout if lstm_num_layers > 1 else 0
        )

        # Informer编码器（接收LSTM的输出）
        self.informer_encoder = InformerEncoder(
            input_size=lstm_hidden_size,
            d_model=informer_d_model,
            n_heads=informer_n_heads,
            dropout=dropout
        )

        # 输出层
        self.fc = nn.Sequential(
            nn.Linear(informer_d_model, informer_d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(informer_d_model // 2, output_size)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x shape: [batch_size, seq_len, input_size]

        # LSTM处理
        lstm_out, _ = self.lstm(x)  # [batch_size, seq_len, lstm_hidden_size]
        lstm_out = self.dropout(lstm_out)

        # Informer处理LSTM的输出
        informer_out = self.informer_encoder(lstm_out)  # [batch_size, seq_len, informer_d_model]

        # 取最后一个时间步的输出
        last_out = informer_out[:, -1, :]  # [batch_size, informer_d_model]

        # 输出预测
        output = self.fc(last_out)  # [batch_size, output_size]

        return output.squeeze()


# 训练模型
def train_model(model, train_loader, criterion, optimizer, device, epochs=100):
    model.train()
    train_losses = []

    for epoch in range(epochs):
        total_loss = 0

        for X_batch, y_batch in train_loader:
            # 将数据移至设备
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            # 前向传播
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # 计算平均损失
        avg_loss = total_loss / len(train_loader)
        train_losses.append(avg_loss)

        # 打印训练进度
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.6f}')

    return model, train_losses


# 评估模型
def evaluate_model(model, test_loader, device):
    model.eval()
    predictions = []
    actuals = []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)

            # 收集预测结果和实际值
            predictions.extend(outputs.cpu().numpy())
            actuals.extend(y_batch.cpu().numpy())

    return np.array(predictions), np.array(actuals)


# 计算评估指标
def calculate_metrics(predictions, actuals):
    """计算MSE, RMSE, MAPE和R²"""
    mse = mean_squared_error(actuals, predictions)
    rmse = np.sqrt(mse)
    # 避免除零错误
    mape = np.mean(np.abs((actuals - predictions) / (actuals + 1e-10))) * 100
    r2 = r2_score(actuals, predictions)

    return {
        'MSE': mse,
        'RMSE': rmse,
        'MAPE': mape,
        'R²': r2
    }


# 可视化结果
def plot_results(predictions, actuals, dates, train_losses, split_idx):
    """可视化训练损失和预测结果"""
    # 绘制训练损失
    plt.figure(figsize=(16, 12))

    plt.subplot(2, 1, 1)
    plt.plot(train_losses)
    plt.title('Training Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.grid(True)

    # 绘制预测结果与实际值
    plt.subplot(2, 1, 2)
    plt.plot(dates, actuals, label='Actual Price', alpha=0.7)
    plt.plot(dates, predictions, label='Predicted Price', alpha=0.7)
    plt.axvline(x=dates[split_idx], color='r', linestyle='--', label='Train-Test Split')
    plt.title('Bitcoin Price Prediction (Close)')
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.show()


# 主函数
def main(file_path='bitcoin_prices.csv', seq_length=30, epochs=100):
    # 设备配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')

    # 加载和准备数据
    print("加载数据...")
    X, y, dates, features = load_data(file_path, seq_length)
    print(f"使用的特征: {features}")
    print(f"数据形状: X={X.shape}, y={y.shape}")

    # 准备数据
    print("预处理数据...")
    (train_loader, test_loader, X_train, X_test, y_train, y_test,
     target_scaler, _) = prepare_data(X, y, batch_size=32)

    # 计算训练测试分割点，用于可视化
    split_idx = len(y_train)

    # 模型参数
    input_size = X_train.shape[2]  # 特征数量（原始5个 + 5个技术指标 = 10个）
    lstm_hidden_size = 64
    lstm_num_layers = 2
    informer_d_model = 32
    informer_n_heads = 2
    output_size = 1
    dropout = 0.2

    # 初始化模型
    model = LSTMInformerModel(
        input_size=input_size,
        lstm_hidden_size=lstm_hidden_size,
        lstm_num_layers=lstm_num_layers,
        informer_d_model=informer_d_model,
        informer_n_heads=informer_n_heads,
        output_size=output_size,
        dropout=dropout
    ).to(device)

    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 训练模型
    print("开始训练模型...")
    trained_model, train_losses = train_model(
        model, train_loader, criterion, optimizer, device, epochs=epochs
    )

    # 评估模型
    print("评估模型...")
    # 获取所有预测（包括训练集和测试集）
    train_predictions_scaled, _ = evaluate_model(trained_model, train_loader, device)
    test_predictions_scaled, _ = evaluate_model(trained_model, test_loader, device)

    # 合并预测结果
    all_predictions_scaled = np.concatenate([train_predictions_scaled, test_predictions_scaled])
    all_actuals = np.concatenate([y_train, y_test])

    # 反标准化
    all_predictions = target_scaler.inverse_transform(all_predictions_scaled.reshape(-1, 1)).flatten()

    # 计算评估指标
    metrics = calculate_metrics(all_predictions, all_actuals)
    print("\n整体评估指标:")
    print(f"MSE: {metrics['MSE']:.6f}")
    print(f"RMSE: {metrics['RMSE']:.6f}")
    print(f"MAPE: {metrics['MAPE']:.2f}%")
    print(f"R²: {metrics['R²']:.6f}")

    # 单独计算测试集指标
    test_predictions = target_scaler.inverse_transform(test_predictions_scaled.reshape(-1, 1)).flatten()
    test_metrics = calculate_metrics(test_predictions, y_test)
    print("\n测试集评估指标:")
    print(f"MSE: {test_metrics['MSE']:.6f}")
    print(f"RMSE: {test_metrics['RMSE']:.6f}")
    print(f"MAPE: {test_metrics['MAPE']:.2f}%")
    print(f"R²: {test_metrics['R²']:.6f}")

    # 可视化结果
    plot_results(all_predictions, all_actuals, dates, train_losses, split_idx)


if __name__ == "__main__":
    # 指定你的CSV文件路径
    # 例如: main(file_path='data/bitcoin_data.csv', seq_length=30, epochs=100)
    main(file_path=r'BTC-USDT.csv', seq_length=64, epochs=100)
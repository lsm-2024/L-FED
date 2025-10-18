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


# 从CSV文件读取数据
def load_data(file_path, seq_length=30):
    """
    从CSV文件加载比特币价格数据并创建序列
    """
    # 检查文件是否存在
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"数据文件不存在: {file_path}")

    # 读取CSV文件
    df = pd.read_csv(file_path)

    # 确保数据按日期排序
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')

    # 选择所需特征
    features = ['open', 'high', 'low', 'close', 'volume']
    if not set(features).issubset(df.columns):
        missing = set(features) - set(df.columns)
        raise ValueError(f"数据文件缺少必要的特征: {missing}")

    data = df[features].values

    # 创建序列和目标
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])  # 序列数据 [seq_length, n_features]
        y.append(data[i + seq_length, 3])  # 目标值为close（在features中索引为3）

    return np.array(X), np.array(y), df['date'].values[seq_length:]  # 返回日期用于可视化


# 数据预处理
def prepare_data(X, y, test_size=0.2, batch_size=32):
    """预处理数据并创建数据加载器"""
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, shuffle=False  # 时间序列不打乱
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


# 定义CNN-LSTM混合模型
class CNNLSTMModel(nn.Module):
    def __init__(self, input_size, cnn_out_channels, kernel_size,
                 lstm_hidden_size, lstm_num_layers, output_size, dropout=0.2):
        super(CNNLSTMModel, self).__init__()

        # 1D卷积层 - 用于提取局部特征
        self.cnn = nn.Conv1d(
            in_channels=input_size,  # 输入通道数 = 特征数
            out_channels=cnn_out_channels,  # 输出通道数
            kernel_size=kernel_size,  # 卷积核大小
            padding=kernel_size // 2  # 保持序列长度不变
        )

        # ReLU激活函数
        self.relu = nn.ReLU()

        # LSTM层 - 用于处理序列信息
        self.lstm = nn.LSTM(
            input_size=cnn_out_channels,  # LSTM输入大小 = CNN输出通道数
            hidden_size=lstm_hidden_size,  # 隐藏层大小
            num_layers=lstm_num_layers,  # LSTM层数
            batch_first=True,  # 输入格式为 [batch, seq_len, features]
            dropout=dropout if lstm_num_layers > 1 else 0  # 仅当多层时使用dropout
        )

        # 全连接层 - 输出预测结果
        self.fc = nn.Linear(lstm_hidden_size, output_size)

        # Dropout层 - 防止过拟合
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x shape: [batch_size, seq_len, input_size]

        # 调整形状以适应CNN: [batch_size, input_size, seq_len]
        x = x.permute(0, 2, 1)

        # CNN特征提取
        x = self.cnn(x)  # [batch_size, cnn_out_channels, seq_len]

        # 恢复形状以适应LSTM: [batch_size, seq_len, cnn_out_channels]
        x = x.permute(0, 2, 1)

        # 激活函数
        x = self.relu(x)

        # LSTM处理
        lstm_out, _ = self.lstm(x)  # [batch_size, seq_len, lstm_hidden_size]

        # 只使用最后一个时间步的输出
        last_out = lstm_out[:, -1, :]  # [batch_size, lstm_hidden_size]

        # Dropout
        last_out = self.dropout(last_out)

        # 输出层
        output = self.fc(last_out)  # [batch_size, output_size]

        return output.squeeze()  # 去除维度为1的维度


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
    X, y, dates = load_data(file_path, seq_length)

    # 准备数据
    print("预处理数据...")
    (train_loader, test_loader, X_train, X_test, y_train, y_test,
     target_scaler, _) = prepare_data(X, y, batch_size=32)

    # 计算训练测试分割点，用于可视化
    split_idx = len(y_train)

    # 模型参数
    input_size = X_train.shape[2]  # 特征数量（5个: open, high, low, close, volume）
    cnn_out_channels = 32
    kernel_size = 3
    lstm_hidden_size = 64
    lstm_num_layers = 2
    output_size = 1
    dropout = 0.2

    # 初始化模型
    model = CNNLSTMModel(
        input_size=input_size,
        cnn_out_channels=cnn_out_channels,
        kernel_size=kernel_size,
        lstm_hidden_size=lstm_hidden_size,
        lstm_num_layers=lstm_num_layers,
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
    print("\n评估指标:")
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
    # 可以在这里指定你的CSV文件路径
    # 例如: main(file_path='path/to/your/bitcoin_data.csv', seq_length=30, epochs=100)
    main(file_path=r'C:\data\bitcoin_data2.csv', seq_length=30, epochs=100)

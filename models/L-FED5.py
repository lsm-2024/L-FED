import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt

import time
from datetime import datetime

# 设置随机种子，确保结果可复现
torch.manual_seed(42)
np.random.seed(42)


# 数据预处理类
class StockDataset(Dataset):
    def __init__(self, data, seq_len, pred_len, scaler=None):
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.data = data
        self.scaler = scaler

        # 如果没有提供scaler，则创建一个并拟合数据
        if self.scaler is None:
            self.scaler = MinMaxScaler(feature_range=(0, 1))
            self.data_scaled = self.scaler.fit_transform(self.data)
        else:
            self.data_scaled = self.scaler.transform(self.data)

    def __len__(self):
        return len(self.data_scaled) - self.seq_len - self.pred_len + 1

    def __getitem__(self, idx):
        # 获取输入序列
        x = self.data_scaled[idx:idx + self.seq_len, :]
        # 获取目标序列（预测收盘价）
        y = self.data_scaled[idx + self.seq_len:idx + self.seq_len + self.pred_len, 1]  # close在第2列（索引1）

        return torch.FloatTensor(x), torch.FloatTensor(y)


# LSTM模型
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # 初始化隐藏状态
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # LSTM前向传播
        out, _ = self.lstm(x, (h0, c0))

        # 取最后一个时间步的输出进行预测
        out = self.fc(out[:, -1, :])
        return out


# FEDformer的位置编码（简化版）
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (seq_len, batch_size, d_model)
        x = x + self.pe[:x.size(0)]
        return x


# FEDformer模型（简化版）
class FEDformerModel(nn.Module):
    def __init__(self, input_size, d_model, n_heads, num_layers, output_size, dropout=0.2):
        super(FEDformerModel, self).__init__()
        self.d_model = d_model

        # 输入投影，将特征维度映射到d_model
        self.input_proj = nn.Linear(input_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)

        # 多头注意力层
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)

        # 输出层
        self.fc = nn.Linear(d_model, output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # 输入投影并添加位置编码
        x = self.input_proj(x) * np.sqrt(self.d_model)
        x = self.pos_encoder(x.transpose(0, 1)).transpose(0, 1)
        x = self.dropout(x)

        # Transformer编码器
        out = self.transformer_encoder(x)

        # 取最后一个时间步的输出进行预测
        out = self.fc(out[:, -1, :])
        return out


# 动态权值融合的混合模型
class HybridModel(nn.Module):
    def __init__(self, lstm_input_size, lstm_hidden_size, lstm_layers,
                 fed_input_size, fed_d_model, fed_heads, fed_layers,
                 output_size, dropout=0.2):
        super(HybridModel, self).__init__()

        # LSTM子模型
        self.lstm = LSTMModel(
            input_size=lstm_input_size,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_layers,
            output_size=output_size,
            dropout=dropout
        )

        # FEDformer子模型（输入尺寸增加1，用于接收LSTM的预测作为指导价格）
        self.fedformer = FEDformerModel(
            input_size=fed_input_size + 1,  # +1 是因为添加了LSTM的预测作为指导价格
            d_model=fed_d_model,
            n_heads=fed_heads,
            num_layers=fed_layers,
            output_size=output_size,
            dropout=dropout
        )

        # 动态权值层，用于融合两个模型的输出
        self.attention = nn.Sequential(
            nn.Linear(output_size * 2, output_size),
            nn.Tanh(),
            nn.Linear(output_size, 2),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        # LSTM预测
        lstm_pred = self.lstm(x)

        # 将LSTM的预测作为指导价格添加到FEDformer的输入中
        # 扩展维度以匹配序列长度
        lstm_guide = lstm_pred.unsqueeze(1).repeat(1, x.size(1), 1)
        fed_input = torch.cat([x, lstm_guide], dim=2)

        # FEDformer预测
        fed_pred = self.fedformer(fed_input)

        # 动态权值融合
        combined = torch.cat([lstm_pred, fed_pred], dim=1)
        weights = self.attention(combined)

        # 加权求和得到最终预测
        final_pred = weights[:, 0:1] * lstm_pred + weights[:, 1:2] * fed_pred

        return final_pred, lstm_pred, fed_pred, weights


# 训练函数
def train_model(model, train_loader, val_loader, criterion, optimizer, device, epochs=50, patience=5):
    model.train()
    best_val_loss = float('inf')
    early_stopping_counter = 0
    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        start_time = time.time()
        train_loss = 0.0

        # 训练
        model.train()
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            optimizer.zero_grad()

            # 前向传播
            final_pred, _, _, _ = model(batch_x)

            # 计算损失
            loss = criterion(final_pred.squeeze(), batch_y.squeeze())

            # 反向传播和优化
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * batch_x.size(0)

        # 计算平均训练损失
        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)

        # 验证
        val_loss = 0.0
        model.eval()
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)

                final_pred, _, _, _ = model(batch_x)
                loss = criterion(final_pred.squeeze(), batch_y.squeeze())
                val_loss += loss.item() * batch_x.size(0)

        # 计算平均验证损失
        val_loss /= len(val_loader.dataset)
        val_losses.append(val_loss)

        # 打印 epoch 信息
        epoch_time = time.time() - start_time
        print(
            f'Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, Time: {epoch_time:.2f}s')

        # 早停机制
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stopping_counter = 0
            # 保存最佳模型
            torch.save(model.state_dict(), 'best_hybrid_model.pth')
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

    return model, train_losses, val_losses


# 测试函数
def test_model(model, test_loader, criterion, scaler, device):
    model.eval()
    test_loss = 0.0
    all_preds = []
    all_actual = []
    all_lstm_preds = []
    all_fed_preds = []
    all_weights = []

    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            final_pred, lstm_pred, fed_pred, weights = model(batch_x)

            # 计算损失
            loss = criterion(final_pred.squeeze(), batch_y.squeeze())
            test_loss += loss.item() * batch_x.size(0)

            # 保存预测结果和实际值
            all_preds.extend(final_pred.cpu().numpy())
            all_actual.extend(batch_y.cpu().numpy())
            all_lstm_preds.extend(lstm_pred.cpu().numpy())
            all_fed_preds.extend(fed_pred.cpu().numpy())
            all_weights.extend(weights.cpu().numpy())

    # 计算平均测试损失
    test_loss /= len(test_loader.dataset)

    # 反归一化（假设收盘价是第二个特征）
    # 创建一个空数组用于反归一化
    dummy = np.zeros((len(all_preds), scaler.n_features_in_))
    dummy[:, 1] = np.array(all_preds).flatten()  # 假设close在第2列（索引1）
    all_preds_denorm = scaler.inverse_transform(dummy)[:, 1]

    dummy_actual = np.zeros((len(all_actual), scaler.n_features_in_))
    dummy_actual[:, 1] = np.array(all_actual).flatten()
    all_actual_denorm = scaler.inverse_transform(dummy_actual)[:, 1]

    dummy_lstm = np.zeros((len(all_lstm_preds), scaler.n_features_in_))
    dummy_lstm[:, 1] = np.array(all_lstm_preds).flatten()
    all_lstm_denorm = scaler.inverse_transform(dummy_lstm)[:, 1]

    dummy_fed = np.zeros((len(all_fed_preds), scaler.n_features_in_))
    dummy_fed[:, 1] = np.array(all_fed_preds).flatten()
    all_fed_denorm = scaler.inverse_transform(dummy_fed)[:, 1]

    return (test_loss, all_preds_denorm, all_actual_denorm,
            all_lstm_denorm, all_fed_denorm, all_weights)


# 主函数
def main():
    # 配置参数
    config = {
        'seq_len': 60,  # 输入序列长度
        'pred_len': 1,  # 预测长度
        'lstm_hidden_size': 64,  # LSTM隐藏层大小
        'lstm_layers': 2,  # LSTM层数
        'fed_d_model': 64,  # FEDformer特征维度
        'fed_heads': 4,  # FEDformer注意力头数
        'fed_layers': 2,  # FEDformer层数
        'output_size': 1,  # 输出尺寸（预测收盘价）
        'dropout': 0.2,  # dropout率
        'batch_size': 32,  # 批处理大小
        'epochs': 100,  # 训练轮数
        'lr': 0.001,  # 学习率
        'patience': 10,  # 早停耐心值
        'test_size': 0.2,  # 测试集比例
        'val_size': 0.2  # 验证集比例（相对于训练集）
    }

    # 设备配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 加载数据
    print("加载数据...")
    df = pd.read_csv('BTC-USDT2.csv')  # 替换为你的CSV文件路径

    # 检查并转换日期列
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        # 可以从日期中提取特征，但这里我们只使用数值特征
        # 提取开盘价、收盘价、最高价、最低价、成交量
        features = df[['open', 'close', 'high', 'low', 'volume']].values
    else:
        raise ValueError("数据中没有找到'date'列")

    # 划分训练集、验证集和测试集
    print("划分数据集...")
    # 先划分训练集和临时集
    train_data, temp_data = train_test_split(features, test_size=config['test_size'], shuffle=False)
    # 再从临时集中划分验证集
    train_data, val_data = train_test_split(train_data, test_size=config['val_size'], shuffle=False)

    # 创建数据集
    train_dataset = StockDataset(train_data, config['seq_len'], config['pred_len'])
    val_dataset = StockDataset(val_data, config['seq_len'], config['pred_len'], scaler=train_dataset.scaler)
    test_dataset = StockDataset(temp_data, config['seq_len'], config['pred_len'], scaler=train_dataset.scaler)

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)

    # 获取输入特征数量
    n_features = features.shape[1]

    # 初始化模型
    print("初始化模型...")
    model = HybridModel(
        lstm_input_size=n_features,
        lstm_hidden_size=config['lstm_hidden_size'],
        lstm_layers=config['lstm_layers'],
        fed_input_size=n_features,
        fed_d_model=config['fed_d_model'],
        fed_heads=config['fed_heads'],
        fed_layers=config['fed_layers'],
        output_size=config['output_size'],
        dropout=config['dropout']
    ).to(device)

    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config['lr'])

    # 训练模型
    print("开始训练...")
    model, train_losses, val_losses = train_model(
        model, train_loader, val_loader, criterion, optimizer,
        device, config['epochs'], config['patience']
    )

    # 加载最佳模型权重
    model.load_state_dict(torch.load('best_hybrid_model.pth'))

    # 测试模型
    print("测试模型...")
    test_loss, preds, actual, lstm_preds, fed_preds, weights = test_model(
        model, test_loader, criterion, train_dataset.scaler, device
    )
    print(f"测试损失: {test_loss:.6f}")

    # 绘制损失曲线
    plt.figure(figsize=(12, 6))
    plt.plot(train_losses, label='训练损失')
    plt.plot(val_losses, label='验证损失')
    plt.title('训练和验证损失曲线')
    plt.xlabel('Epoch')
    plt.ylabel('损失')
    plt.legend()
    plt.savefig('loss_curve.png')
    plt.close()

    # 绘制预测结果（取最后100个样本）
    plt.figure(figsize=(15, 8))
    plt.plot(actual[-100:], label='实际收盘价', color='blue')
    plt.plot(preds[-100:], label='混合模型预测', color='red', linestyle='--')
    plt.plot(lstm_preds[-100:], label='LSTM预测', color='green', alpha=0.5)
    plt.plot(fed_preds[-100:], label='FEDformer预测', color='purple', alpha=0.5)
    plt.title('股票收盘价预测')
    plt.xlabel('时间步')
    plt.ylabel('收盘价')
    plt.legend()
    plt.savefig('prediction_comparison.png')
    plt.close()

    # 绘制动态权重
    plt.figure(figsize=(15, 6))
    weights_np = np.array(weights)
    plt.plot(weights_np[-100:, 0], label='LSTM权重', color='green')
    plt.plot(weights_np[-100:, 1], label='FEDformer权重', color='purple')
    plt.title('动态融合权重')
    plt.xlabel('时间步')
    plt.ylabel('权重值')
    plt.legend()
    plt.savefig('dynamic_weights.png')
    plt.close()

    print("所有结果已保存为图片文件")


if __name__ == "__main__":
    main()

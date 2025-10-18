import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)


class CryptoDataset(Dataset):
    def __init__(self, data, sequence_length=64, prediction_horizon=1):
        self.data = data
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon

    def __len__(self):
        return len(self.data) - self.sequence_length - self.prediction_horizon + 1

    def __getitem__(self, idx):
        x = self.data[idx:idx + self.sequence_length]  # 历史64天数据
        y = self.data[idx + self.sequence_length + self.prediction_horizon - 1, 1]  # 未来1天收盘价
        return torch.FloatTensor(x), torch.FloatTensor([y])


class FEDformerEncoder(nn.Module):
    def __init__(self, d_model=64, n_heads=8, num_layers=2, dropout=0.1):
        super(FEDformerEncoder, self).__init__()
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=d_model * 4,
                dropout=dropout,
                batch_first=True
            )
            for _ in range(num_layers)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class EnhancedFEDformer(nn.Module):
    def __init__(self, input_dim=5, d_model=64, n_heads=8, num_layers=2, dropout=0.1):
        super(EnhancedFEDformer, self).__init__()

        # 原始特征投影
        self.input_projection = nn.Linear(input_dim, d_model)

        # LSTM指导价格特征投影
        self.guide_price_projection = nn.Linear(1, d_model // 2)

        # 特征融合投影
        self.feature_fusion = nn.Linear(d_model + d_model // 2, d_model)

        # FEDformer编码器
        self.encoder = FEDformerEncoder(d_model, n_heads, num_layers, dropout)

        # 输出层
        self.output_layer = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )

    def forward(self, x, guide_price):
        """
        x: 原始特征 (batch, seq_len, input_dim)
        guide_price: LSTM指导价格 (batch,)
        """
        batch_size, seq_len, _ = x.shape

        # 原始特征投影
        original_features = self.input_projection(x)  # (batch, seq_len, d_model)

        # 处理指导价格特征
        guide_expanded = guide_price.unsqueeze(-1)  # (batch, 1)
        guide_features = self.guide_price_projection(guide_expanded)  # (batch, d_model//2)
        guide_features = guide_features.unsqueeze(1).repeat(1, seq_len, 1)  # (batch, seq_len, d_model//2)

        # 特征融合
        combined_features = torch.cat([original_features, guide_features], dim=-1)
        fused_features = self.feature_fusion(combined_features)

        # FEDformer编码
        encoded = self.encoder(fused_features)

        # 输出预测
        last_output = encoded[:, -1, :]
        output = self.output_layer(last_output)

        return output.squeeze(-1)


class LSTMModel(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=64, num_layers=2, dropout=0.1):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.dropout = nn.Dropout(dropout)
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]
        output = self.output_layer(last_hidden)
        return output.squeeze(-1)


class DynamicWeightFusion(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=32):
        super(DynamicWeightFusion, self).__init__()

        # 动态权重生成网络
        self.weight_network = nn.Sequential(
            nn.Linear(input_dim * 64, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 2),
            nn.Softmax(dim=-1)
        )

    def forward(self, x, lstm_pred, fedformer_pred):
        batch_size = x.shape[0]

        # 展平输入特征用于权重计算
        x_flat = x.reshape(batch_size, -1)

        # 生成动态权重
        weights = self.weight_network(x_flat)  # (batch, 2)

        # 加权融合
        final_pred = weights[:, 0] * lstm_pred + weights[:, 1] * fedformer_pred

        return final_pred, weights


class LSTM_FEDformer_Hybrid(nn.Module):
    def __init__(self, input_dim=5, lstm_hidden=64, fedformer_dim=64,
                 num_layers=2, n_heads=8, dropout=0.1):
        super(LSTM_FEDformer_Hybrid, self).__init__()

        # LSTM分支
        self.lstm_branch = LSTMModel(input_dim, lstm_hidden, num_layers, dropout)

        # 增强版FEDformer分支（包含LSTM指导价格）
        self.fedformer_branch = EnhancedFEDformer(input_dim, fedformer_dim, n_heads, num_layers, dropout)

        # 动态权值融合
        self.fusion_layer = DynamicWeightFusion(input_dim)

    def forward(self, x):
        # LSTM预测
        lstm_pred = self.lstm_branch(x)

        # FEDformer预测（使用LSTM预测作为指导价格）
        fedformer_pred = self.fedformer_branch(x, lstm_pred)

        # 动态权值融合
        final_pred, weights = self.fusion_layer(x, lstm_pred, fedformer_pred)

        return final_pred, lstm_pred, fedformer_pred, weights


class BitcoinPricePredictor:
    def __init__(self, sequence_length=64, prediction_horizon=1):
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.scaler = StandardScaler()
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

    def load_and_preprocess_data(self, data_path):
        """加载和预处理比特币数据"""
        # 读取数据
        df = pd.read_csv(data_path)
        print(f"原始数据形状: {df.shape}")

        # 确保日期列正确解析
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date').reset_index(drop=True)

        # 选择特征列
        feature_columns = ['open', 'close', 'high', 'low', 'volume']
        data = df[feature_columns].values

        print(f"数据范围: {len(data)} 天")
        print(f"特征数量: {len(feature_columns)}")

        # 数据标准化
        data_scaled = self.scaler.fit_transform(data)

        # 数据集划分 (6:2:2)
        n = len(data_scaled)
        train_size = int(0.6 * n)
        val_size = int(0.2 * n)

        train_data = data_scaled[:train_size]
        val_data = data_scaled[train_size:train_size + val_size]
        test_data = data_scaled[train_size + val_size:]

        print(f"训练集: {len(train_data)} 样本")
        print(f"验证集: {len(val_data)} 样本")
        print(f"测试集: {len(test_data)} 样本")

        # 创建数据集
        train_dataset = CryptoDataset(train_data, self.sequence_length, self.prediction_horizon)
        val_dataset = CryptoDataset(val_data, self.sequence_length, self.prediction_horizon)
        test_dataset = CryptoDataset(test_data, self.sequence_length, self.prediction_horizon)

        return train_dataset, val_dataset, test_dataset, df

    def create_model(self, input_dim=5):
        """创建混合模型"""
        model = LSTM_FEDformer_Hybrid(
            input_dim=input_dim,
            lstm_hidden=64,
            fedformer_dim=64,
            num_layers=2,
            n_heads=8,
            dropout=0.1
        )
        return model.to(self.device)

    def train_model(self, train_loader, val_loader, epochs=100, lr=0.001):
        """训练模型"""
        self.model = self.create_model()
        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-5)
        criterion = nn.MSELoss()

        train_losses = []
        val_losses = []
        best_val_loss = float('inf')

        print("开始训练混合模型...")

        for epoch in range(epochs):
            # 训练阶段
            self.model.train()
            train_loss = 0
            for batch_idx, (batch_x, batch_y) in enumerate(train_loader):
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)

                optimizer.zero_grad()
                final_pred, lstm_pred, fedformer_pred, weights = self.model(batch_x)
                loss = criterion(final_pred, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()

                train_loss += loss.item()

            # 验证阶段
            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                    final_pred, _, _, _ = self.model(batch_x)
                    loss = criterion(final_pred, batch_y)
                    val_loss += loss.item()

            train_loss /= len(train_loader)
            val_loss /= len(val_loader)

            train_losses.append(train_loss)
            val_losses.append(val_loss)

            # 保存最佳模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': epoch,
                    'val_loss': val_loss
                }, 'best_bitcoin_model.pth')

            if epoch % 10 == 0:
                print(f'Epoch {epoch:3d}/{epochs} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}')

        # 绘制训练曲线
        self.plot_training_curve(train_losses, val_losses)

        # 加载最佳模型
        checkpoint = torch.load('best_bitcoin_model.pth')
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"训练完成! 最佳验证损失: {checkpoint['val_loss']:.6f}")

    def evaluate_model(self, test_loader):
        """评估模型性能"""
        self.model.eval()
        all_predictions = []
        all_actuals = []
        all_lstm_pred = []
        all_fedformer_pred = []
        all_weights = []

        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                final_pred, lstm_pred, fedformer_pred, weights = self.model(batch_x)

                all_predictions.extend(final_pred.cpu().numpy())
                all_actuals.extend(batch_y.cpu().numpy())
                all_lstm_pred.extend(lstm_pred.cpu().numpy())
                all_fedformer_pred.extend(fedformer_pred.cpu().numpy())
                all_weights.extend(weights.cpu().numpy())

        # 转换为numpy数组
        predictions = np.array(all_predictions)
        actuals = np.array(all_actuals)
        lstm_pred = np.array(all_lstm_pred)
        fedformer_pred = np.array(all_fedformer_pred)
        weights = np.array(all_weights)

        # 反标准化
        predictions_actual = self.inverse_transform_predictions(predictions)
        actuals_actual = self.inverse_transform_predictions(actuals)
        lstm_pred_actual = self.inverse_transform_predictions(lstm_pred)
        fedformer_pred_actual = self.inverse_transform_predictions(fedformer_pred)

        # 计算评估指标
        self.calculate_metrics(actuals_actual, predictions_actual, lstm_pred_actual, fedformer_pred_actual, weights)

        # 绘制结果
        self.plot_results(actuals_actual, predictions_actual, lstm_pred_actual, fedformer_pred_actual, weights)

        return predictions_actual, actuals_actual, weights

    def inverse_transform_predictions(self, predictions):
        """反标准化预测结果"""
        # 确保 predictions 是一维数组
        if len(predictions.shape) > 1:
            predictions = predictions.flatten()

        dummy = np.zeros((len(predictions), 5))
        dummy[:, 1] = predictions  # 收盘价在第二列
        predictions_actual = self.scaler.inverse_transform(dummy)[:, 1]
        return predictions_actual

    def calculate_metrics(self, actuals, predictions, lstm_pred, fedformer_pred, weights):
        """计算评估指标"""
        # 基础指标
        mse = mean_squared_error(actuals, predictions)
        mae = mean_absolute_error(actuals, predictions)
        rmse = np.sqrt(mse)
        r2 = r2_score(actuals, predictions)

        # 方向准确率
        pred_direction = np.diff(predictions) > 0
        actual_direction = np.diff(actuals) > 0
        direction_accuracy = np.mean(pred_direction == actual_direction)

        print("\n" + "=" * 60)
        print("模型评估结果")
        print("=" * 60)
        print(f"混合模型 - MSE: {mse:.4f}, MAE: {mae:.4f}, RMSE: {rmse:.4f}")
        print(f"混合模型 - R² Score: {r2:.4f}")
        print(f"混合模型 - 方向准确率: {direction_accuracy:.4f}")
        print(f"平均LSTM权重: {weights[:, 0].mean():.4f}")
        print(f"平均FEDformer权重: {weights[:, 1].mean():.4f}")

        # 比较单个模型性能
        lstm_mse = mean_squared_error(actuals, lstm_pred)
        fedformer_mse = mean_squared_error(actuals, fedformer_pred)
        print(f"LSTM单独 - MSE: {lstm_mse:.4f}")
        print(f"FEDformer单独 - MSE: {fedformer_mse:.4f}")
        print(f"混合模型相对LSTM提升: {((lstm_mse - mse) / lstm_mse * 100):.2f}%")
        print(f"混合模型相对FEDformer提升: {((fedformer_mse - mse) / fedformer_mse * 100):.2f}%")

    def plot_training_curve(self, train_losses, val_losses):
        """绘制训练曲线"""
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label='Training Loss', linewidth=2)
        plt.plot(val_losses, label='Validation Loss', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('LSTM+FEDformer混合模型训练过程')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig('training_curve.png', dpi=300, bbox_inches='tight')
        plt.close()

    def plot_results(self, actuals, predictions, lstm_pred, fedformer_pred, weights):
        """绘制预测结果"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # 1. 价格预测对比
        axes[0, 0].plot(actuals[:200], label='实际价格', color='black', linewidth=2)
        axes[0, 0].plot(predictions[:200], label='混合模型预测', color='red', alpha=0.8)
        axes[0, 0].set_title('比特币价格预测对比')
        axes[0, 0].set_xlabel('时间')
        axes[0, 0].set_ylabel('价格')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # 2. 各模型预测对比
        axes[0, 1].plot(actuals[:100], label='实际价格', color='black', linewidth=2)
        axes[0, 1].plot(predictions[:100], label='混合模型', color='red', alpha=0.9)
        axes[0, 1].plot(lstm_pred[:100], label='LSTM', color='blue', alpha=0.6)
        axes[0, 1].plot(fedformer_pred[:100], label='FEDformer', color='green', alpha=0.6)
        axes[0, 1].set_title('各模型预测对比')
        axes[0, 1].set_xlabel('时间')
        axes[0, 1].set_ylabel('价格')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # 3. 权重分布
        axes[1, 0].hist(weights[:, 0], bins=50, alpha=0.7, label='LSTM权重', color='blue')
        axes[1, 0].hist(weights[:, 1], bins=50, alpha=0.7, label='FEDformer权重', color='green')
        axes[1, 0].set_title('动态权重分布')
        axes[1, 0].set_xlabel('权重值')
        axes[1, 0].set_ylabel('频次')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # 4. 权重随时间变化
        axes[1, 1].plot(weights[:200, 0], label='LSTM权重', color='blue', alpha=0.7)
        axes[1, 1].plot(weights[:200, 1], label='FEDformer权重', color='green', alpha=0.7)
        axes[1, 1].set_title('动态权重变化')
        axes[1, 1].set_xlabel('时间步')
        axes[1, 1].set_ylabel('权重值')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('prediction_results.png', dpi=300, bbox_inches='tight')
        plt.close()

def main():
    # 初始化预测器
    predictor = BitcoinPricePredictor(sequence_length=64, prediction_horizon=1)

    # 加载数据 - 请替换为您的比特币数据文件路径
    data_path = "BTC-USDT2.csv"  # 替换为您的实际文件路径

    try:
        train_dataset, val_dataset, test_dataset, original_df = predictor.load_and_preprocess_data(data_path)
    except FileNotFoundError:
        print(f"错误: 找不到数据文件 {data_path}")
        print("请确保文件路径正确，且包含以下列: date, open, close, high, low, volume")
        return

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)

    print(f"\n数据加载完成!")
    print(f"训练批次: {len(train_loader)}, 验证批次: {len(val_loader)}, 测试批次: {len(test_loader)}")

    # 训练模型
    predictor.train_model(train_loader, val_loader, epochs=50, lr=0.001)

    # 评估模型
    print("\n开始模型评估...")
    predictions, actuals, weights = predictor.evaluate_model(test_loader)

    # 保存结果
    results_df = pd.DataFrame({
        'Actual_Price': actuals,
        'Predicted_Price': predictions,
        'LSTM_Weight': weights[:, 0],
        'FEDformer_Weight': weights[:, 1]
    })
    results_df.to_csv('bitcoin_prediction_results.csv', index=False)
    print("\n预测结果已保存至: bitcoin_prediction_results.csv")


if __name__ == "__main__":
    main()
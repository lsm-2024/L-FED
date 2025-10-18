import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

import dataprocess
from config import DATA_CONFIG


class BiLSTMPredictor(nn.Module):
    """改进的BiLSTM时序预测模型"""

    def __init__(self, input_size=6, hidden_size=128, num_layers=2, dropout=0.3):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # 双向LSTM层
        self.bilstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # 注意力机制
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, 64),  # 双向拼接后维度翻倍
            nn.Tanh(),
            nn.Linear(64, 1),
            nn.Softmax(dim=1)
        )

        # 输出层
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

        # 初始化参数
        self._init_weights()

    def _init_weights(self):
        for name, param in self.bilstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_normal_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0.01)

    def forward(self, x):
        # 双向LSTM
        bilstm_out, _ = self.bilstm(x)  # [batch, seq_len, hidden_size*2]

        # 注意力机制
        attn_weights = self.attention(bilstm_out)  # [batch, seq_len, 1]
        context = torch.sum(bilstm_out * attn_weights, dim=1)  # [batch, hidden_size*2]

        # 最终预测
        return self.fc(context)


# 训练配置（与数据处理配置保持一致）
CONFIG = {
    "input_size": 6,  # 输入特征维度
    "hidden_size": 128,  # 隐藏单元数
    "num_layers": 2,  # BiLSTM层数
    "dropout": 0.5,  # Dropout概率
    "learning_rate": 0.0005,
    "weight_decay": 0.001,  # L2正则化
    "num_epochs": 100,
    "batch_size": 64,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu")
}


def train_bilstm(model, train_loader, test_loader, target_scaler):
    """优化后的训练流程"""
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=CONFIG["learning_rate"],
        weight_decay=CONFIG["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        patience=5,
        factor=0.5,
        verbose=True
    )

    best_loss = float('inf')
    train_losses = []
    val_losses = []

    for epoch in range(CONFIG["num_epochs"]):
        # 训练阶段
        model.train()
        epoch_train_loss = 0
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(CONFIG["device"])
            batch_y = batch_y.to(CONFIG["device"])

            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()

        # 验证阶段
        model.eval()
        epoch_val_loss = 0
        with torch.no_grad():
            for val_x, val_y in test_loader:
                val_x = val_x.to(CONFIG["device"])
                val_y = val_y.to(CONFIG["device"])
                preds = model(val_x)
                epoch_val_loss += criterion(preds, val_y).item()

        # 计算平均损失
        avg_train_loss = epoch_train_loss / len(train_loader)
        avg_val_loss = epoch_val_loss / len(test_loader)
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)

        # 更新学习率
        scheduler.step(avg_val_loss)

        # 早停机制
        # if avg_val_loss < best_loss:
        #     best_loss = avg_val_loss
        #     torch.save(model.state_dict(), "best_bilstm.pth")
        #     patience_counter = 0
        # else:
        #     patience_counter += 1
        #     if patience_counter >= 10:
        #         print(f"Early stopping at epoch {epoch + 1}")
        #         break

        # 打印训练进度
        if (epoch + 1) % 5 == 0:
            print(f"Epoch [{epoch + 1}/{CONFIG['num_epochs']}] | "
                  f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

    # 绘制损失曲线
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training Progress')
    plt.xlabel('Epochs')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.show()


def evaluate_bilstm(model, test_loader, target_scaler):
    """改进的评估函数"""
    model.eval()
    predictions = []
    true_values = []

    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(CONFIG["device"])
            batch_pred = model(batch_x).cpu().numpy()

            # 反标准化
            inv_pred = target_scaler.inverse_transform(batch_pred)
            inv_true = target_scaler.inverse_transform(batch_y.numpy())

            predictions.extend(inv_pred.flatten())
            true_values.extend(inv_true.flatten())

    # 计算指标
    mse = mean_squared_error(true_values, predictions)
    rmse = np.sqrt(mse)

    # 可视化对比
    plt.figure(figsize=(14, 6))
    plt.plot(true_values[:200], label='True Values', alpha=0.7)
    plt.plot(predictions[:200], label='Predictions', linestyle='--')
    plt.title(f"Prediction Comparison (RMSE: {rmse:.2f})")
    plt.xlabel("Time Steps")
    plt.ylabel("Price")
    plt.legend()
    plt.show()

    return rmse


# 主程序
if __name__ == "__main__":
    # 加载预处理数据（使用之前的数据处理函数）
    train_loader, test_loader, processed_info = dataprocess.process_financial_data(DATA_CONFIG)
    target_scaler = processed_info["scaler"]

    # 初始化模型
    model = BiLSTMPredictor(
        input_size=CONFIG["input_size"],
        hidden_size=CONFIG["hidden_size"],
        num_layers=CONFIG["num_layers"],
        dropout=CONFIG["dropout"]
    ).to(CONFIG["device"])

    print(model)

    # 训练模型
    train_bilstm(model, train_loader, test_loader, target_scaler)

    # 加载最佳模型
    # model.load_state_dict(torch.load("best_bilstm.pth"))

    # 评估模型
    final_rmse = evaluate_bilstm(model, test_loader, target_scaler)
    print(f"Final Test RMSE: {final_rmse:.2f}")
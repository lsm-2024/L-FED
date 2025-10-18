import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from torch.cuda.amp import GradScaler

from config import DATA_CONFIG
from dataprocess import process_financial_data
from evaluate import evaluate_model_origin


class TimeSeriesCNN(nn.Module):
    """适用于金融时序预测的CNN模型"""

    def __init__(self, input_channels=6, seq_length=30, pred_length=1):
        super().__init__()

        # 卷积特征提取模块
        self.conv_block = nn.Sequential(
            # 第一卷积层
            nn.Conv1d(input_channels, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),

            # 第二卷积层（残差连接）
            ResidualBlock(64, 128, kernel_size=3),

            # 第三卷积层
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )

        # 计算卷积后的序列长度
        self.flatten_size = self._get_flatten_size(seq_length)

        # 全连接预测模块
        self.fc = nn.Sequential(
            nn.Linear(256 * self.flatten_size, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, pred_length))

    def _get_flatten_size(self, seq_len):
        # 模拟计算卷积后的尺寸
        def conv_out(h_in, pool=False):
            h_out = h_in  # 卷积padding保持尺寸
            if pool: h_out = h_out // 2
            return h_out

        l = seq_len
        l = conv_out(l)  # 第一层卷积
        l = conv_out(l, True)  # 第一层池化
        l = conv_out(l)  # 第二层卷积（残差）
        l = conv_out(l, True)  # 第三层池化
        return l

    def forward(self, x):
        # 输入形状: (batch, seq_len, channels)
        x = x.permute(0, 2, 1)  # 转换为 (batch, channels, seq_len)

        # 特征提取
        features = self.conv_block(x)

        # 展平
        features = features.view(features.size(0), -1)

        # 预测输出
        return self.fc(features)


class ResidualBlock(nn.Module):
    """残差模块"""

    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=1)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=1)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        residual = self.shortcut(x)
        out = nn.ReLU()(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return nn.ReLU()(out)


# 训练配置
CONFIG = {
    "input_channels": 6,  # 特征数量
    "seq_length": 60,  # 输入序列长度
    "pred_length": 1,  # 预测步长
    "batch_size": 64,
    "learning_rate": 0.0005,
    "weight_decay": 0.001,  # L2正则化
    "num_epochs": 100,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu")
}


def train_cnn(model, train_loader, test_loader, target_scaler):
    """优化的训练流程"""
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=CONFIG["learning_rate"],
        weight_decay=CONFIG["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=0.001,
        steps_per_epoch=len(train_loader),
        epochs=CONFIG["num_epochs"]
    )

    best_rmse = float('inf')
    train_loss = []
    val_rmse = []

    for epoch in range(CONFIG["num_epochs"]):
        # 训练阶段
        model.train()
        epoch_loss = 0
        for x, y in train_loader:
            x = x.to(CONFIG["device"])
            y = y.to(CONFIG["device"])

            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)

            # 混合精度训练
            scaler = GradScaler()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()

        # 验证阶段
        current_rmse = evaluate_cnn(model, test_loader, target_scaler)
        val_rmse.append(current_rmse)
        avg_loss = epoch_loss / len(train_loader)
        train_loss.append(avg_loss)

        # 保存最佳模型
        if current_rmse < best_rmse:
            best_rmse = current_rmse
            torch.save(model.state_dict(), "best_cnn_model.pth")

        # 打印进度
        if (epoch + 1) % 5 == 0:
            print(f"Epoch [{epoch + 1}/{CONFIG['num_epochs']}] "
                  f"Train Loss: {avg_loss:.4f} | Val RMSE: {current_rmse:.2f}")

    # 绘制训练曲线
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_loss, label='Train Loss')
    plt.title("Training Loss")
    plt.subplot(1, 2, 2)
    plt.plot(val_rmse, label='Validation RMSE')
    plt.title("Validation RMSE")
    plt.tight_layout()
    plt.show()


def evaluate_cnn(model, test_loader, target_scaler):
    """增强的评估函数"""
    model.eval()
    preds = []
    trues = []

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(CONFIG["device"])
            batch_pred = model(x).cpu().numpy()

            # 反标准化
            inv_pred = target_scaler.inverse_transform(batch_pred)
            inv_true = target_scaler.inverse_transform(y.numpy())

            preds.extend(inv_pred.flatten())
            trues.extend(inv_true.flatten())

    # 计算指标
    rmse = np.sqrt(mean_squared_error(trues, preds))

    return rmse


# 主程序
if __name__ == "__main__":
    # 加载预处理数据
    train_loader, test_loader, processed_info = process_financial_data(DATA_CONFIG)
    target_scaler = processed_info["scaler"]

    # 初始化模型
    model = TimeSeriesCNN(
        input_channels=CONFIG["input_channels"],
        seq_length=CONFIG["seq_length"],
        pred_length=CONFIG["pred_length"]
    ).to(CONFIG["device"])

    print(f"Model Architecture:\n{model}")

    # 训练模型
    train_cnn(model, train_loader, test_loader, target_scaler)

    # 加载最佳模型
    model.load_state_dict(torch.load("best_cnn_model.pth"))

    # 最终评估
    final_rmse = evaluate_model_origin(model, test_loader, processed_info)


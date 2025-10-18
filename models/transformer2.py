import time

import torch
import torch.nn as nn
import math
import matplotlib.pyplot as plt

import dataprocess
import evaluate

# 配置参数（与数据处理流程保持一致）
CONFIG = {
    "seq_length": 60,  # 输入序列长度
    "pred_length": 1,  # 预测步长
    "num_features": 6,  # 输入特征维度
    "d_model": 64,  # 模型维度
    "nhead": 4,  # 注意力头数
    "num_encoder_layers": 2,  # 编码器层数
    "dim_feedforward": 128,  # 前馈网络维度
    "dropout": 0.3,  # Dropout概率
    "learning_rate": 0.0001,  # 降低学习率
    "num_epochs": 100,
    "batch_size": 64,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu")
}


class PositionalEncoding(nn.Module):
    """改进的位置编码（支持任意序列长度）"""

    def __init__(self, d_model, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=CONFIG["dropout"])

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class FinancialTransformer(nn.Module):
    """多特征时序预测Transformer"""

    def __init__(self):
        super().__init__()

        # 输入嵌入层
        self.input_embedding = nn.Linear(CONFIG["num_features"], CONFIG["d_model"])
        self.pos_encoder = PositionalEncoding(CONFIG["d_model"])

        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=CONFIG["d_model"],
            nhead=CONFIG["nhead"],
            dim_feedforward=CONFIG["dim_feedforward"],
            dropout=CONFIG["dropout"],
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, CONFIG["num_encoder_layers"])

        # 输出层
        self.decoder = nn.Sequential(
            nn.Linear(CONFIG["d_model"], 32),
            nn.ReLU(),
            nn.Linear(32, CONFIG["pred_length"])
        )

        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src):
        # 输入形状: (batch, seq_len, num_features)
        src = self.input_embedding(src) * math.sqrt(CONFIG["d_model"])
        src = self.pos_encoder(src)
        memory = self.transformer(src)
        output = self.decoder(memory[:, -1, :])  # 取最后时间步
        return output


def train_transformer(model, train_loader, test_loader, processed_info):
    """模型训练与验证"""
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["learning_rate"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)

    train_losses = []
    val_losses = []

    for epoch in range(CONFIG["num_epochs"]):
        model.train()
        epoch_loss = 0

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
            epoch_loss += loss.item()

        # 验证阶段
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for val_x, val_y in val_loader:
                val_x = val_x.to(CONFIG["device"])
                val_y = val_y.to(CONFIG["device"])
                preds = model(val_x)
                val_loss += criterion(preds, val_y).item()

        # 记录指标
        avg_loss = epoch_loss / len(train_loader)
        avg_val_loss = val_loss / len(test_loader)
        train_losses.append(avg_loss)
        val_losses.append(avg_val_loss)
        scheduler.step(avg_val_loss)

        # 打印训练进度
        if (epoch + 1) % 10 == 0:
            lr = optimizer.param_groups[0]['lr']
            print(f"Epoch [{epoch + 1}/{CONFIG['num_epochs']}] | "
                  f"Train Loss: {avg_loss:.4f} | Val Loss: {avg_val_loss:.4f} | LR: {lr:.2e}")

    # 绘制损失曲线
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Loss Curve')
    plt.xlabel('Epochs')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.show()

    return model
# 主程序
if __name__ == "__main__":
    # 加载预处理数据
    #train_loader, test_loader, processed_info = dataprocess.process_financial_data(dataprocess.DATA_CONFIG)
    train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset, processed_info = dataprocess.process_financial_data(
        CONFIG,    )
    # 初始化模型
    model = FinancialTransformer().to(CONFIG["device"])
    print(CONFIG)
    print(model)

    print("\n=== 维度验证 ===")
    sample_x, sample_y = next(iter(train_loader))
    print(f"输入数据形状: {sample_x.shape}")  # 应满足 (batch, seq_len, num_features)
    print(f"模型预期输入维度: {model.input_embedding.in_features}")

    # 训练模型
    s_time = time.time()
    train_transformer(model, train_loader, val_loader, processed_info)
    e_time = time.time()
    print(f"耗时:{e_time-s_time:.4f}")

    # 评估模型
    evaluate.evaluate_model(model, test_loader, processed_info)
    evaluate.evaluate_model_origin(model, test_loader, processed_info)
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from models import FEDformer

from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader


# 设置随机种子保证可重复性
torch.manual_seed(42)
np.random.seed(42)




# 4. 训练过程
def train(model, train_loader, val_loader, epochs=10, lr=0.001):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.1)

    best_val_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            if len(batch_y.shape) == 2:
                batch_y = batch_y.unsqueeze(-1)
                batch_y = batch_y.to(device)

            print("batch_x shape:", batch_x.shape)  # 应与 dec_inp 一致，如 [64, 96, 6]
            print("batch_y shape:", batch_y.shape)  # 应与 dec_inp 一致，如 [64, 96, 6]
            print("label_len:", model.configs.label_len)  # 应 ≤ batch_y 的第1维度（96）

            enc_inp = torch.zeros_like(batch_x).to(device)
            enc_inp[:, :model.configs.seq_len,:] = batch_x[:, :model.configs.seq_len,:]
            # 解码器输入：前label_len个时间步作为start token，其余用0填充
            dec_inp = torch.zeros_like(batch_y).to(device)
            dec_inp[:, :model.configs.label_len] = batch_y[:, :model.configs.label_len]

            optimizer.zero_grad()

            # 前向传播
            outputs = model(batch_x, enc_inp,batch_y, dec_inp)
            loss = criterion(outputs, batch_y)

            # 反向传播
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # 验证
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)

                dec_inp = torch.zeros_like(batch_y).to(device)
                dec_inp[:, :model.configs.label_len] = batch_y[:, :model.configs.label_len]

                outputs = model(batch_x, dec_inp)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)

        scheduler.step(val_loss)

        print(f'Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_fedformer_model.pth')

    print('Training completed.')


# 5. 测试和预测
def evaluate(model, test_loader, scaler):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(torch.load('best_fedformer_model.pth'))
    model = model.to(device)
    model.eval()

    criterion = nn.MSELoss()
    test_loss = 0.0
    predictions = []
    actuals = []

    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            dec_inp = torch.zeros_like(batch_y).to(device)
            dec_inp[:, :model.configs.label_len] = batch_y[:, :model.configs.label_len]

            outputs = model(batch_x, dec_inp)
            loss = criterion(outputs, batch_y)
            test_loss += loss.item()

            # 反归一化
            outputs = outputs.cpu().numpy()
            batch_y = batch_y.cpu().numpy()

            outputs = scaler.inverse_transform(outputs.reshape(-1, 1))
            batch_y = scaler.inverse_transform(batch_y.reshape(-1, 1))

            predictions.append(outputs)
            actuals.append(batch_y)

    test_loss /= len(test_loader)
    predictions = np.concatenate(predictions, axis=0)
    actuals = np.concatenate(actuals, axis=0)

    print(f'Test Loss: {test_loss:.4f}')

    # 可视化预测结果
    plt.figure(figsize=(12, 6))
    plt.plot(actuals, label='Actual Prices')
    plt.plot(predictions, label='Predicted Prices', alpha=0.7)
    plt.title('Price Prediction using FEDformer')
    plt.xlabel('Time Steps')
    plt.ylabel('Price')
    plt.legend()
    plt.show()

    return predictions, actuals



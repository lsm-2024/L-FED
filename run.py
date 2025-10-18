import np
import torch
import torch.nn as nn
import caculate_metrics
import models
from models import lstm
import dataprocess
from models.lstm import FinancialLSTM
from models import FEDformer
from models.FEDformer import  Fedformer

from models import Transformer
from models.transformer2 import TransModel


from config import CONFIG
'''
        self.enc_in = 6  # 输入特征维度
        self.dec_in = 6  # 解码器输入特征维度
        self.c_out = 1  # 输出特征维度
        self.seq_len = 64  # 输入序列长度
        self.label_len = 48  # 解码器输入中start token的长度
        self.pred_len = 1  # 预测序列长度
        self.d_model = 512  # 模型维度
        self.n_heads = 8  # 多头注意力头数
        self.e_layers = 2  # 编码器层数
        self.d_layers = 1  # 解码器层数
        self.d_ff = 2048  # 前馈网络维度
        self.moving_avg = 25  # 移动平均窗口大小
        self.dropout = 0.1  # dropout概率
        self.version = 'Fourier'  # FEDformer变体: 'Fourier'或'Wavelet'
        self.mode_select = 'random'  # 模式选择方式
        self.modes = 64  # 模式数量
        self.activation = 'gelu'  # 激活函数
        self.output_attention = False  # 是否输出注意力权重
        self.embed = 'timeF'  # 时间特征编码方式
        self.freq = 'h'  # 时间频率
'''


class Configs(object):
    ab = 0
    use_amp = 0
    train_epochs = 100
    modes = 32
    mode_select = 'random'
    # version = 'Fourier'
    version = 'Wavelets'
    moving_avg = [12, 24]
    L = 1
    base = 'legendre'
    cross_activation = 'tanh'
    seq_len = 16
    label_len = 16
    pred_len = 96
    output_attention = True
    enc_in = 6
    dec_in = 6
    d_model = 16
    embed = 'timeF'
    dropout = 0.05
    freq = 'h'
    factor = 1
    n_heads = 8
    d_ff = 16
    e_layers = 2
    d_layers = 1
    c_out = 7
    activation = 'gelu'
    wavelet = 0
    patience = 5
    learning_rate = 1e-3
    beta1 = 0.9
    beta2 = 0.999


def train_model(model, train_loader, val_loader, test_loader, scaler):
    """模型训练、验证与测试"""
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["learning_rate"])

    # 训练记录
    train_loss = []
    val_loss = []
    test_loss = []

    for epoch in range(CONFIG["num_epochs"]):
        # 训练阶段
        model.train()
        epoch_train_loss = 0
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(CONFIG["device"])
            batch_y = batch_y.to(CONFIG["device"])

            # 前向传播
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 梯度裁剪
            optimizer.step()

            epoch_train_loss += loss.item()

        # 验证阶段
        model.eval()
        epoch_val_loss = 0
        with torch.no_grad():
            for val_x, val_y in val_loader:
                val_x = val_x.to(CONFIG["device"])
                val_y = val_y.to(CONFIG["device"])
                preds = model(val_x)
                epoch_val_loss += criterion(preds, val_y).item()

        # 测试阶段（每个epoch都查看测试集表现，但不参与参数更新）
        epoch_test_loss = 0
        with torch.no_grad():
            for test_x, test_y in test_loader:
                test_x = test_x.to(CONFIG["device"])
                test_y = test_y.to(CONFIG["device"])
                preds = model(test_x)
                epoch_test_loss += criterion(preds, test_y).item()

        # 记录损失
        avg_train_loss = epoch_train_loss / len(train_loader)
        avg_val_loss = epoch_val_loss / len(val_loader)
        avg_test_loss = epoch_test_loss / len(test_loader)

        train_loss.append(avg_train_loss)
        val_loss.append(avg_val_loss)
        test_loss.append(avg_test_loss)

        # 打印进度
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch + 1}/{CONFIG['num_epochs']}] | "
                  f"Train Loss: {avg_train_loss:.4f} | "
                  f"Val Loss: {avg_val_loss:.4f} | "
                  f"Test Loss: {avg_test_loss:.4f}")

    # 绘制损失曲线
    #plt.figure(figsize=(10, 5))
    #plt.plot(train_loss, label="Training Loss")
    #plt.plot(val_loss, label="Validation Loss")
    #plt.plot(test_loss, label="Test Loss")
    #plt.title("Loss Evolution")
    #plt.xlabel("Epochs")
    #plt.ylabel("MSE Loss")
    #plt.legend()
    #plt.show()

    return model




# 主程序 --------------------------------------------------
if __name__ == "__main__":
    # 加载预处理数据，获取三个数据集的加载器
    # 注意：需要确保dataprocess.process_financial_data能接受分割比例参数
    train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset, processed_info = dataprocess.process_financial_data(
        CONFIG,

    )
    scaler = processed_info["scaler"]
    feature_names = processed_info["feature_names"]

    # 打印数据形状
    sample_x, sample_y = next(iter(train_loader))
    print(f"输入数据形状: {sample_x.shape}")  # (batch, seq_len, num_features)
    print(f"目标数据形状: {sample_y.shape}")  # (batch, pred_length)
    print(f"训练集批次: {len(train_loader)}, 验证集批次: {len(val_loader)}, 测试集批次: {len(test_loader)}")

    # 初始化模型
    model_type = CONFIG["model_type"]
    torch.cuda.empty_cache()
    match model_type:
        case "1":
            model = FinancialLSTM(
                input_size=CONFIG["input_size"],
                hidden_size=CONFIG["hidden_size"],
                num_layers=CONFIG["num_layers"],
                output_size=CONFIG["output_size"],
                dropout=CONFIG["dropout"]
            ).to(CONFIG["device"])
            trained_model = train_model(model, train_loader, val_loader, test_loader, scaler)
        case "2":
            configs = Configs
            # 初始化模型
            model = Fedformer(configs)
            trained_model = model.train()
            model.test()

        case "3":
            configs = Configs()
            model = TransModel(configs).to(CONFIG["device"])
            trained_model = train_model(model, train_loader, val_loader, test_loader, scaler)



    # 打印模型结构和配置
    print(CONFIG)
    print(model)

    # 训练模型
    #s_time = time.time()
    #trained_model = train_model(model, train_loader, val_loader, test_loader, scaler)
    #e_time = time.time()
    #print(f"训练耗时: {e_time - s_time:.4f}秒")

    # 在测试集上计算评估指标
    caculate_metrics.calculate_metrics(trained_model, test_loader, processed_info["scaler"], processed_info["feature_names"])

    # 示例预测
    sample_x, _ = next(iter(test_loader))
    sample_x = sample_x.to(CONFIG["device"])
    prediction = trained_model(sample_x).cpu().detach().numpy()

    # 反标准化示例
    dummy_matrix = np.zeros((1, len(feature_names)))
    dummy_matrix[:, -1] = prediction[0][0]
    original_pred = scaler.inverse_transform(dummy_matrix)[0, -1]
    print(f"\n示例预测值: {original_pred:.4f}")
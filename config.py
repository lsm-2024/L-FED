import torch
# 配置参数

CONFIG = {
    "data_path": "data/ETH-USDT.csv",
    "features": ["open", "high", "low", "close", "volume"],  # 使用的特征列
    "target": "close",  # 要预测的目标列
    "seq_length": 8,  # 输入序列长度（时间窗口）
    "pred_length": 1,  # 预测步长
    "scaler_range": (-1, 1),  # 归一化范围
    "input_size": 6,  # 输入特征维度
    "hidden_size": 256,  # LSTM隐藏层维度
    "num_layers": 1,  # LSTM堆叠层数
    "output_size": 1,  # 输出维度
    "dropout": 0.3,  # Dropout概率
    "learning_rate": 0.001,
    "num_epochs": 100,
    "batch_size": 64,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "train_ratio": 0.6,  # 训练集比例
    "val_ratio": 0.2,  # 验证集比例
    "test_ratio": 0.2,  # 测试集比例
    "model_type":"2"  #1:lstm , 2: fedformer, 3: transformer

}


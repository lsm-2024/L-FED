import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler

from config import CONFIG


class FinancialDataset(Dataset):
    """金融时间序列数据集"""

    def __init__(self, data, seq_length, pred_length):
        self.data = data
        self.seq_length = seq_length
        self.pred_length = pred_length

    def __len__(self):
        return len(self.data) - self.seq_length - self.pred_length + 1

    def __getitem__(self, idx):
        # 输入特征序列：包含所有特征
        x = self.data[idx:idx + self.seq_length, :]

        # 预测目标序列：仅目标列
        y = self.data[idx + self.seq_length:idx + self.seq_length + self.pred_length, -1]

        return torch.FloatTensor(x), torch.FloatTensor(y)


def process_financial_data(config,train_ratio=CONFIG["train_ratio"],
        val_ratio=CONFIG["val_ratio"],
        test_ratio=CONFIG["test_ratio"]):
    """完整数据处理流程"""
    # 1. 读取数据并解析时间
    df = pd.read_csv(config["data_path"])
    df['candle_begin_time'] = pd.to_datetime(df['candle_begin_time'])
    df.set_index('candle_begin_time', inplace=True)

    # 2. 数据清洗
    # print(f"原始数据缺失值统计：\n{df.isnull().sum()}")
    # df[config["features"]] = df[config["features"]].ffill()  # 前向填充

    # 3. 特征工程
    # df['price_change'] = df['close'].pct_change()  # 价格变化率
    # df['vol_change'] = df['volume'].pct_change()  # 成交量变化率
    # config["features"] += ['price_change', 'vol_change']

    # 4. 提取数据
    feature_data = df[config["features"]].values
    target_data = df[[config["target"]]].values

    # 5. 数据标准化
    scaler = MinMaxScaler(feature_range=config["scaler_range"])
    scaled_features = scaler.fit_transform(feature_data)
    scaled_target = scaler.fit_transform(target_data)

    # 6. 合并特征和目标
    full_data = np.concatenate([scaled_features, scaled_target], axis=1)
    # 7. 创建数据集
    dataset = FinancialDataset(full_data, config["seq_length"], config["pred_length"])

    # 8. 时序划分数据集
    # 计算分割索引
    total_size = len(dataset)
    train_idx = int(total_size * train_ratio)
    val_idx = int(total_size * (train_ratio + val_ratio))

    # 划分数据集
    train_dataset = torch.utils.data.Subset(dataset, range(train_idx))
    val_dataset = torch.utils.data.Subset(dataset, range(train_idx, val_idx))
    test_dataset = torch.utils.data.Subset(dataset, range(val_idx, total_size))


    # 9. 创建DataLoader
    train_loader = DataLoader(train_dataset,
                              batch_size=config["batch_size"],
                              shuffle=False,
                              drop_last=True)


    val_loader = DataLoader(val_dataset,
                             batch_size=config["batch_size"],
                             shuffle=False)
    test_loader = DataLoader(test_dataset,
                             batch_size=config["batch_size"],
                             shuffle=False)

    # 记录特征名称和标准化器
    processed_info = {
        "feature_names": config["features"] + [config["target"]],  # 所有列名
        "target_index": len(config["features"]),  # 目标列位置（最后一列）
        "scaler": scaler  # 使用的标准化器
    }

    # 添加维度验证
    print(f"特征矩阵形状: {feature_data.shape}")
    print(f"目标矩阵形状: {target_data.shape}")

    # 检查合并后的数据维度
    print(f"合并后数据形状: {full_data.shape}")
    # fig1 = plt.figure(figsize=(12, 6))
    # plt.plot(dataset)
    # plt.title('close price')
    # plt.xlabel('time')
    # plt.ylabel('price')
    # plt.show()

    # 验证标准化
    # print("特征均值:", scaled_features.mean_)
    # print("目标最大值:", scaled_target.data_max_)

    return train_loader, val_loader,test_loader, train_dataset,val_dataset,test_dataset,processed_info








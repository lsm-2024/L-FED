import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error

from config import CONFIG
def calculate_metrics(model, test_loader, scaler, feature_names):
    """计算测试集上的评估指标"""
    model.eval()
    all_preds = []
    all_actuals = []

    with torch.no_grad():
        for test_x, test_y in test_loader:
            test_x = test_x.to(CONFIG["device"])
            preds = model(test_x).cpu().numpy()
            actuals = test_y.numpy()

            # 反标准化预测值
            for pred in preds:
                dummy_matrix = np.zeros((1, len(feature_names)))
                dummy_matrix[:, -1] = pred[0]
                original_pred = scaler.inverse_transform(dummy_matrix)[0, -1]
                all_preds.append(original_pred)

            # 反标准化实际值
            for actual in actuals:
                dummy_matrix = np.zeros((1, len(feature_names)))
                dummy_matrix[:, -1] = actual[0]
                original_actual = scaler.inverse_transform(dummy_matrix)[0, -1]
                all_actuals.append(original_actual)

    # 转换为numpy数组
    all_preds = np.array(all_preds)
    all_actuals = np.array(all_actuals)

    # 计算指标
    rmse = np.sqrt(mean_squared_error(all_actuals, all_preds))
    r2 = r2_score(all_actuals, all_preds)
    # 避免除以零，添加一个小的epsilon
    mape = mean_absolute_percentage_error(all_actuals, all_preds) * 100

    print("\n测试集评估指标:")
    print(f"RMSE: {rmse:.4f}")
    print(f"R2 Score: {r2:.4f}")
    print(f"MAPE: {mape:.2f}%")

    return rmse, r2, mape
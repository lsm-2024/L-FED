import torch
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

from config import CONFIG

def evaluate_model(model, test_loader, processed_info):
    model.eval()
    standardized_preds = []
    standardized_trues = []

    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(CONFIG["device"])
            batch_pred = model(batch_x).cpu().numpy()

            # 标准化空间数据
            standardized_preds.append(batch_pred)
            standardized_trues.append(batch_y.numpy())

    # 合并结果
    standardized_preds = np.concatenate(standardized_preds)
    standardized_trues = np.concatenate(standardized_trues)

    # 计算标准化空间指标
    std_mse = mean_squared_error(standardized_trues, standardized_preds)
    std_rmse = np.sqrt(std_mse)

    print("\n=== 标准化空间评估 ===")
    print(f"MSE: {std_mse:.6f}")
    print(f"RMSE: {std_rmse:.6f}")

    # 可视化对比
    plt.figure(figsize=(12, 6))
    plt.plot(standardized_trues[:200], label='True (Standardized)', alpha=0.7)
    plt.plot(standardized_preds[:200], label='Pred (Standardized)', linestyle='--')

    plt.title("Prediction Comparison")
    plt.xlabel("Time Step")
    plt.ylabel("Value")
    plt.legend()
    plt.show()

    return {
        'standardized': (std_mse, std_rmse),
    }

def evaluate_model_origin(model, test_loader, processed_info):
    model.eval()
    original_preds = []
    original_trues = []
    scaler = processed_info["scaler"]
    feature_names = processed_info["feature_names"]
    target_index = processed_info["target_index"]

    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(CONFIG["device"])
            batch_pred = model(batch_x).cpu().numpy()

            # 反标准化处理（如果提供scaler）
            #     # 创建反标准化矩阵
            dummy_matrix = np.zeros((len(batch_pred), len(feature_names)))
            dummy_matrix[:, target_index] = batch_pred.flatten()
            inv_pred = scaler.inverse_transform(dummy_matrix)[:, target_index]

            dummy_matrix[:, target_index] = batch_y.numpy().flatten()
            inv_true = scaler.inverse_transform(dummy_matrix)[:, target_index]

            original_preds.append(inv_pred)
            original_trues.append(inv_true)

    # dummy_test = np.zeros((len(batch_pred), len(feature_names)))
    # dummy_test[0, target_index] = 0.5
    # print("反标准化测试:", scaler.inverse_transform(dummy_test))

    # 计算原始空间指标
    original_preds = np.concatenate(original_preds)
    original_trues = np.concatenate(original_trues)
    orig_mse = mean_squared_error(original_trues, original_preds)
    orig_rmse = np.sqrt(orig_mse)

    print("\n=== 原始空间评估 ===")
    print(f"MSE: {orig_mse:.2f}")
    print(f"RMSE: {orig_rmse:.2f}")

    calculate_prediction_indicators(original_preds)

    # 可视化对比
    plt.figure(figsize=(12, 6))
    plt.plot(original_trues, label='True (Original)', alpha=0.7)
    plt.plot(original_preds, label='Pred (Original)', linestyle='--')

    plt.title("Prediction Comparison")
    plt.xlabel("Time Step")
    plt.ylabel("Value")
    plt.legend()
    plt.show()

    return {
        'original': (orig_mse, orig_rmse)
    }

def calculate_prediction_indicators(pred_series, window=14):
    """
    预测结果技术指标计算器

    """
    df = pd.DataFrame({'pred_close': pred_series})

    # 1. SMA
    df['SMA'] = df['pred_close'].rolling(window=window).mean().shift(1)
    print("SMA:", df['SMA'])

    # 2. EMA
    df['EMA'] = df['pred_close'].ewm(span=window, adjust=False).mean().shift(1)
    print("EMA:", df['EMA'])

    # 3. Bollinger Bands
    if len(df) >= window:
        rolling_mean = df['pred_close'].rolling(window=window).mean().shift(1)
        rolling_std = df['pred_close'].rolling(window=window).std().shift(1)
        df['BB_upper'] = rolling_mean + 2 * rolling_std
        df['BB_middle'] = rolling_mean
        df['BB_lower'] = rolling_mean - 2 * rolling_std
    print("BB:", df['EMA'])

    # 4. RSI
    delta = df['pred_close'].diff().shift(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    df['RSI'] = 100 - (100 / (1 + rs))
    print("RSI:", df['RSI'])

    return df
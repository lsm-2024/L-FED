import torch
import torch.nn as nn
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import time

import dataprocess
import evaluate

from config import CONFIG



class FinancialLSTM(nn.Module):
    """多特征金融时序预测LSTM模型"""

    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # 网络结构
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # 初始化隐藏状态
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(CONFIG["device"])
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(CONFIG["device"])

        # LSTM前向传播
        out, _ = self.lstm(x, (h0, c0))

        # 取最后一个时间步输出
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        return out



import torch
import torch.nn as nn

class CMAPSS_LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout_prob):
        super(CMAPSS_LSTM, self).__init__()
        
        # 核心 LSTM 层。论文通过网格搜索测试了 2, 4, 6, 8 层 
        self.lstm = nn.LSTM(
            input_size=input_size, 
            hidden_size=hidden_size, 
            num_layers=num_layers, 
            batch_first=True, 
            dropout=dropout_prob if num_layers > 1 else 0
        )
        
        # 全连接层与回归层 (参考图1结构) 
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.dropout = nn.Dropout(dropout_prob)
        self.relu = nn.ReLU()
        # 最后的回归层输出一个节点，即预测的 RUL 
        self.fc2 = nn.Linear(hidden_size // 2, 1)

    def forward(self, x):
        # x 形状: (batch_size, sequence_length, input_size)
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # 提取序列最后一个时间步的输出作为特征
        last_time_step_out = lstm_out[:, -1, :]
        
        # 经过全连接层
        out = self.fc1(last_time_step_out)
        out = self.relu(out)
        out = self.dropout(out)
        
        # 回归输出
        out = self.fc2(out)
        return out

def calculate_metrics(y_pred, y_true):
    """
    计算 RMSE 和论文中使用的不对称 Score 
    y_pred, y_true 应为 1D Tensor
    """
    # 1. 计算 RMSE 
    mse = nn.functional.mse_loss(y_pred, y_true)
    rmse = torch.sqrt(mse).item()
    
    # 2. 计算 Score Function (Eq 15 & 16) 
    h = y_pred - y_true
    
    # 当 h < 0 (提前预测): penalty = exp(-h / 13) - 1 
    # 当 h >= 0 (滞后预测): penalty = exp(h / 10) - 1 
    score = torch.where(
        h < 0, 
        torch.exp(-h / 13.0) - 1, 
        torch.exp(h / 10.0) - 1
    )
    total_score = torch.sum(score).item()
    
    return rmse, total_score
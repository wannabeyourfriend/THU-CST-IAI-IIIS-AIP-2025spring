import torch
import torch.nn as nn
import torch.nn.functional as F

class TextCNN(nn.Module):
    def __init__(self, input_dim, num_filters=100, filter_sizes=(3, 4, 5), num_classes=2, dropout=0.5):
        super(TextCNN, self).__init__()
        
        self.convs = nn.ModuleList([
            nn.Conv2d(1, num_filters, (k, input_dim)) 
            for k in filter_sizes
        ])
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(len(filter_sizes) * num_filters, num_classes)
        
    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_dim)
        # 添加通道维度
        x = x.unsqueeze(1)  # (batch_size, 1, sequence_length, input_dim)
        
        # 应用卷积和池化
        x = [
            F.relu(conv(x)).squeeze(3) for conv in self.convs
        ]  # [(batch_size, num_filters, seq_len-k+1) for k in filter_sizes]
        
        # 最大池化
        x = [
            F.max_pool1d(i, i.size(2)).squeeze(2) for i in x
        ]  # [(batch_size, num_filters) for _ in filter_sizes]
        
        # 拼接不同卷积核的结果
        x = torch.cat(x, 1)  # (batch_size, num_filters * len(filter_sizes))
        
        # Dropout和分类
        x = self.dropout(x)
        x = self.fc(x)
        return x
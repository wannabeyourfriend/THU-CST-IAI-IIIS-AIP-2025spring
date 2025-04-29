import torch
import torch.nn as nn

class RNNClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, num_layers=3, rnn_type='LSTM',
                 bidirectional=True, num_classes=2, dropout=0.5):
        super(RNNClassifier, self).__init__()
        
        # RNN类型选择
        if rnn_type == 'LSTM':
            self.rnn = nn.LSTM(input_dim, hidden_dim, num_layers,
                              batch_first=True, bidirectional=bidirectional,
                              dropout=dropout if num_layers > 1 else 0)
        else:  # GRU
            self.rnn = nn.GRU(input_dim, hidden_dim, num_layers,
                             batch_first=True, bidirectional=bidirectional,
                             dropout=dropout if num_layers > 1 else 0)
        
        # 输出层
        self.fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, num_classes)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_dim)
        
        # RNN前向传播
        output, _ = self.rnn(x)  # output: (batch_size, seq_len, hidden_dim*2)
        
        # 取最后一个时间步的隐状态
        if isinstance(self.rnn, nn.LSTM):
            last_hidden = output[:, -1, :]
        else:
            last_hidden = output[:, -1, :]
        
        # Dropout和分类
        out = self.dropout(last_hidden)
        out = self.fc(out)
        return out
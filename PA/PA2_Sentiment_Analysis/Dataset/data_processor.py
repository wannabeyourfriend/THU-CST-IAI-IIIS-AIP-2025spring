import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import gensim

class SentimentDataset(Dataset):
    def __init__(self, file_path, word2vec_model):
        self.data = []
        self.labels = []
        self.word2vec = word2vec_model
        
        # 读取数据
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                label, text = line.strip().split('\t')
                self.labels.append(int(label))
                self.data.append(text.split())
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        words = self.data[idx]
        label = self.labels[idx]
        
        # 将词转换为词向量
        word_vectors = []
        for word in words:
            try:
                vector = self.word2vec[word]
                word_vectors.append(vector)
            except KeyError:
                # 对于未知词，使用零向量
                vector = np.zeros(self.word2vec.vector_size)
                word_vectors.append(vector)
        
        # 将词向量列表转换为tensor
        word_vectors = torch.tensor(word_vectors, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.long)
        
        return word_vectors, label

def load_word2vec(model_path):
    """加载预训练的word2vec模型"""
    return gensim.models.KeyedVectors.load_word2vec_format(model_path, binary=True)

def create_data_loader(file_path, word2vec_model, batch_size=32, shuffle=True):
    """创建数据加载器"""
    dataset = SentimentDataset(file_path, word2vec_model)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                     collate_fn=collate_fn)

def collate_fn(batch):
    """处理变长序列的批处理函数"""
    # 分离特征和标签
    sequences, labels = zip(*batch)
    
    # 获取每个序列的长度
    lengths = [seq.size(0) for seq in sequences]
    max_len = max(lengths)
    
    # 填充序列到最大长度
    padded_seqs = torch.zeros(len(sequences), max_len, sequences[0].size(1))
    for i, seq in enumerate(sequences):
        end = lengths[i]
        padded_seqs[i, :end] = seq
    
    # 转换为tensor
    labels = torch.stack(labels)
    
    return padded_seqs, labels
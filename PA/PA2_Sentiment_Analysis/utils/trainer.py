import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import wandb
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

class Trainer:
    def __init__(self, model, device, config):
        self.model = model.to(device)
        self.device = device
        self.config = config
        
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
        
        # 初始化wandb
        wandb.init(
            project="sentiment-analysis",
            config=config
        )
    
    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0
        predictions = []
        true_labels = []
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
            pred = output.argmax(dim=1, keepdim=True)
            predictions.extend(pred.cpu().numpy())
            true_labels.extend(target.cpu().numpy())
        
        # 计算指标
        accuracy = accuracy_score(true_labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_labels, predictions, average='binary'
        )
        
        metrics = {
            'train_loss': total_loss / len(train_loader),
            'train_accuracy': accuracy,
            'train_precision': precision,
            'train_recall': recall,
            'train_f1': f1
        }
        
        return metrics
    
    def evaluate(self, val_loader):
        self.model.eval()
        total_loss = 0
        predictions = []
        true_labels = []
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                
                loss = self.criterion(output, target)
                total_loss += loss.item()
                
                pred = output.argmax(dim=1, keepdim=True)
                predictions.extend(pred.cpu().numpy())
                true_labels.extend(target.cpu().numpy())
        
        # 计算指标
        accuracy = accuracy_score(true_labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_labels, predictions, average='binary'
        )
        
        metrics = {
            'val_loss': total_loss / len(val_loader),
            'val_accuracy': accuracy,
            'val_precision': precision,
            'val_recall': recall,
            'val_f1': f1
        }
        
        return metrics
    
    def train(self, train_loader, val_loader, num_epochs):
        best_val_f1 = 0
        
        for epoch in range(num_epochs):
            # 训练
            train_metrics = self.train_epoch(train_loader)
            
            # 验证
            val_metrics = self.evaluate(val_loader)
            
            # 记录到wandb
            wandb.log({**train_metrics, **val_metrics})
            
            # 保存最佳模型
            if val_metrics['val_f1'] > best_val_f1:
                best_val_f1 = val_metrics['val_f1']
                torch.save(self.model.state_dict(), 'best_model.pth')
            
            print(f"Epoch {epoch+1}/{num_epochs}:")
            print(f"Train Loss: {train_metrics['train_loss']:.4f}, "
                  f"Train F1: {train_metrics['train_f1']:.4f}")
            print(f"Val Loss: {val_metrics['val_loss']:.4f}, "
                  f"Val F1: {val_metrics['val_f1']:.4f}\n")
        
        wandb.finish()
    
    def test(self, test_loader):
        self.model.eval()
        predictions = []
        true_labels = []
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                pred = output.argmax(dim=1, keepdim=True)
                predictions.extend(pred.cpu().numpy())
                true_labels.extend(target.cpu().numpy())
        
        # 计算测试集指标
        accuracy = accuracy_score(true_labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_labels, predictions, average='binary'
        )
        
        metrics = {
            'test_accuracy': accuracy,
            'test_precision': precision,
            'test_recall': recall,
            'test_f1': f1
        }
        
        return metrics
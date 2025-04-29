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
    
    # 在train方法中添加wandb.log调用
    def train(self, train_loader, val_loader, num_epochs):
        best_val_f1 = 0
        
        for epoch in range(num_epochs):
            # 训练阶段
            self.model.train()
            train_loss = 0.0
            train_preds = []
            train_labels = []
            
            for batch_idx, (inputs, labels) in enumerate(train_loader):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                # 前向传播
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                
                # 反向传播和优化
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                # 累积损失
                train_loss += loss.item()
                
                # 收集预测和标签
                _, predicted = torch.max(outputs.data, 1)
                train_preds.extend(predicted.cpu().numpy())
                train_labels.extend(labels.cpu().numpy())
                
                # 每N个batch记录一次训练指标
                if batch_idx % 10 == 0:
                    wandb.log({
                        'batch': epoch * len(train_loader) + batch_idx,
                        'batch_loss': loss.item()
                    })
            
            # 计算训练指标
            train_metrics = self.compute_metrics(train_preds, train_labels)
            train_loss /= len(train_loader)
            
            # 验证阶段
            val_metrics, val_loss = self.validate(val_loader)
            
            # 记录每个epoch的指标
            wandb.log({
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'train_accuracy': train_metrics['accuracy'],
                'train_precision': train_metrics['precision'],
                'train_recall': train_metrics['recall'],
                'train_f1': train_metrics['f1'],
                'val_loss': val_loss,
                'val_accuracy': val_metrics['accuracy'],
                'val_precision': val_metrics['precision'],
                'val_recall': val_metrics['recall'],
                'val_f1': val_metrics['f1']
            })
            
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
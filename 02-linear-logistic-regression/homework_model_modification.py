import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from utils import (
    make_regression_data, mse, log_epoch, RegressionDataset,
    make_classification_data, accuracy, ClassificationDataset,
    precision, recall, f1, roc_auc, plot_confusion_matrix
)
import matplotlib.pyplot as plt

# 1.1 Модифицируйте существующую линейную регрессию:
# - Добавьте L1 и L2 регуляризацию
# - Добавьте early stopping
class LinearRegression(nn.Module):
    def __init__(self, in_features, l1_lambda=0.01, l2_lambda=0.01):
        super().__init__()
        self.linear = nn.Linear(in_features, 1)
        self.l1_lambda = l1_lambda
        self.l2_lambda = l2_lambda

    def forward(self, x):
        return self.linear(x)
    
    def regularization_loss(self):
        l1_loss = sum(p.abs().sum() for p in self.parameters())
        l2_loss = sum(p.pow(2).sum() for p in self.parameters())
        return self.l1_lambda * l1_loss + self.l2_lambda * l2_loss

def train_linear_regression():
    X, y = make_regression_data(n=200)
    dataset = RegressionDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    model = LinearRegression(in_features=1)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    
    # обучение с early stopping
    best_loss = float('inf')
    patience = 5
    patience_counter = 0
    train_losses = []
    
    for epoch in range(1, 101):
        total_loss = 0
        for i, (batch_X, batch_y) in enumerate(dataloader):
            optimizer.zero_grad()
            y_pred = model(batch_X)
            loss = criterion(y_pred, batch_y) + model.regularization_loss()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        train_losses.append(avg_loss)
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'models/linreg.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'Early stopping at epoch {epoch}')
                break
        
        if epoch % 10 == 0:
            log_epoch(epoch, avg_loss)
    
    # сохранение графиков
    plt.plot(train_losses)
    plt.title('Linear Regression Training Loss')
    plt.savefig('plots/linreg_loss.png')
    plt.close()

# 1.2 Модифицируйте существующую логистическую регрессию:
# - Добавьте поддержку многоклассовой классификации
# - Реализуйте метрики: precision, recall, F1-score, ROC-AUC
# - Добавьте визуализацию confusion matrix
class LogisticRegression(nn.Module):
    def __init__(self, in_features, num_classes=1):
        super().__init__()
        self.num_classes = num_classes
        if num_classes == 1:
            self.linear = nn.Linear(in_features, 1)
        else:
            self.linear = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.linear(x)

def train_logistic_regression():
    X, y = make_classification_data(n=200, n_classes=3)
    num_classes = len(torch.unique(y))
    
    dataset = ClassificationDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    model = LogisticRegression(in_features=2, num_classes=num_classes)
    criterion = nn.CrossEntropyLoss() if num_classes > 1 else nn.BCEWithLogitsLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    
    train_losses = []
    train_accuracies = []
    all_preds = []
    all_targets = []

    for epoch in range(1, 101):
        total_loss = 0
        total_acc = 0
        
        for i, (batch_X, batch_y) in enumerate(dataloader):
            optimizer.zero_grad()
            logits = model(batch_X)
            
            if num_classes == 1:
                loss = criterion(logits, batch_y.float())
            else:
                loss = criterion(logits, batch_y.squeeze().long())
            
            loss.backward()
            optimizer.step()
            
            with torch.no_grad():
                if num_classes == 1:
                    y_pred = torch.sigmoid(logits)
                    acc = accuracy(y_pred, batch_y)
                else:
                    y_pred = torch.softmax(logits, dim=1)
                    acc = accuracy(y_pred, batch_y, num_classes)
                
                all_preds.extend(y_pred.argmax(dim=1).cpu().numpy() if num_classes > 1 
                               else (y_pred > 0.5).float().cpu().numpy())
                all_targets.extend(batch_y.cpu().numpy())
                total_loss += loss.item()
                total_acc += acc
        
        avg_loss = total_loss / len(dataloader)
        avg_acc = total_acc / len(dataloader)
        train_losses.append(avg_loss)
        train_accuracies.append(avg_acc)
        
        if epoch % 10 == 0:
            log_epoch(epoch, avg_loss, acc=avg_acc)
    
    # вычисление метрик
    print(f"Precision: {precision(all_targets, all_preds, num_classes)}")
    print(f"Recall: {recall(all_targets, all_preds, num_classes)}")
    print(f"F1: {f1(all_targets, all_preds, num_classes)}")
    
    # сохранение модели и графиков
    torch.save(model.state_dict(), 'models/logreg.pth')
    
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses)
    plt.title('Training Loss')
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies)
    plt.title('Training Accuracy')
    
    plt.savefig('plots/logreg_training.png')
    plt.close()
    
    plot_confusion_matrix(all_targets, all_preds, save_path='plots/confusion_matrix.png')

if __name__ == '__main__':
    train_linear_regression()
    train_logistic_regression()
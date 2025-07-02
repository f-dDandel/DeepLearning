import torch
import numpy as np
from torch.utils.data import Dataset
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, mean_absolute_error, r2_score

class RegressionDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class ClassificationDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def make_regression_data(n=100, noise=0.1, source='random'):
    if source == 'random':
        X = torch.rand(n, 1)
        w, b = 2.0, -1.0
        y = w * X + b + noise * torch.randn(n, 1)
        return X, y
    elif source == 'diabetes':
        from sklearn.datasets import load_diabetes
        data = load_diabetes()
        X = torch.tensor(data['data'], dtype=torch.float32)
        y = torch.tensor(data['target'], dtype=torch.float32).unsqueeze(1)
        return X, y
    else:
        raise ValueError('Unknown source')

def make_classification_data(n=100, source='random', n_classes=2):
    if source == 'random':
        X = torch.rand(n, 2)
        
        if n_classes == 2:
            w = torch.tensor([2.0, -3.0])
            b = 0.5
            logits = X @ w + b
            y = (logits > 0).long().unsqueeze(1)
        else:
            centers = torch.rand(n_classes, 2) * 4 - 2
            y = torch.randint(0, n_classes, (n, 1))
            X = centers[y.squeeze()] + torch.randn(n, 2) * 0.3
            
        return X, y
    
    elif source == 'breast_cancer':
        from sklearn.datasets import load_breast_cancer
        data = load_breast_cancer()
        X = torch.tensor(data['data'], dtype=torch.float32)
        y = torch.tensor(data['target'], dtype=torch.float32).unsqueeze(1)
        return X, y
    
    else:
        raise ValueError('Unknown source')

def mse(y_pred, y_true):
    return ((y_pred - y_true) ** 2).mean().item()

def mae(y_true, y_pred):
    return mean_absolute_error(y_true, y_pred)

def r2(y_true, y_pred):
    return r2_score(y_true, y_pred)

def accuracy(y_pred, y_true, num_classes=None):
    if num_classes == 1 or len(y_pred.shape) == 1 or y_pred.shape[1] == 1:
        # Бинарная классификация
        probs = torch.sigmoid(y_pred).squeeze()
        y_pred_labels = (probs > 0.5).float()
    else:
        # Многоклассовая
        _, y_pred_labels = torch.max(y_pred, 1)

    correct = (y_pred_labels == y_true).float().sum().item()
    total = y_true.shape[0]
    return correct / total

def precision(y_true, y_pred, num_classes=2):
    y_true_np = y_true.numpy() if torch.is_tensor(y_true) else y_true
    y_pred_np = y_pred.numpy() if torch.is_tensor(y_pred) else y_pred
    if num_classes == 2:
        return precision_score(y_true_np, y_pred_np)
    return precision_score(y_true_np, y_pred_np, average='macro')

def recall(y_true, y_pred, num_classes=2):
    y_true_np = y_true.numpy() if torch.is_tensor(y_true) else y_true
    y_pred_np = y_pred.numpy() if torch.is_tensor(y_pred) else y_pred
    if num_classes == 2:
        return recall_score(y_true_np, y_pred_np)
    return recall_score(y_true_np, y_pred_np, average='macro')

def f1(y_true, y_pred, num_classes=2):
    y_true_np = y_true.numpy() if torch.is_tensor(y_true) else y_true
    y_pred_np = y_pred.numpy() if torch.is_tensor(y_pred) else y_pred
    if num_classes == 2:
        return f1_score(y_true_np, y_pred_np)
    return f1_score(y_true_np, y_pred_np, average='macro')

def roc_auc(y_true, y_probs, num_classes=2):
    y_true_np = y_true.numpy() if torch.is_tensor(y_true) else y_true
    y_probs_np = y_probs.numpy() if torch.is_tensor(y_probs) else y_probs
    if num_classes == 2:
        return roc_auc_score(y_true_np, y_probs_np)
    return roc_auc_score(y_true_np, y_probs_np, multi_class='ovo', average='macro')

def plot_confusion_matrix(y_true, y_pred, save_path=None):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title('Confusion Matrix')
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def log_epoch(epoch, loss, **metrics):
    msg = f"Epoch {epoch}: loss={loss:.4f}"
    for k, v in metrics.items():
        msg += f", {k}={v:.4f}"
    print(msg)
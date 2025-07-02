import torch
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler, LabelEncoder
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from homework_model_modification import LinearRegression, LogisticRegression
from utils import mse, r2, mae, log_epoch, accuracy, precision, recall, f1, plot_confusion_matrix

# 2.1 Создайте кастомный класс датасета для работы с CSV файлами:
# - Загрузка данных из файла
# - Предобработка (нормализация, кодирование категорий)
# - Поддержка различных форматов данных (категориальные, числовые, бинарные и т.д.)

class CustomCSVDataset(Dataset):
    def __init__(self, csv_path, target_column, normalize_numeric=True, encode_categorical=True):
        """
        Args:
            csv_path: Путь к CSV файлу
            target_column: Название целевой колонки
            normalize_numeric: Нормализовать числовые признаки
            encode_categorical: Закодировать категориальные признаки
        """
        self.data = pd.read_csv(csv_path)
        self.target_column = target_column
        self.normalize_numeric = normalize_numeric
        self.encode_categorical = encode_categorical
        
        # предобработка данных
        self._preprocess_data()
    
    def _preprocess_data(self):
        # отделяем целевую переменную
        self.y = self.data[self.target_column]
        self.X = self.data.drop(columns=[self.target_column])

        # удаление выбросов для числовых признаков
        numeric_cols = self.X.select_dtypes(include=['int64', 'float64']).columns
        if len(numeric_cols) > 0:
            #  IQR для каждого числового признака
            for col in numeric_cols:
                Q1 = self.X[col].quantile(0.25)
                Q3 = self.X[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                self.X = self.X[(self.X[col] >= lower_bound) & (self.X[col] <= upper_bound)]
            
            # лбновляем целевую переменную
            self.y = self.y.loc[self.X.index]

        # удаление выбросов для целевой переменной
        if self.y.dtype != object and len(set(self.y)) > 10:
            Q1 = self.y.quantile(0.25)
            Q3 = self.y.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            mask = (self.y >= lower_bound) & (self.y <= upper_bound)
            self.X = self.X[mask]
            self.y = self.y[mask]

        # кодируем целевую переменную, если она категориальная
        if self.y.dtype == object or isinstance(self.y.iloc[0], str):
            le = LabelEncoder()
            self.y = le.fit_transform(self.y)
        else:
            # нормализуем целевую переменную для регрессии
            self.y = StandardScaler().fit_transform(self.y.values.reshape(-1, 1)).flatten()

        # обработка категориальных признаков
        categorical_cols = self.X.select_dtypes(include=['object', 'category', 'bool']).columns
        if self.encode_categorical and len(categorical_cols) > 0:
            for col in categorical_cols:
                self.X[col] = LabelEncoder().fit_transform(self.X[col].astype(str))

        # нормализация числовых признаков
        numeric_cols = self.X.select_dtypes(include=['int64', 'float64']).columns
        if self.normalize_numeric and len(numeric_cols) > 0:
            scaler = StandardScaler()
            self.X[numeric_cols] = scaler.fit_transform(self.X[numeric_cols])

        # проверяем, что все колонки - числовой тип
        non_numeric = self.X.select_dtypes(exclude=['int64', 'float64']).columns
        if len(non_numeric) > 0:
            raise ValueError(f"Необработанные нечисловые колонки: {list(non_numeric)}")

        # конвертируем в тензоры
        self.X = torch.tensor(self.X.values, dtype=torch.float32)
        self.y = torch.tensor(self.y, dtype=torch.float32 if len(set(self.y)) > 2 else torch.long)
    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
  
    def get_feature_dim(self):
        return self.X.shape[1]

# 2.2 Найдите csv датасеты для регрессии и бинарной классификации и, применяя наработки из предыдущей части задания, обучите линейную и логистическую регрессию

#  Линейная регрессия
def train_regression(dataset: str, target: str):
    dataset = CustomCSVDataset(dataset, target_column=target)
    train_data, test_data = train_test_split(dataset, test_size=0.2)
    
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=32)
    
    model = LinearRegression(dataset.get_feature_dim())
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    
    best_loss = float('inf')
    patience = 5
    patience_counter = 0
    train_losses = []
    
    for epoch in range(100):
        total_loss = 0
        for X, y in train_loader:
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y.unsqueeze(1)) + model.regularization_loss()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        train_losses.append(avg_loss)
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'models/house_sales_regression.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'Early stopping at epoch {epoch}')
                break
        
        if epoch % 10 == 0:
            log_epoch(epoch, avg_loss)
    
    plt.plot(train_losses)
    plt.title('Linear Regression Training Loss')
    plt.savefig('plots/house_sales_regression_loss.png')
    plt.close()

    # оценка на тестовой выборке
    model.load_state_dict(torch.load('models/house_sales_regression.pth'))
    model.eval()
    preds = []
    truths = []

    with torch.no_grad():
        for X, y in test_loader:
            outputs = model(X)
            preds.append(outputs.squeeze().cpu())
            truths.append(y.cpu())

    preds = torch.cat(preds).numpy()
    truths = torch.cat(truths).numpy()

    print(f"MSE: {mse(truths, preds):.4f}")
    print(f"MAE: {mae(truths, preds):.4f}")
    print(f"R²: {r2(truths, preds):.4f}")

# Логистическая регрессия 
def train_classification(dataset: str, target: str):
    dataset = CustomCSVDataset(dataset, target_column=target)
    num_classes = len(torch.unique(dataset.y))
    
    train_data, test_data = train_test_split(dataset, test_size=0.2)
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=32)
    
    model = LogisticRegression(dataset.get_feature_dim(), num_classes)
    criterion = nn.CrossEntropyLoss() if num_classes > 1 else nn.BCEWithLogitsLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    
    train_losses = []
    train_accuracies = []
    all_preds = []
    all_targets = []

    for epoch in range(100):
        total_loss = 0
        total_acc = 0
        
        for X, y in train_loader:
            optimizer.zero_grad()
            logits = model(X)
            
            if num_classes == 1:
                loss = criterion(logits.squeeze(), y.float())
            else:
                loss = criterion(logits, y.long().squeeze())
            
            loss.backward()
            optimizer.step()
            
            with torch.no_grad():
                if num_classes == 1:
                    probs = torch.sigmoid(logits)
                    y_pred = (probs > 0.5).float()
                    acc = accuracy(y_pred, y)
                else:
                    probs = torch.softmax(logits, dim=1)
                    y_pred = torch.argmax(probs, dim=1)
                    acc = accuracy(y_pred, y, num_classes)

                all_preds.extend(y_pred.cpu().numpy())
                all_targets.extend(y.cpu().numpy())
                total_loss += loss.item()
                total_acc += acc
        
        avg_loss = total_loss / len(train_loader)
        avg_acc = total_acc / len(train_loader)
        train_losses.append(avg_loss)
        train_accuracies.append(avg_acc)
        
        if epoch % 10 == 0:
            log_epoch(epoch, avg_loss, acc=avg_acc)
    
    print(f"Precision: {precision(all_targets, all_preds, num_classes)}")
    print(f"Recall: {recall(all_targets, all_preds, num_classes)}")
    print(f"F1: {f1(all_targets, all_preds, num_classes)}")
    
    plot_confusion_matrix(all_targets, all_preds, save_path='plots/tennis_confusion_matrix.png')
    
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses)
    plt.title('Training Loss')
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies)
    plt.title('Training Accuracy')
    plt.savefig('plots/tennis_classification_training.png')
    plt.close()
    
    torch.save(model.state_dict(), 'models/classification.pth')

if __name__ == '__main__':
    train_regression('data/house_sales.csv', 'AdjSalePrice')
    train_classification('data/play_tennis_dataset.csv', 'Play')
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from itertools import product

from homework_datasets import CustomCSVDataset
from homework_model_modification import LinearRegression, LogisticRegression
from utils import mse, r2, mae, accuracy, precision, recall, f1, plot_confusion_matrix, log_epoch

# 3.1 Проведите эксперименты с различными:
# - Скоростями обучения (learning rate)
# - Размерами батчей
# - Оптимизаторами (SGD, Adam, RMSprop)
# Визуализируйте результаты в виде графиков или таблиц

def hyperparameter_experiment(dataset_path, target_column, task_type='regression'):
    dataset = CustomCSVDataset(dataset_path, target_column)
    train_data, test_data = train_test_split(dataset, test_size=0.2)
    
    # определяем num_classes
    num_classes = len(torch.unique(dataset.y)) if task_type == 'classification' else None
    
    # параметры для экспериментов
    learning_rates = [0.01, 0.1]
    batch_sizes = [32, 64]
    optimizers = {
        'SGD': optim.SGD,
        'Adam': optim.Adam
    }
    
    results = []
    
    for lr, batch_size, (opt_name, opt_class) in product(learning_rates, batch_sizes, optimizers.items()):
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_data, batch_size=batch_size)
        
        if task_type == 'regression':
            model = LinearRegression(dataset.get_feature_dim())
            criterion = nn.MSELoss()
        else:
            model = LogisticRegression(dataset.get_feature_dim(), num_classes)
            criterion = nn.CrossEntropyLoss() if num_classes > 1 else nn.BCEWithLogitsLoss()
        
        optimizer = opt_class(model.parameters(), lr=lr)
        
        # обучение
        train_losses = []
        for epoch in range(40):
            total_loss = 0
            for X, y in train_loader:
                optimizer.zero_grad()
                outputs = model(X)
                
                if task_type == 'regression':
                    loss = criterion(outputs, y.unsqueeze(1)) + model.regularization_loss()
                else:
                    if num_classes == 1:
                        loss = criterion(outputs.squeeze(), y.float())
                    else:
                        loss = criterion(outputs, y.long())
                
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            avg_loss = total_loss / len(train_loader)
            train_losses.append(avg_loss)
        
        # Оценка
        model.eval()
        preds, truths = [], []
        with torch.no_grad():
            for X, y in test_loader:
                outputs = model(X)
                preds.append(outputs.cpu())
                truths.append(y.cpu())
        
        preds_tensor = torch.cat(preds)
        truths_tensor = torch.cat(truths)
        
        if task_type == 'regression':
            preds_array = preds_tensor.numpy()
            truths_array = truths_tensor.numpy()
            metrics = {
                'MSE': mse(truths_array, preds_array),
                'MAE': mae(truths_array, preds_array),
                'R2': r2(truths_array, preds_array)
            }
        else:
            if num_classes == 1:
                # для бинарной классификации
                preds_labels = (preds_tensor > 0.5).long()  # Получаем метки классов (тензор)
                preds_probs = torch.sigmoid(preds_tensor)  # Получаем вероятности (тензор)
            else:
                # для многоклассовой
                preds_labels = torch.argmax(preds_tensor, dim=1)  # Метки классов (тензор)
                preds_probs = torch.softmax(preds_tensor, dim=1)  # Вероятности (тензор)
            
            # преобразуем в numpy только то, что нужно для sklearn метрик
            truths_np = truths_tensor.numpy()
            preds_labels_np = preds_labels.numpy()
            preds_probs_np = preds_probs.numpy()
            
            # Вычисляем метрики
            metrics = {
                'Accuracy': accuracy(preds_tensor, truths_tensor, num_classes),  # Работает с тензорами
                'Precision': precision(truths_np, preds_labels_np, num_classes),  # Работает с numpy
                'Recall': recall(truths_np, preds_labels_np, num_classes),       # Работает с numpy
                'F1': f1(truths_np, preds_labels_np, num_classes)                # Работает с numpy
            }
        
        results.append({
            'Learning Rate': lr,
            'Batch Size': batch_size,
            'Optimizer': opt_name,
            **metrics,
            'Final Loss': avg_loss
        })
    
    # сохраняем результат
    results_df = pd.DataFrame(results)
    
    # визуализация
    for metric in results_df.columns[3:-1]:
        plt.figure(figsize=(10, 6))
        for opt in optimizers.keys():
            subset = results_df[results_df['Optimizer'] == opt]
            plt.plot(subset[metric], label=opt)
        
        plt.title(f'{metric} for different optimizers')
        plt.xlabel('Experiment')
        plt.ylabel(metric)
        plt.legend()
        plt.savefig(f'plots/{task_type}_{metric.lower()}_comparison.png')
        plt.close()
    
    return pd.DataFrame(results)

# 3.2 Создайте новые признаки для улучшения модели:
# - Полиномиальные признаки
# - Взаимодействия между признаками
# - Статистические признаки (среднее, дисперсия)
# Сравните качество с базовой моделью
def feature_engineering_experiment(dataset_path, target_column, task_type='regression'):
    #загрузка данных без предобработки
    data = pd.read_csv(dataset_path)
    X = data.drop(columns=[target_column])
    y = data[target_column]
    
    #преобразуем категориальные признаки в числовые
    X_processed = pd.get_dummies(X, drop_first=True)
    
    # проверяем, есть ли числовые признаки для преобразования
    numeric_cols = X_processed.select_dtypes(include=['number']).columns
    if len(numeric_cols) == 0:
        print("Warning: No numeric features found for polynomial transformation")
        X_enhanced = X_processed.copy()
    else:
        # полиномиальные признаки для числовых колонок
        poly = PolynomialFeatures(degree=2, include_bias=False)
        X_poly = poly.fit_transform(X_processed[numeric_cols])
        poly_columns = poly.get_feature_names_out(numeric_cols)
        X_poly = pd.DataFrame(X_poly, columns=poly_columns)
        
        # объединяем с исходными признаками
        X_enhanced = pd.concat([X_processed, X_poly], axis=1)
    
    #взаимодействия между признаками ( для числовых)
    if len(numeric_cols) > 1:
        interactions = []
        for i, col1 in enumerate(numeric_cols):
            for col2 in numeric_cols[i+1:]:
                new_col = f'{col1}_x_{col2}'
                interactions.append(pd.Series(X_processed[col1] * X_processed[col2], name=new_col))
        
        if interactions:
            interactions_df = pd.concat(interactions, axis=1)
            X_enhanced = pd.concat([X_enhanced, interactions_df], axis=1)
    
    # статистические признаки (для числовых)
    if len(numeric_cols) > 0:
        stats = pd.DataFrame()
        for col in numeric_cols:
            stats[f'{col}_mean'] = X_processed[col].rolling(window=5, min_periods=1).mean()
            stats[f'{col}_std'] = X_processed[col].rolling(window=5, min_periods=1).std()
        
        X_enhanced = pd.concat([X_enhanced, stats], axis=1)
    
    # заполняем NaN
    X_enhanced.fillna(0, inplace=True)
    
    # создание датасетов
    base_dataset = CustomCSVDataset(dataset_path, target_column)
    
    enhanced_data = pd.concat([X_enhanced, y], axis=1)
    enhanced_data.to_csv('data/enhanced_dataset.csv', index=False)
    enhanced_dataset = CustomCSVDataset('data/enhanced_dataset.csv', target_column)
    
    # рбучение и оценка
    def evaluate(dataset, name):
        train_data, test_data = train_test_split(dataset, test_size=0.2)
        train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_data, batch_size=32)
        
        if task_type == 'regression':
            model = LinearRegression(dataset.get_feature_dim())
            criterion = nn.MSELoss()
        else:
            num_classes = len(torch.unique(dataset.y))
            model = LogisticRegression(dataset.get_feature_dim(), num_classes)
            criterion = nn.CrossEntropyLoss() if num_classes > 1 else nn.BCEWithLogitsLoss()
        
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        
        # обучение
        for epoch in range(50):
            for X, y in train_loader:
                optimizer.zero_grad()
                outputs = model(X)
                
                if task_type == 'regression':
                    loss = criterion(outputs, y.unsqueeze(1)) + model.regularization_loss()
                else:
                    if num_classes == 1:
                        loss = criterion(outputs.squeeze(), y.float())
                    else:
                        loss = criterion(outputs, y.long())
                
                loss.backward()
                optimizer.step()
        
        # оценка
        model.eval()
        preds, truths = [], []
        with torch.no_grad():
            for X, y in test_loader:
                outputs = model(X)
                preds.append(outputs.cpu())
                truths.append(y.cpu())
        
        preds_array = torch.cat(preds).cpu().numpy()
        truths_array = torch.cat(truths).cpu().numpy()
        
        if task_type == 'regression':
            return {
                'Dataset': name,
                'MSE': mse(truths_array, preds_array),
                'MAE': mae(truths_array, preds_array),
                'R2': r2(truths_array, preds_array)
            }
        else:
            if num_classes == 1:
                preds_labels = (torch.cat(preds) > 0.5).long()
                preds_probs = torch.sigmoid(torch.cat(preds))
            else:
                preds_labels = torch.argmax(torch.cat(preds), dim=1)
                preds_probs = torch.softmax(torch.cat(preds), dim=1)
            
            truths_np = torch.cat(truths).numpy()
            preds_labels_np = preds_labels.numpy()
            
            return {
                'Dataset': name,
                'Accuracy': accuracy(torch.cat(preds), torch.cat(truths), num_classes),
                'Precision': precision(truths_np, preds_labels_np, num_classes),
                'Recall': recall(truths_np, preds_labels_np, num_classes),
                'F1': f1(truths_np, preds_labels_np, num_classes)
            }
    
    base_results = evaluate(base_dataset, 'Base')
    enhanced_results = evaluate(enhanced_dataset, 'Enhanced')
    
    # Сохраняем результат
    results_df = pd.DataFrame([base_results, enhanced_results])
    
    # визуализация
    metrics = list(base_results.keys())[1:]
    x = np.arange(len(metrics))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width/2, [base_results[m] for m in metrics], width, label='Base')
    ax.bar(x + width/2, [enhanced_results[m] for m in metrics], width, label='Enhanced')
    
    ax.set_ylabel('Score')
    ax.set_title('Feature Engineering Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    
    plt.savefig(f'plots/{task_type}_feature_engineering_comparison.png')
    plt.close()
    
    return results_df

if __name__ == '__main__':
    # Регрессия
    print("regression experiments")
    regression_hyper = hyperparameter_experiment('data/house_sales.csv', 'AdjSalePrice', 'regression')
    regression_fe = feature_engineering_experiment('data/house_sales.csv', 'AdjSalePrice', 'regression')
    
    # Классификация
    print("\nclassification experiments")
    classification_hyper = hyperparameter_experiment('data/play_tennis_dataset.csv', 'Play', 'classification')
    classification_fe = feature_engineering_experiment('data/play_tennis_dataset.csv', 'Play', 'classification')
    
    print("\nexperiments completed")
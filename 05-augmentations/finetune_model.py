import json
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import DataLoader
from datasets import CustomImageDataset
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tqdm import tqdm

def prepare_data():
    """Подготовка данных с использованием test в качестве val"""
    
    # Трансформации
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    
    train_dataset = CustomImageDataset('data/train', transform=train_transform)
    val_dataset = CustomImageDataset('data/test', transform=val_transform)  # test как val
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
    
    return train_loader, val_loader, train_dataset.get_class_names()

def prepare_model(class_names):
    """Подготовка предобученной модели"""
    model = models.resnet18(weights='IMAGENET1K_V1')
    
    # Заменяем последний слой
    num_classes = len(class_names)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    return model

def train_model(model, train_loader, val_loader, class_names, epochs=10):
    """Обучение модели с сохранением результатов"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    # История обучения
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_acc': []
    }
    
    best_acc = 0.0
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        # Прогресс-бар для эпохи
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')
        
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
        
        # Средний loss за эпоху
        epoch_loss = running_loss / len(train_loader)
        history['train_loss'].append(epoch_loss)
        
        # Валидация
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f'Epoch {epoch+1}/{epochs} - '
              f'Train Loss: {epoch_loss:.4f}, '
              f'Val Loss: {val_loss:.4f}, '
              f'Val Acc: {val_acc:.2f}%')
        
        # Сохраняем лучшую модель
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'results/finetuning/best_model.pth')
    
    # сохраняем историю обучения
    torch.save(history, 'results/finetuning/history.pth')
    
    # Визуализация
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history['val_acc'], label='Val Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    
    plt.savefig('results/finetuning/training_history.png')
    plt.close()

    history = torch.load('results/finetuning/history.pth')
    with open('results/finetuning/history.json', 'w') as f:
        json.dump({k: [float(x) for x in v] for k,v in history.items()}, f)
    
    return model, history

def validate(model, val_loader, criterion, device):
    """Валидация модели"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    val_loss = running_loss / len(val_loader)
    val_acc = 100. * correct / total
    
    return val_loss, val_acc

# Подготовка данных
'''train_loader, val_loader, class_names = prepare_data()
print(f"Количество классов: {len(class_names)}")
print(f"Размер train: {len(train_loader.dataset)}")
print(f"Размер val (test): {len(val_loader.dataset)}")

# Подготовка модели
model = prepare_model(class_names)
print("Модель подготовлена для дообучения")

# Обучение модели
print("Начало обучения...")
model, history = train_model(model, train_loader, val_loader, class_names, epochs=10)

print("Обучение завершено, результаты в results/finetuning/")'''



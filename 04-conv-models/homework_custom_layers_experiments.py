import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import numpy as np
import time
from tqdm import tqdm


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# загрузка CIFAR10
train_dataset = CIFAR10(root='./data', train=True, transform=ToTensor(), download=True)
test_dataset = CIFAR10(root='./data', train=False, transform=ToTensor())
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# 3.1 Реализуйте кастомные слои:
# - Кастомный сверточный слой с дополнительной логикой
# - Attention механизм для CNN
# - Кастомная функция активации
# - Кастомный pooling слой
# 
# Для каждого слоя:
# - Реализуйте forward и backward проходы
# - Добавьте параметры если необходимо
# - Протестируйте на простых примерах
# - Сравните с стандартными аналогами
class CustomConv2d(nn.Module):
    """Кастомный сверточный слой с дополнительной нормализацией и обучаемым порогом"""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        # Обучаемые веса
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.zeros(out_channels))
        
        # добавление кастомных параметров
        self.threshold = nn.Parameter(torch.tensor(0.1))  #обучаемый порог
        self.scale = nn.Parameter(torch.tensor(1.0))  #обучаемый масштабный коэффициент
        
        # инициализация весов
        nn.init.kaiming_normal_(self.weight, mode='fan_out', nonlinearity='relu')
        
    def forward(self, x):
        #стандартная свертка
        x = F.conv2d(x, self.weight, self.bias, self.stride, self.padding)
        
        # кастомные операции
        x = self.scale * F.relu(x - self.threshold)  # применение обучаемого порога и масштаба
        return x
    
    def extra_repr(self):
        return (f'in_channels={self.weight.size(1)}, out_channels={self.weight.size(0)}, '
                f'kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding}, '
                f'threshold={self.threshold.item():.3f}, scale={self.scale.item():.3f}')

class CNNWithAttention(nn.Module):
    """CNN с механизмом внимания"""
    def __init__(self, in_channels, reduction_ratio=8):
        super().__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction_ratio)
        self.spatial_attention = SpatialAttention()
        
    def forward(self, x):
        # применение channel attention
        x = self.channel_attention(x)
        
        # применение spatial attention
        x = self.spatial_attention(x)
        
        return x

class ChannelAttention(nn.Module):
    """Модуль Channel attention"""
    def __init__(self, in_channels, reduction_ratio=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        hidden_size = max(in_channels // reduction_ratio, 1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, hidden_size, bias=False),
            nn.ReLU(),
            nn.Linear(hidden_size, in_channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        b, c, _, _ = x.size()
        
        # усредняющий и максимальный пулинг
        avg_out = self.fc(self.avg_pool(x).view(b, c))
        max_out = self.fc(self.max_pool(x).view(b, c))
        
        #комбинирование и применение сигмоиды
        out = avg_out + max_out
        scale = self.sigmoid(out).view(b, c, 1, 1)
        
        return x * scale

class SpatialAttention(nn.Module):
    """Модуль Spatial attention"""
    def __init__(self, kernel_size=7):
        super().__init__()
        assert kernel_size % 2 == 1, "Kernel size must be odd"
        padding = kernel_size // 2
        
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # поиск максимума и среднего по каналам
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out = torch.max(x, dim=1, keepdim=True)[0]
        combined = torch.cat([avg_out, max_out], dim=1)
        
        # Применение свертки и сигмоиды
        attention = self.sigmoid(self.conv(combined))
        
        return x * attention

class CustomActivation(nn.Module):
    """Кастомная функция активации с обучаемыми параметрами"""
    def __init__(self):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(1.0))  # Коэффициент для отрицательных значений
        self.beta = nn.Parameter(torch.tensor(1.0))   # Коэффициент для положительных значений
        self.gamma = nn.Parameter(torch.tensor(0.0))  # параметр сдвига
        
    def forward(self, x):
        # Кусочно-линейная активация с обучаемыми параметрами
        pos = F.relu(x) * self.beta
        neg = -F.relu(-x) * self.alpha
        return pos + neg + self.gamma
    
    def extra_repr(self):
        return f'alpha={self.alpha.item():.3f}, beta={self.beta.item():.3f}, gamma={self.gamma.item():.3f}'

class CustomPooling(nn.Module):
    """кастомный слой пулинга с обучаемыми весами"""
    def __init__(self, kernel_size=2, stride=2):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        
        # веса имеют размер [groups, 1, kernel_size, kernel_size]
        self.weights = nn.Parameter(torch.ones(1, 1, kernel_size, kernel_size) / (kernel_size**2))
        
    def forward(self, x):
        # Применяем взвешенное среднее pooling
        weights = F.softmax(self.weights.view(-1), dim=0).view(1, 1, self.kernel_size, self.kernel_size)
        
        # Дублируем веса для всех каналов
        weights = weights.repeat(x.size(1), 1, 1, 1)
        
        return F.conv2d(x, weights, stride=self.stride, padding=0, groups=x.size(1))
    
    def extra_repr(self):
        return f'kernel_size={self.kernel_size}, stride={self.stride}'
    
def test_custom_layers():
    """Тестовая функция для кастомных слоев"""
    # тест CustomConv2d
    print("\nTesting CustomConv2d...")
    conv = CustomConv2d(3, 16, 3, padding=1).to(device)
    test_input = torch.randn(1, 3, 32, 32).to(device)
    output = conv(test_input)
    print(f"Input shape: {test_input.shape}, Output shape: {output.shape}")
    print(f"Conv parameters: {dict(conv.named_parameters())}")
    
    # тест CNNWithAttention
    print("\nTesting CNNWithAttention...")
    attn = CNNWithAttention(16).to(device)
    output = attn(output)
    print(f"Output shape after attention: {output.shape}")
    
    # тест CustomActivation
    print("\nTesting CustomActivation...")
    activation = CustomActivation().to(device)
    output = activation(output)
    print(f"Output shape after activation: {output.shape}")
    print(f"Activation parameters: alpha={activation.alpha.item():.3f}, beta={activation.beta.item():.3f}, gamma={activation.gamma.item():.3f}")
    
    # тест CustomPooling
    print("\nTesting CustomPooling...")
    pool = CustomPooling(2, 2).to(device)
    output = pool(output)
    print(f"Output shape after pooling: {output.shape}")
    print(f"Pooling weights: {pool.weights.data.squeeze()}")

# 3.2 Исследуйте различные варианты Residual блоков:
# - Базовый Residual блок
# - Bottleneck Residual блок
# - Wide Residual блок
# 
# Для каждого варианта:
# - Реализуйте блок с нуля
# - Сравните производительность
# - Проанализируйте количество параметров
# - Исследуйте стабильность обучения
class BasicResidualBlock(nn.Module):
    """Базовый остаточный блок с двумя свертками 3x3"""
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)

class BottleneckResidualBlock(nn.Module):
    """Остаточный блок типа 'Bottleneck' со свертками 1x1-3x3-1x1"""
    def __init__(self, in_channels, out_channels, stride=1, expansion=4):
        super().__init__()
        mid_channels = out_channels // expansion
        
        self.conv1 = nn.Conv2d(in_channels, mid_channels, 1, 1, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, 3, stride, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_channels)
        self.conv3 = nn.Conv2d(mid_channels, out_channels, 1, 1, 0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        return F.relu(out)

class WideResidualBlock(nn.Module):
    """Широкий остаточный блок с увеличенным количеством карт признаков"""
    def __init__(self, in_channels, out_channels, stride=1, widen_factor=2):
        super().__init__()
        mid_channels = out_channels * widen_factor
        
        self.conv1 = nn.Conv2d(in_channels, mid_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.dropout = nn.Dropout(0.3)
        self.conv2 = nn.Conv2d(mid_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)

def build_resnet(block_type, num_blocks, num_classes=10):
    """Вспомогательная функция для построения различных вариантов ResNet"""
    class ResNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.in_channels = 64
            
            self.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
            self.bn1 = nn.BatchNorm2d(64)
            self.layer1 = self._make_layer(block_type, 64, num_blocks[0], 1)
            self.layer2 = self._make_layer(block_type, 128, num_blocks[1], 2)
            self.layer3 = self._make_layer(block_type, 256, num_blocks[2], 2)
            self.layer4 = self._make_layer(block_type, 512, num_blocks[3], 2)
            self.avgpool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Linear(512, num_classes)
            
        def _make_layer(self, block_type, out_channels, num_blocks, stride):
            strides = [stride] + [1]*(num_blocks-1)
            layers = []
            for stride in strides:
                layers.append(block_type(self.in_channels, out_channels, stride))
                self.in_channels = out_channels
            return nn.Sequential(*layers)
        
        def forward(self, x):
            x = F.relu(self.bn1(self.conv1(x)))
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
            return self.fc(x)
    
    return ResNet()

def train_and_evaluate(model, name, train_loader, test_loader, epochs=10, lr=0.001):
    """Обучение и оценка модели"""
    model = model().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # история обучения
    history = {
        'train_loss': [],
        'train_acc': [],
        'test_loss': [],
        'test_acc': []
    }
    
    # цикл обучения
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        
        # запись метрик
        train_loss /= len(train_loader)
        train_acc = correct / total
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        
        # оценка
        model.eval()
        test_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        # запись метрик
        test_loss /= len(test_loader)
        test_acc = correct / total
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)
        
        print(f"{name} - Epoch {epoch+1}/{epochs}: "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
              f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    
    return history, num_params

def run_residual_experiments():
    """Проведение экспериментов с различными остаточными блоками"""
    # Определение моделей для сравнения
    models = [
        (lambda: build_resnet(BasicResidualBlock, [1, 1, 1, 1]), "BasicResNet"),
        (lambda: build_resnet(BottleneckResidualBlock, [1, 1, 1, 1]), "BottleneckResNet"),
        (lambda: build_resnet(WideResidualBlock, [1, 1, 1, 1]), "WideResNet")
    ]
    
    results = {}
    
    #Обучение и оценка каждой модели
    for model_fn, name in models:
        print(f"\nTraining {name}...")
        start_time = time.time()
        history, num_params = train_and_evaluate(model_fn, name, train_loader, test_loader, epochs=10)
        train_time = time.time() - start_time
        
        results[name] = {
            'history': history,
            'num_params': num_params,
            'train_time': train_time
        }
        
        # построение кривых обучения
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.plot(history['train_loss'], label='Train')
        plt.plot(history['test_loss'], label='Test')
        plt.title(f'{name} Loss')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(history['train_acc'], label='Train')
        plt.plot(history['test_acc'], label='Test')
        plt.title(f'{name} Accuracy')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(f'plots/custom_layers/{name}_learning_curve.png')
        plt.close()
    
    # сохранить и показать результат
    with open('results/custom_layers/residual_results.txt', 'w') as f:
        f.write("Model Comparison Results:\n")
        f.write("{:<20} {:<15} {:<15} {:<15} {:<15}\n".format(
            "Model", "Train Acc", "Test Acc", "Params", "Time (s)"))
        
        for name, res in results.items():
            f.write("{:<20} {:<15.4f} {:<15.4f} {:<15} {:<15.2f}\n".format(
                name,
                res['history']['train_acc'][-1],
                res['history']['test_acc'][-1],
                res['num_params'],
                res['train_time']
            ))
    
    # вывксти результат
    print("\nModel Comparison Results:")
    print("{:<20} {:<15} {:<15} {:<15} {:<15}".format(
        "Model", "Train Acc", "Test Acc", "Params", "Time (s)"))
    
    for name, res in results.items():
        print("{:<20} {:<15.4f} {:<15.4f} {:<15} {:<15.2f}".format(
            name,
            res['history']['train_acc'][-1],
            res['history']['test_acc'][-1],
            res['num_params'],
            res['train_time']
        ))

if __name__ == "__main__":
    print("Running custom layers tests...")
    test_custom_layers()
    
    print("\nRunning residual blocks experiments...")
    run_residual_experiments()
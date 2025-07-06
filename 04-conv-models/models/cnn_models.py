import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
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

class SimpleCNN(nn.Module):
    def __init__(self, input_channels=1, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, 3, 1, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1, 1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.25)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)

class ResidualCNN(nn.Module):
    def __init__(self, input_channels=1, num_classes=10, dropout_rate=0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(32)
        
        self.res1 = ResidualBlock(32, 32)
        self.res2 = ResidualBlock(32, 64, 2)
        self.res3 = ResidualBlock(64, 64)
        
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(64 * 4 * 4, num_classes)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        return self.fc(x)
    
class ConvSizeExperiment(nn.Module):
    """Модель для эксперимента с размером ядра"""
    def __init__(self, input_channels=3, num_classes=10, kernels=[3, 3, 3]):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernels[0], padding=kernels[0]//2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernels[1], padding=kernels[1]//2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernels[2], padding=kernels[2]//2),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        self.classifier = nn.Linear(128, num_classes)
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)
    
    def calculate_receptive_field(self):
        """Вычисляет рецептивное поле"""
        rf = 1
        for layer in self.features:
            if isinstance(layer, nn.Conv2d):
                rf += (layer.kernel_size[0] - 1) * layer.dilation[0]
            elif isinstance(layer, nn.MaxPool2d):
                rf *= layer.kernel_size
        return rf

class DepthExperiment(nn.Module):
    """Модель для эксперимента с глубиной"""
    def __init__(self, input_channels=3, num_classes=10, num_blocks=4, residual=False):
        super().__init__()
        self.residual = residual
        
        layers = []
        in_channels = input_channels
        out_channels = 32
        
        for i in range(num_blocks):
            # Уменьшаем размер только каждые 2 блока
            pool = nn.MaxPool2d(2) if i % 2 == 1 else nn.Identity()
            layers.append(self._make_block(in_channels, out_channels, pool))
            in_channels = out_channels
            out_channels = min(out_channels * 2, 256)
            
        self.features = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(in_channels, num_classes)
        
    def _make_block(self, in_channels, out_channels, pool_layer):
        """Создает блок (обычный или residual) с заданным пулингом"""
        if self.residual and in_channels == out_channels:
            return nn.Sequential(
                ResidualBlock(in_channels, out_channels),
                pool_layer
            )
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            pool_layer
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)
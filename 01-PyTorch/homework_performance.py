import torch
import time 

# 3.1 Создайте большие матрицы размеров:
# - 64 x 1024 x 1024
# - 128 x 512 x 512
# - 256 x 256 x 256
# Заполните их случайными числами

tensors = {
    '64x1024x1024': torch.rand(64, 1024, 1024),
    '128x512x512': torch.rand(128, 512, 512),
    '256x256x256': torch.rand(256, 256, 256)
}

# 3.2 Создайте функцию для измерения времени выполнения операций
# Используйте torch.cuda.Event() для точного измерения на GPU
# Используйте time.time() для измерения на CPU

def measure_cpu(op, t1, t2=None):
    """Измеряет время выполнения операции на CPU.
    
    Args:
        op (str): Тип операции. Допустимые значения:
            - 'matmul': матричное умножение
            - 'add': поэлементное сложение
            - 'mul': поэлементное умножение
            - 'transpose': транспонирование
            - 'sum': сумма элементов
        t1 (torch.Tensor): Основной входной тензор
        t2 (torch.Tensor, optional): Второй тензор для бинарных операций. По умолчанию None.
    """
    start = time.time()
    if op == 'matmul':
        torch.matmul(t1, t2)
    elif op == 'add':
        t1 + t2
    elif op == 'mul':
        t1 * t2
    elif op == 'transpose':
        t1.transpose(-2, -1).contiguous() #для явного перераспределения памяти
    elif op == 'sum':
        t1.sum()
    return (time.time() - start) * 1000  # мс

def measure_gpu(op, t1, t2=None):
    """Измеряет время выполнения операции на GPU с использованием CUDA Events.
    
    Args:
        op (str): Тип операции. Допустимые значения:
            - 'matmul': матричное умножение
            - 'add': поэлементное сложение
            - 'mul': поэлементное умножение
            - 'transpose': транспонирование
            - 'sum': сумма элементов
        t1 (torch.Tensor): Основной входной тензор (на GPU)
        t2 (torch.Tensor, optional): Второй тензор для бинарных операций. По умолчанию None.
    """
    if not torch.cuda.is_available():
        return float('nan')
    
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record()
    if op == 'matmul':
        torch.matmul(t1, t2)
    elif op == 'add':
        t1 + t2
    elif op == 'mul':
        t1 * t2
    elif op == 'transpose':
        t1.transpose(-2, -1).contiguous()
    elif op == 'sum':
        t1.sum()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end)  # мс

# 3.3 Сравните время выполнения следующих операций на CPU и CUDA:
# - Матричное умножение (torch.matmul)
# - Поэлементное сложение
# - Поэлементное умножение
# - Транспонирование
# - Вычисление суммы всех элементов

# Для каждой операции:
# 1. Измерьте время на CPU
# 2. Измерьте время на GPU (если доступен)
# 3. Вычислите ускорение (speedup)
# 4. Выведите результаты в табличном виде
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
operations = [
    ('matmul', 'Матричное умножение'),
    ('add', 'Сложение'),
    ('mul', 'Умножение'),
    ('transpose', 'Транспонирование'),
    ('sum', 'Сумма')
]

for size_name, tensor in tensors.items():
    print(f"\nРазмер тензора: {size_name}")
    print("-"*80)
    print(f"{'Операция':<20} | {'CPU (мс)':>10} | {'GPU (мс)':>10} | {'Ускорение':>10}")
    print("-"*80)
    
    # Для matmul нужен квадратный тензор
    if size_name == '64x1024x1024':
        matmul_tensor = tensor[:, :1024, :1024]
    else:
        matmul_tensor = tensor
    
    for op_key, op_name in operations:
        # Копии тензоров для каждого теста
        t_cpu = matmul_tensor.clone() if op_key == 'matmul' else tensor.clone()
        t_gpu = t_cpu.to(device) if torch.cuda.is_available() else None
        
        # Измерение времени
        cpu_time = measure_cpu(op_key, t_cpu, t_cpu if op_key in ['matmul', 'add', 'mul'] else None)
        
        if torch.cuda.is_available():
            gpu_time = measure_gpu(op_key, t_gpu, t_gpu if op_key in ['matmul', 'add', 'mul'] else None)
            speedup = cpu_time / gpu_time
            gpu_str = f"{gpu_time:>10.1f}"
            speedup_str = f"{speedup:>8.1f}x"
        else:
            gpu_str = "      N/A"
            speedup_str = "     N/A"
        
        print(f"{op_name:<20} | {cpu_time:>10.1f} | {gpu_str} | {speedup_str}")
    print("-"*80)
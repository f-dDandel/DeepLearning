import os
import time
import torch
import psutil
import matplotlib.pyplot as plt
from datasets import CustomImageDataset
from augmentation_pipeline import create_pipeline_configurations

def measure_performance(dataset, pipeline, num_images=100):
    """Измеряет время и память для обработки изображений"""
    process = psutil.Process(os.getpid())
    
    # Измеряем начальное использование памяти
    start_mem = process.memory_info().rss / (1024 * 1024)  # в MB
    
    start_time = time.time()
    
    for i in range(min(num_images, len(dataset))):
        img, _ = dataset[i]
        _ = pipeline.apply(img)
    
    end_time = time.time()
    
    # Измеряем пиковое использование памяти
    end_mem = process.memory_info().rss / (1024 * 1024)  # в MB
    
    return {
        "time": end_time - start_time,
        "memory": end_mem - start_mem
    }

def run_experiment(sizes, num_images=100):
    """Запускает эксперимент для разных размеров"""
    results = {}
    pipeline = create_pipeline_configurations()["medium"]  # используем средние аугментации
    
    for size in sizes:
        print(f"Запуск для размера {size}x{size}...")
        
        # Загружаем датасет с текущим размером
        dataset = CustomImageDataset(
            root_dir="data/train",
            transform=None,
            target_size=(size, size))
        
        # Измеряем производительность
        metrics = measure_performance(dataset, pipeline, num_images)
        results[size] = metrics
        
        print(f"Размер {size}x{size}: {metrics['time']:.2f} сек, {metrics['memory']:.2f} MB")
    
    return results

def plot_results(results, output_dir="results/size_experiment"):
    """Строит графики результатов"""
    
    sizes = list(results.keys())
    times = [results[size]["time"] for size in sizes]
    memories = [results[size]["memory"] for size in sizes]
    
    # График времени выполнения
    plt.figure(figsize=(10, 5))
    plt.plot(sizes, times, 'o-')
    plt.title("Зависимость времени обработки от размера изображения")
    plt.xlabel("Размер изображения (пиксели)")
    plt.ylabel("Время обработки 100 изображений (сек)")
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "time_vs_size.png"))
    plt.close()
    
    # График использования памяти
    plt.figure(figsize=(10, 5))
    plt.plot(sizes, memories, 'o-')
    plt.title("Зависимость потребления памяти от размера изображения")
    plt.xlabel("Размер изображения (пиксели)")
    plt.ylabel("Дополнительная память (MB)")
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "memory_vs_size.png"))
    plt.close()
    
    # Сохраняем результаты в файл
    with open(os.path.join(output_dir, "results.txt"), "w", encoding='utf-8') as f:
        f.write("Размер\tВремя (с)\tПамять (MB)\n")
        for size in sizes:
            f.write(f"{size}\t{results[size]['time']:.2f}\t{results[size]['memory']:.2f}\n")

def main():
    # Размеры для эксперимента
    sizes = [64, 128, 224, 512]
    
    # Запускаем эксперимент
    results = run_experiment(sizes)
    
    # Строим графики
    plot_results(results)
    
    print("Эксперимент завершен. Результаты сохранены в results/size_experiment/")

if __name__ == "__main__":
    main()
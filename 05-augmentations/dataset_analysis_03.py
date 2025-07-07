import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from collections import defaultdict

def analyze_dataset(dataset_path):
    """Анализирует датасет и собирает статистику"""
    # Словарь для подсчета изображений по классам
    class_counts = defaultdict(int)
    # Список для хранения размеров изображений
    sizes = []
    
    # Проверяем существование папки с данными
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Директория {dataset_path} не найдена")
    
    # Получаем список классов (папок)
    try:
        classes = [d for d in os.listdir(dataset_path) 
                 if os.path.isdir(os.path.join(dataset_path, d))]
    except Exception as e:
        raise Exception(f"Ошибка при чтении директории: {e}")

    if not classes:
        raise ValueError("В указанной директории нет подпапок с классами")

    # Собираем данные по каждому классу
    for class_name in classes:
        class_dir = os.path.join(dataset_path, class_name)
        
        try:
            images = [f for f in os.listdir(class_dir) 
                     if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
        except Exception as e:
            print(f"Ошибка при чтении класса {class_name}: {e}")
            continue
        
        class_counts[class_name] = len(images)
        
        # Анализируем размеры изображений
        for img_name in images:
            img_path = os.path.join(class_dir, img_name)
            try:
                with Image.open(img_path) as img:
                    sizes.append(img.size)  # (width, height)
            except Exception as e:
                print(f"Ошибка при обработке {img_path}: {e}")
                continue
    
    if not sizes:
        raise ValueError("Не найдено ни одного изображения для анализа")

    sizes_array = np.array(sizes)
    
    # Рассчитываем статистику
    stats = {
        "total_images": sum(class_counts.values()),
        "num_classes": len(class_counts),
        "min_size": tuple(np.min(sizes_array, axis=0)),
        "max_size": tuple(np.max(sizes_array, axis=0)),
        "mean_size": tuple(np.mean(sizes_array, axis=0)),
        "median_size": tuple(np.median(sizes_array, axis=0)),
        "class_distribution": dict(class_counts),
        "sizes": sizes_array  # сохраняем массив размеров
    }
    
    return stats

def visualize_stats(stats, output_dir="results/dataset_analysis"):
    """Визуализирует статистику датасета"""
    
    # Гистограмма распределения по классам
    plt.figure(figsize=(12, 6))
    classes = list(stats["class_distribution"].keys())
    counts = list(stats["class_distribution"].values())
    
    plt.bar(classes, counts)
    plt.title(f"Распределение изображений по классам (всего {stats['total_images']})")
    plt.xlabel("Классы")
    plt.ylabel("Количество изображений")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "class_distribution.png"))
    plt.close()
    
    #  Распределение размеров изображений
    plt.figure(figsize=(12, 6))
    widths = stats["sizes"][:, 0]
    heights = stats["sizes"][:, 1]
    
    plt.scatter(widths, heights, alpha=0.5)
    plt.title("Распределение размеров изображений")
    plt.xlabel("Ширина (пиксели)")
    plt.ylabel("Высота (пиксели)")
    
    plt.scatter(*stats["min_size"], color='red', label=f'Min: {stats["min_size"][0]}x{stats["min_size"][1]}')
    plt.scatter(*stats["max_size"], color='green', label=f'Max: {stats["max_size"][0]}x{stats["max_size"][1]}')
    plt.scatter(*stats["mean_size"], color='blue', 
               label=f'Mean: {int(stats["mean_size"][0])}x{int(stats["mean_size"][1])}')
    
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "size_distribution.png"))
    plt.close()
    
    #  Сохраняем текстовый отчет
    with open(os.path.join(output_dir, "dataset_report.txt"), 'w', encoding='utf-8') as f:
        f.write("=== Анализ датасета ===\n\n")
        f.write(f"Всего изображений: {stats['total_images']}\n")
        f.write(f"Количество классов: {stats['num_classes']}\n\n")
        
        f.write("Размеры изображений:\n")
        f.write(f"Минимальный: {stats['min_size'][0]}x{stats['min_size'][1]} (ширина x высота)\n")
        f.write(f"Максимальный: {stats['max_size'][0]}x{stats['max_size'][1]}\n")
        f.write(f"Средний: {int(stats['mean_size'][0])}x{int(stats['mean_size'][1])}\n")
        f.write(f"Медианный: {int(stats['median_size'][0])}x{int(stats['median_size'][1])}\n\n")
        
        f.write("Распределение по классам:\n")
        max_name_length = max(len(name) for name in stats["class_distribution"].keys())
        
        for class_name, count in stats["class_distribution"].items():
            f.write(f"{class_name.ljust(max_name_length)} : {count} изображений\n")

def main():
    try:
        dataset_path = "data/train"
        stats = analyze_dataset(dataset_path)
        visualize_stats(stats)
        print("Анализ датасета завершен. Результаты сохранены в results/dataset_analysis/")
    except Exception as e:
        print(f"Произошла ошибка: {e}")

if __name__ == "__main__":
    main()
import os
import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from datasets import CustomImageDataset

def create_standard_augmentations():
    """Создает пайплайн стандартных аугментаций torchvision"""
    return {
        "RandomHorizontalFlip": transforms.RandomHorizontalFlip(p=1.0),
        "RandomCrop": transforms.RandomCrop(200, padding=20),
        "ColorJitter": transforms.ColorJitter(
            brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1
        ),
        "RandomRotation": transforms.RandomRotation(degrees=30),
        "RandomGrayscale": transforms.RandomGrayscale(p=1.0),
    }

def apply_and_save_augmentations(dataset, num_images=5):
    """Применяет аугментации к изображениям и сохраняет результаты"""
    augs = create_standard_augmentations()
    
    # Выбираем по одному изображению из разных классов
    selected_indices = []
    current_class = -1
    for idx in range(len(dataset)):
        _, label = dataset[idx]
        if label != current_class:
            selected_indices.append(idx)
            current_class = label
            if len(selected_indices) == num_images:
                break
    
    # Создаем комбинированную аугментацию
    combined_aug = transforms.Compose(list(augs.values()))
    
    # Применяем аугментации к каждому выбранному изображению
    for i, idx in enumerate(selected_indices):
        original_img, label = dataset[idx]
        class_name = dataset.get_class_names()[label]
        
        # Создаем фигуру для визуализации отдельных аугментаций
        plt.figure(figsize=(15, 8))
        plt.suptitle(f"Класс: {class_name}", fontsize=16)
        
        # Отображаем оригинальное изображение
        plt.subplot(2, 3, 1)
        plt.imshow(original_img)
        plt.title("Оригинал")
        plt.axis('off')
        
        # Применяем и отображаем каждую аугментацию отдельно
        for j, (name, aug) in enumerate(augs.items()):
            augmented_img = aug(original_img)
            plt.subplot(2, 3, j+2)
            plt.imshow(augmented_img)
            plt.title(name)
            plt.axis('off')
        
        # Сохраняем результаты отдельных аугментаций
        plt.tight_layout()
        plt.savefig(f"results/standard_augs/class_{class_name}_separate.png", bbox_inches='tight', dpi=150)
        plt.close()
        
        # Применяем и сохраняем комбинированные аугментации
        combined_img = combined_aug(original_img)
        plt.figure(figsize=(8, 4))
        plt.subplot(1, 2, 1)
        plt.imshow(original_img)
        plt.title("Оригинал")
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.imshow(combined_img)
        plt.title("Все аугментации")
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(f"results/standard_augs/class_{class_name}_combined.png", bbox_inches='tight', dpi=150)
        plt.close()

def main():
    # Загружаем датасет без аугментаций
    train_dataset = CustomImageDataset(
        root_dir="data/train",  # Исправлено с root на root_dir
        transform=None,
        target_size=(224, 224))
    
    # Проверяем, что датасет загрузился
    print(f"Всего классов: {len(train_dataset.get_class_names())}")
    print(f"Всего изображений: {len(train_dataset)}")
    
    # Применяем и сохраняем аугментации
    apply_and_save_augmentations(train_dataset)
    
    print("Результаты сохранены в папку results/standard_augs/")

if __name__ == "__main__":
    main()
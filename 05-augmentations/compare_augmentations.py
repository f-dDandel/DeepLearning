import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from datasets import CustomImageDataset
from extra_augs import (AddGaussianNoise, RandomErasingCustom, CutOut)
from custom_augs import (RandomBlur, RandomPerspective, RandomBrightnessContrast)

def compare_custom_augmentations(dataset, num_images=3):
    """Сравнивает кастомные аугментации с готовыми"""
    # Наши новые аугментации
    custom_augs = {
        "RandomBlur": RandomBlur(p=1.0),
        "RandomPerspective": RandomPerspective(p=1.0),
        "RandomBrightnessContrast": RandomBrightnessContrast(p=1.0)
    }
    
    # Готовые аугментации из extra_augs.py
    existing_augs = {
        "AddGaussianNoise": AddGaussianNoise(0., 0.2),
        "RandomErasing": RandomErasingCustom(p=1.0),
        "CutOut": CutOut(p=1.0, size=(32, 32))
    }
    
    # Выбираем изображения из разных классов
    selected_indices = []
    current_class = -1
    for idx in range(len(dataset)):
        _, label = dataset[idx]
        if label != current_class:
            selected_indices.append(idx)
            current_class = label
            if len(selected_indices) == num_images:
                break
    
    
    for i, idx in enumerate(selected_indices):
        original_img, label = dataset[idx]
        class_name = dataset.get_class_names()[label]
        
        # Создаем фигуру для сравнения
        plt.figure(figsize=(15, 10))
        plt.suptitle(f"Сравнение аугментаций для класса: {class_name}", fontsize=16)
        
        # Оригинальное изображение
        plt.subplot(3, 3, 1)
        plt.imshow(original_img)
        plt.title("Оригинал")
        plt.axis('off')
        
        # Наши кастомные аугментации
        for j, (name, aug) in enumerate(custom_augs.items()):
            augmented_img = aug(original_img)
            plt.subplot(3, 3, j+2)
            plt.imshow(augmented_img)
            plt.title(f"Кастомная: {name}")
            plt.axis('off')
        
        # Готовые аугментации
        to_tensor = transforms.ToTensor()
        for k, (name, aug) in enumerate(existing_augs.items()):
            # Преобразуем в тензор для готовых аугментаций
            img_tensor = to_tensor(original_img)
            augmented_img = aug(img_tensor)
            # Обратно в PIL для отображения
            augmented_img = transforms.ToPILImage()(augmented_img)
            
            plt.subplot(3, 3, k+5)
            plt.imshow(augmented_img)
            plt.title(f"Готовая: {name}")
            plt.axis('off')
        
        # Сохраняем результаты
        plt.tight_layout()
        plt.savefig(f"results/custom_augs/compare_{class_name}.png", bbox_inches='tight', dpi=150)
        plt.close()

def main():
    # Загружаем датасет без аугментаций
    train_dataset = CustomImageDataset(
        root_dir="data/train",
        transform=None,
        target_size=(224, 224))
    
    # Сравниваем аугментации
    compare_custom_augmentations(train_dataset)
    
    print("Результаты сравнения сохранены в папку results/custom_augs/")

if __name__ == "__main__":
    main()
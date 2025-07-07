import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from datasets import CustomImageDataset

class AugmentationPipeline:
    """Класс для создания и управления пайплайном аугментаций"""
    
    def __init__(self):
        self.augmentations = {}
        self.pipeline = transforms.Compose([])
    
    def add_augmentation(self, name, augmentation):
        """Добавляет аугментацию в пайплайн"""
        if name in self.augmentations:
            raise ValueError(f"Аугментация с именем '{name}' уже существует")
        self.augmentations[name] = augmentation
        self._update_pipeline()
    
    def remove_augmentation(self, name):
        """Удаляет аугментацию из пайплайна"""
        if name not in self.augmentations:
            raise ValueError(f"Аугментация с именем '{name}' не найдена")
        del self.augmentations[name]
        self._update_pipeline()
    
    def _update_pipeline(self):
        """Обновляет внутренний пайплайн трансформаций"""
        self.pipeline = transforms.Compose(list(self.augmentations.values()))
    
    def apply(self, image):
        """Применяет все аугментации к изображению"""
        return self.pipeline(image)
    
    def get_augmentations(self):
        """Возвращает словарь всех аугментаций"""
        return self.augmentations.copy()

def create_pipeline_configurations():
    """Создает несколько конфигураций аугментаций"""
    configurations = {}
    
    # Легкие аугментации (минимальные изменения)
    light = AugmentationPipeline()
    light.add_augmentation("random_flip", transforms.RandomHorizontalFlip(p=0.3))
    light.add_augmentation("color_jitter", transforms.ColorJitter(
        brightness=0.1, contrast=0.1, saturation=0.1))
    configurations["light"] = light
    
    # Средние аугментации (умеренные изменения)
    medium = AugmentationPipeline()
    medium.add_augmentation("random_flip", transforms.RandomHorizontalFlip(p=0.5))
    medium.add_augmentation("color_jitter", transforms.ColorJitter(
        brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05))
    medium.add_augmentation("random_rotate", transforms.RandomRotation(degrees=15))
    configurations["medium"] = medium
    
    # Сильные аугментации (значительные изменения)
    heavy = AugmentationPipeline()
    heavy.add_augmentation("random_flip", transforms.RandomHorizontalFlip(p=0.7))
    heavy.add_augmentation("color_jitter", transforms.ColorJitter(
        brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1))
    heavy.add_augmentation("random_rotate", transforms.RandomRotation(degrees=30))
    heavy.add_augmentation("random_crop", transforms.RandomResizedCrop(
        size=224, scale=(0.7, 1.0)))
    configurations["heavy"] = heavy
    
    return configurations

def apply_and_save_pipelines(dataset, pipelines, num_images=5):
    """Применяет пайплайны к изображениям и сохраняет результаты"""
    
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
    
    # Применяем каждый пайплайн к выбранным изображениям
    for idx in selected_indices:
        original_img, label = dataset[idx]
        class_name = dataset.get_class_names()[label]
        
        # Создаем фигуру для сравнения
        fig, axes = plt.subplots(1, len(pipelines) + 1, figsize=(15, 5))
        fig.suptitle(f"Сравнение пайплайнов для класса: {class_name}", fontsize=14)
        
        # Оригинальное изображение
        axes[0].imshow(original_img)
        axes[0].set_title("Оригинал")
        axes[0].axis('off')
        
        # Применяем каждый пайплайн
        for i, (name, pipeline) in enumerate(pipelines.items(), start=1):
            augmented_img = pipeline.apply(original_img)
            axes[i].imshow(augmented_img)
            axes[i].set_title(name.capitalize())
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.savefig(f"results/pipelines/compare_{class_name}.png", bbox_inches='tight', dpi=150)
        plt.close()
        
        # Сохраняем по одному изображению для каждого пайплайна
        for name, pipeline in pipelines.items():
            augmented_img = pipeline.apply(original_img)
            plt.figure(figsize=(5, 5))
            plt.imshow(augmented_img)
            plt.title(f"{name.capitalize()} pipeline")
            plt.axis('off')
            plt.savefig(f"results/pipelines/{class_name}_{name}.png", bbox_inches='tight', dpi=150)
            plt.close()

def main():
    # Загружаем датасет без аугментаций
    train_dataset = CustomImageDataset(
        root_dir="data/train",
        transform=None,
        target_size=(224, 224))
    
    # Создаем конфигурации пайплайнов
    pipelines = create_pipeline_configurations()
    
    # Применяем и сохраняем результаты
    apply_and_save_pipelines(train_dataset, pipelines)
    
    print("Результаты сохранены в папку results/pipelines/")

if __name__ == "__main__":
    main()
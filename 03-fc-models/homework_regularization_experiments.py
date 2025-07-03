import json
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from utils.dataset_utils import get_mnist_loaders
from utils.model_utils import create_model, count_parameters
from utils.visualization_utils import plot_learning_curves, plot_weight_distributions


# 3.1 Изучите различные методы регуляризации:
# - Без регуляризации
# - Только Dropout (разные коэффициенты: 0,1, 0,3, 0,5)
# - Только BatchNorm
# - Dropout + BatchNorm
# - Регуляризация L2 (снижение веса)
# Для каждого варианта:
# - Используйте одинаковую архитектуру
# - Сравните итоговую точность
# - Проанализируйте стабильность обучения
# - Визуализируйте распределение весов
class RegularizationExperiment:
    def __init__(self, dataset='mnist', batch_size=128):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dataset = dataset
        self.batch_size = batch_size
        self.base_config = {
            "input_size": 784 if dataset == 'mnist' else 32*32*3,
            "num_classes": 10,
            "layers": [
                {"type": "linear", "size": 256}, {"type": "relu"},
                {"type": "linear", "size": 128}, {"type": "relu"},
                {"type": "linear", "size": 64}, {"type": "relu"}
            ]
        }

    def run(self, epochs=15):
        """Запуск всех экспериментов по регуляризации"""
        train_loader, test_loader = get_mnist_loaders(batch_size=self.batch_size)
        
        # 3.1 сравнение методов регуляризации
        print("\nRunning Regularization Comparison...")
        reg_results = self.compare_regularization_methods(train_loader, test_loader, epochs)
        
        # 3.2 адаптивная регуляризация
        print("\nRunning Adaptive Regularization...")
        adaptive_results = self.run_adaptive_regularization(train_loader, test_loader, epochs)
        
        # сохранение результатов
        all_results = {
            "regularization_comparison": reg_results,
            "adaptive_regularization": adaptive_results
        }
        with open(f"results/regularization_experiments/{self.dataset}_results.json", 'w') as f:
            json.dump(all_results, f, indent=4)
        
        return all_results

    def compare_regularization_methods(self, train_loader, test_loader, epochs):
        """3.1 Сравнение методов регуляризации"""
        methods = [
            {"name": "no_reg", "label": "Без регуляризации"},
            {"name": "dropout_0.1", "label": "Dropout (p=0.1)", "dropout": 0.1},
            {"name": "dropout_0.3", "label": "Dropout (p=0.3)", "dropout": 0.3},
            {"name": "dropout_0.5", "label": "Dropout (p=0.5)", "dropout": 0.5},
            {"name": "batchnorm", "label": "Только BatchNorm", "batchnorm": True},
            {"name": "dropout_batchnorm", "label": "Dropout+BN", "dropout": 0.3, "batchnorm": True},
            {"name": "l2", "label": "L2 регуляризация", "weight_decay": 1e-3}
        ]
        
        results = {}
        for method in methods:
            print(f"\nTraining with {method['label']}...")
            model, history = self._train_model(
                train_loader, test_loader, epochs, 
                dropout=method.get("dropout", 0.0),
                batchnorm=method.get("batchnorm", False),
                weight_decay=method.get("weight_decay", 0.0)
            )
            
            # анализ весов
            weights = self._get_layer_weights(model)
            
            results[method["name"]] = {
                "label": method["label"],
                "history": history,
                "weights": {k: v.tolist() for k, v in weights.items()},  # конвертируем в list для JSON
                "final_test_acc": history["test_accs"][-1],
                "final_train_acc": history["train_accs"][-1],
                "params": count_parameters(model)
            }
            
            # визуализация
            self._save_regularization_results(model, history, weights, method["name"])
        
        return results
    
# 3.2 Реализуйте адаптивные методы:
# - Dropout с изменяющимся коэффициентом
# - BatchNorm с различными значениями импульса
# - Комбинирование нескольких методов
# - Анализ влияния на разные слои сети

    def run_adaptive_regularization(self, train_loader, test_loader, epochs):
        """3.2 Адаптивная регуляризация"""
        methods = [
            {"name": "adaptive_dropout", "label": "Адаптивный Dropout", 
             "dropout_schedule": lambda epoch: min(0.5, 0.1 + epoch * 0.03)},
            {"name": "batchnorm_momentum", "label": "BatchNorm (momentum=0.1)", 
             "batchnorm": True, "batchnorm_momentum": 0.1},
            {"name": "combined", "label": "Комбинированная", 
             "dropout": 0.3, "batchnorm": True, "weight_decay": 1e-4},
            {"name": "layerwise", "label": "Постепенная регуляризация", 
             "layerwise_reg": True}
        ]
        
        results = {}
        for method in methods:
            print(f"\nTraining with {method['label']}...")
            
            if method["name"] == "layerwise":
                model = self._create_layerwise_regularized_model()
            else:
                model = self._create_adaptive_model(method)
                
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), 
                                  weight_decay=method.get("weight_decay", 0.0))
            
            history = self._train_adaptive_model(
                model, train_loader, test_loader, 
                criterion, optimizer, epochs, method
            )
            
            weights = self._get_layer_weights(model)
            
            results[method["name"]] = {
                "label": method["label"],
                "history": history,
                "weights": {k: v.tolist() for k, v in weights.items()},
                "final_test_acc": history["test_accs"][-1],
                "final_train_acc": history["train_accs"][-1],
                "params": count_parameters(model)
            }
            
            self._save_regularization_results(model, history, weights, method["name"])
        
        return results

    def _train_model(self, train_loader, test_loader, epochs, 
                    dropout=0.0, batchnorm=False, weight_decay=0.0):
        """Обучение модели с заданными параметрами регуляризации"""
        config = self.base_config.copy()
        if dropout > 0 or batchnorm:
            config["layers"] = self._add_regularization_layers(config["layers"], dropout, batchnorm)
        
        model = create_model(config).to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), weight_decay=weight_decay)
        
        history = {
            'train_losses': [], 'test_losses': [],
            'train_accs': [], 'test_accs': []
        }
        
        for epoch in range(epochs):
            model.train()
            train_loss, train_acc = 0, 0
            for inputs, labels in train_loader:
                inputs = inputs.view(inputs.size(0), -1).to(self.device)
                labels = labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                _, predicted = torch.max(outputs.data, 1)
                train_acc += (predicted == labels).sum().item()
                train_loss += loss.item()
            
            test_loss, test_acc = self._evaluate(model, test_loader, criterion)
            
            history['train_losses'].append(train_loss/len(train_loader))
            history['test_losses'].append(test_loss)
            history['train_accs'].append(train_acc/len(train_loader.dataset))
            history['test_accs'].append(test_acc)
            
            print(f"Epoch {epoch+1}/{epochs} - Test Acc: {test_acc:.4f}")
        
        return model, history

    def _train_adaptive_model(self, model, train_loader, test_loader, 
                            criterion, optimizer, epochs, method_config):
        """Обучение модели с адаптивной регуляризацией"""
        history = {
            'train_losses': [], 'test_losses': [],
            'train_accs': [], 'test_accs': []
        }
        
        for epoch in range(epochs):
            # адаптация параметров регуляризации
            if "dropout_schedule" in method_config:
                p = method_config["dropout_schedule"](epoch)
                for module in model.modules():
                    if isinstance(module, nn.Dropout):
                        module.p = p
            
            model.train()
            train_loss, train_acc = 0, 0
            for inputs, labels in train_loader:
                inputs = inputs.view(inputs.size(0), -1).to(self.device)
                labels = labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                _, predicted = torch.max(outputs.data, 1)
                train_acc += (predicted == labels).sum().item()
                train_loss += loss.item()
            
            test_loss, test_acc = self._evaluate(model, test_loader, criterion)
            
            history['train_losses'].append(train_loss/len(train_loader))
            history['test_losses'].append(test_loss)
            history['train_accs'].append(train_acc/len(train_loader.dataset))
            history['test_accs'].append(test_acc)
            
            print(f"Epoch {epoch+1}/{epochs} - Test Acc: {test_acc:.4f}")
        
        return history

    def _create_adaptive_model(self, method_config):
        """Создание модели с адаптивной регуляризацией"""
        config = self.base_config.copy()
        config["layers"] = self._add_regularization_layers(
            config["layers"],
            dropout=method_config.get("dropout", 0.0),
            batchnorm=method_config.get("batchnorm", False),
            bn_momentum=method_config.get("batchnorm_momentum", 0.1)
        )
        return create_model(config).to(self.device)

    def _create_layerwise_regularized_model(self):
        """Модель с разной регуляризацией по слоям"""
        config = self.base_config.copy()
        config["layers"] = [
            {"type": "linear", "size": 256}, {"type": "relu"}, {"type": "dropout", "p": 0.1},
            {"type": "linear", "size": 128}, {"type": "relu"}, {"type": "batchnorm"}, {"type": "dropout", "p": 0.2},
            {"type": "linear", "size": 64}, {"type": "relu"}, {"type": "batchnorm"}, {"type": "dropout", "p": 0.3}
        ]
        return create_model(config).to(self.device)

    def _add_regularization_layers(self, layers, dropout, batchnorm, bn_momentum=0.1):
        """Добавляет слои регуляризации в архитектуру"""
        new_layers = []
        for layer in layers:
            new_layers.append(layer)
            if layer["type"] == "relu":
                if batchnorm:
                    new_layers.append({"type": "batchnorm", "momentum": bn_momentum})
                if dropout > 0:
                    new_layers.append({"type": "dropout", "p": dropout})
        return new_layers

    def _evaluate(self, model, test_loader, criterion):
        """Оценка модели на тестовых данных"""
        model.eval()
        test_loss, correct = 0, 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.view(inputs.size(0), -1).to(self.device)
                labels = labels.to(self.device)
                outputs = model(inputs)
                test_loss += criterion(outputs, labels).item()
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == labels).sum().item()
        return test_loss/len(test_loader), correct/len(test_loader.dataset)

    def _get_layer_weights(self, model):
        """Сбор весов всех линейных слоёв для анализа"""
        weights = {}
        for name, param in model.named_parameters():
            if 'weight' in name and 'bn' not in name:  # Исключаем BatchNorm веса
                weights[name] = param.data.cpu().numpy().flatten()
        return weights

    def _save_regularization_results(self, model, history, weights, method_name):
        """Графики и визуализации"""
        # кривые обучения
        plot_learning_curves(
            history, 
            title=f"{self.dataset} - {method_name}",
            save_path=f"plots/regularization_experiments/{self.dataset}_{method_name}_learning.png"
        )
        
        # распределение весов
        plot_weight_distributions(
            weights,
            title=f"{self.dataset} - {method_name}",
            save_path=f"plots/regularization_experiments/{self.dataset}_{method_name}_weights.png"
        )
        
        # сохранение модели
        torch.save(
            model.state_dict(),
            f"results/regularization_experiments/{self.dataset}_{method_name}.pth"
        )


if __name__ == "__main__":
    experiment = RegularizationExperiment(dataset='mnist')
    results = experiment.run(epochs=15)
    print("\nExperiments completed")
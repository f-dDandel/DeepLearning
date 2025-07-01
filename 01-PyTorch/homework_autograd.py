import torch

# 2.1 Создайте тензоры x, y, z с requires_grad=True
# Вычислите функцию: f(x,y,z) = x^2 + y^2 + z^2 + 2*x*y*z
# Найдите градиенты по всем переменным
# Проверьте результат аналитически
x = torch.rand(2,3, requires_grad=True)
y = torch.rand(2,3, requires_grad=True)
z = torch.rand(2,3, requires_grad=True)

f = x**2 + y**2 + z**2 + 2*x*y*z

f.backward(torch.ones_like(f)) #градиенты для всех элементов f

print(x.grad)
print(y.grad)
print(z.grad)

# аналитическая проверка (по формулам производных)
x_grad = 2*x + 2*y*z
y_grad = 2*y + 2*x*z
z_grad = 2*z + 2*x*y

print(x_grad, y_grad, z_grad)

# 2.2 Реализуйте функцию MSE (Mean Squared Error):
# MSE = (1/n) * Σ(y_pred - y_true)^2
# где y_pred = w * x + b (линейная функция)
# Найдите градиенты по w и b

# параметры модели
w = torch.tensor(1.0, requires_grad=True)
b = torch.tensor(0.0, requires_grad=True)

#случайные данные
x = torch.rand(4)
y_true = torch.rand(4)
y_pred = w * x + b

# MSE
loss = torch.mean((y_pred - y_true)**2)
#вычисляем градиент от loss, т.к. это уже скаляр
loss.backward()

print(w.grad.item(), b.grad.item())

# 2.3 Реализуйте составную функцию: f(x) = sin(x^2 + 1)
# Найдите градиент df/dx
# Проверьте результат с помощью torch.autograd.grad
x = torch.tensor(2.0, requires_grad=True)
func_sin = torch.sin(x**2 + 1)

# вычисляем градиент df/dx
func_sin.backward(retain_graph=True)  # сохраняем граф

#проверяем через torch.autograd.grad
manual_grad = torch.autograd.grad(func_sin, x)[0]

print(x.grad.item(), manual_grad.item())
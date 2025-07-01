import torch

# 1.1 Создайте следующие тензоры:
# - Тензор размером 3x4, заполненный случайными числами от 0 до 1
# - Тензор размером 2x3x4, заполненный нулями
# - Тензор размером 5x5, заполненный единицами
# - Тензор размером 4x4 с числами от 0 до 15 (используйте reshape)
rand_tensor = torch.rand(3,4)
zeros_tensor = torch.zeros(2,3,4)
ones_tensor = torch.ones(5,5)
arange_tensor = torch.arange(16).reshape(4,4)
print(arange_tensor)

#1.2 Дано: тензор A размером 3x4 и тензор B размером 4x3
# Выполните:
# - Транспонирование тензора A
# - Матричное умножение A и B
# - Поэлементное умножение A и транспонированного B
# - Вычислите сумму всех элементов тензора A
a = torch.rand(3,4)
b = torch.rand(4,3)
a_transp = a.T
mul_matrix = a @ b
mul_element = a * b.T
a_sum = a.sum()
print(mul_matrix, mul_element, a_sum)

# 1.3 Создайте тензор размером 5x5x5
# Извлеките:
# - Первую строку
# - Последний столбец
# - Подматрицу размером 2x2 из центра тензора
# - Все элементы с четными индексами
tensor_5x5x5 = torch.rand(5,5,5)
first_str = tensor_5x5x5[:,0, :] # первая строка из каждого слоя
last_col = tensor_5x5x5[:,:,4] #последний столбец из каждого слоя
center = tensor_5x5x5[1:3, 1:3, 1:3] #центральные индексы по всем осям
even = tensor_5x5x5[::2, ::2, ::2] #шаг 2 по всем осям
print(tensor_5x5x5, first_str, last_col, center, even)

# 1.4 Создайте тензор размером 24 элемента
# Преобразуйте его в формы:
# - 2x12
# - 3x8
# - 4x6
# - 2x3x4
# - 2x2x2x3
tensor24 = torch.arange(24)
print(tensor24)
tensor_2x12 = tensor24.reshape(2,12)
print(tensor_2x12)
tensor_3x8 = tensor24.reshape(3,8)
print(tensor_3x8)
tensor_4x6 = tensor24.reshape(4,6)
tensor_2x3x4 = tensor24.reshape(2,3,4)
tensor_2x2x2x3 = tensor24.reshape(2,2,2,3)
print(tensor_2x2x2x3)
import random
import numpy as np
import torch
import torchvision.transforms.functional as F
from PIL import Image, ImageFilter, ImageEnhance, ImageOps

class RandomBlur:
    """Случайное размытие изображения с разными радиусами"""
    def __init__(self, p=0.5, max_radius=3):
        self.p = p
        self.max_radius = max_radius
    
    def __call__(self, img):
        if random.random() < self.p:
            radius = random.uniform(0.1, self.max_radius)
            return img.filter(ImageFilter.GaussianBlur(radius))
        return img

class RandomPerspective:
    """Случайное перспективное искажение изображения"""
    def __init__(self, p=0.5, distortion_scale=0.5):
        self.p = p
        self.distortion_scale = distortion_scale
    
    def __call__(self, img):
        if random.random() < self.p:
            # Генерируем случайные точки для перспективы
            w, h = img.size
            startpoints = [(0, 0), (w, 0), (w, h), (0, h)]
            endpoints = [
                (random.randint(-int(w*0.1), int(w*0.1)), 
                random.randint(-int(h*0.1), int(h*0.1))),
                (w + random.randint(-int(w*0.1), int(w*0.1)), 
                random.randint(-int(h*0.1), int(h*0.1))),
                (w + random.randint(-int(w*0.1), int(w*0.1)), 
                h + random.randint(-int(h*0.1), int(h*0.1))),
                (random.randint(-int(w*0.1), int(w*0.1)), 
                h + random.randint(-int(h*0.1), int(h*0.1)))
            ]
            return F.perspective(img, startpoints, endpoints)
        return img

class RandomBrightnessContrast:
    """Случайное изменение яркости и контрастности"""
    def __init__(self, p=0.5, brightness_range=(0.7, 1.3), contrast_range=(0.7, 1.3)):
        self.p = p
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
    
    def __call__(self, img):
        if random.random() < self.p:
            # меняем яркость
            enhancer = ImageEnhance.Brightness(img)
            brightness_factor = random.uniform(*self.brightness_range)
            img = enhancer.enhance(brightness_factor)
            
            # меняем контраст
            enhancer = ImageEnhance.Contrast(img)
            contrast_factor = random.uniform(*self.contrast_range)
            img = enhancer.enhance(contrast_factor)
        return img
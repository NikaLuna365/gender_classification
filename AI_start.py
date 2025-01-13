import os
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import load_model

# Параметры
model_path = 'C:/Games/AI/gender_classification_model_20241215_172708.keras'
test_dir = 'C:/Games/AI/test'
target_size = (224, 224)  # Размер, к которому приводим все изображения

# Загрузка модели
model = load_model(model_path)

# Перебираем все изображения в папке test
for filename in os.listdir(test_dir):
    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        img_path = os.path.join(test_dir, filename)
        
        # Открываем изображение и приводим к нужному размеру
        img = Image.open(img_path).convert('RGB')
        img = img.resize(target_size)  # Изменяем размер до (224, 224)
        
        img_array = np.array(img)
        
        # Применяем ту же предобработку, что и при обучении
        img_array = preprocess_input(img_array)
        
        # Добавляем ось батча
        img_array = np.expand_dims(img_array, axis=0)
        
        # Предсказание
        pred = model.predict(img_array, verbose=0)
        
        # Интерпретация результата
        predicted_class = 1 if pred[0][0] > 0.5 else 0
        class_name = "female" if predicted_class == 1 else "male"
        
        # Вывод результата
        print(f"Изображение: {filename} => Предсказано: {class_name} (вероятность female={pred[0][0]:.4f})")

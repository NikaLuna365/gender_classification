import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from datetime import datetime  # для уникального имени файла

# === Параметры ===
data_dir = "C:/Games/AI/dataset"
image_size = (224, 224)
batch_size = 32
initial_epochs = 15
fine_tune_epochs = 15
total_epochs = initial_epochs + fine_tune_epochs
classes = ['male', 'female']

# === Проверяем наличие папок и количество изображений ===
def count_images_in_class(class_dir):
    return len([f for f in os.listdir(class_dir) if os.path.isfile(os.path.join(class_dir, f))])

for class_name in classes:
    class_dir = os.path.join(data_dir, class_name)
    if not os.path.exists(class_dir):
        raise FileNotFoundError(f"Папка '{class_name}' не найдена в: {data_dir}")
    num_images = count_images_in_class(class_dir)
    print(f"Количество изображений в классе '{class_name}': {num_images}")

# === Создаем генераторы данных ===
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    validation_split=0.2,
    horizontal_flip=True,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    brightness_range=[0.9, 1.1],
    zoom_range=0.1,
    shear_range=0.1,
    fill_mode='nearest'
)

validation_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    validation_split=0.2
)

train_generator = train_datagen.flow_from_directory(
    directory=data_dir,
    target_size=image_size,
    batch_size=batch_size,
    classes=classes,
    class_mode='binary',
    subset='training',
    shuffle=True,
    seed=42
)

validation_generator = validation_datagen.flow_from_directory(
    directory=data_dir,
    target_size=image_size,
    batch_size=batch_size,
    classes=classes,
    class_mode='binary',
    subset='validation',
    shuffle=False,
    seed=42
)

print(f"Сопоставление классов и меток: {train_generator.class_indices}")

# === Создаем модель на базе MobileNetV2 ===
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(image_size[0], image_size[1], 3))
base_model.trainable = False  # Сначала замораживаем

inputs = Input(shape=(image_size[0], image_size[1], 3))
# preprocess_input уже применен в генераторе, НЕ применяем здесь повторно
x = base_model(inputs, training=False)
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
x = Dropout(0.5)(x)
outputs = Dense(1, activation='sigmoid')(x)
model = Model(inputs, outputs)

# === Компиляция модели ===
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
model.compile(
    optimizer=optimizer,
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# === Коллбеки ===
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6)

# === Начальное обучение (головная часть) ===
history = model.fit(
    train_generator,
    epochs=initial_epochs,
    validation_data=validation_generator,
    callbacks=[early_stopping, reduce_lr]
)

# === Разморозка базы и Fine-tuning ===
base_model.trainable = True
fine_tune_at = 100
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

optimizer_fine = tf.keras.optimizers.Adam(learning_rate=1e-5)
model.compile(
    optimizer=optimizer_fine,
    loss='binary_crossentropy',
    metrics=['accuracy']
)

history_fine = model.fit(
    train_generator,
    epochs=fine_tune_epochs,
    validation_data=validation_generator,
    callbacks=[early_stopping, reduce_lr]
)

# === Объединяем историю обучения ===
acc = history.history['accuracy'] + history_fine.history['accuracy']
val_acc = history.history['val_accuracy'] + history_fine.history['val_accuracy']
loss = history.history['loss'] + history_fine.history['loss']
val_loss = history.history['val_loss'] + history_fine.history['val_loss']

epochs_range = range(len(acc))

# === Оценка модели ===
loss_val, accuracy_val = model.evaluate(validation_generator)
print(f"Validation Accuracy: {accuracy_val * 100:.2f}%")

# === Сохранение модели ===
# Генерируем уникальное имя с датой-временем
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
model_name = f"gender_classification_model_{timestamp}.keras"
model.save(model_name, save_format='keras')
print(f"Модель сохранена как '{model_name}'")

# === Графики ===
plt.figure(figsize=(8, 6))
plt.plot(epochs_range, acc, label='Точность на обучении')
plt.plot(epochs_range, val_acc, label='Точность на валидации')
plt.title('Точность модели')
plt.xlabel('Эпоха')
plt.ylabel('Точность')
plt.legend(loc='lower right')
plt.show()

plt.figure(figsize=(8, 6))
plt.plot(epochs_range, loss, label='Потери на обучении')
plt.plot(epochs_range, val_loss, label='Потери на валидации')
plt.title('Потери модели')
plt.xlabel('Эпоха')
plt.ylabel('Потери')
plt.legend(loc='upper right')
plt.show()

# === Анализ предсказаний ===
validation_generator.reset()
preds = model.predict(validation_generator)
predicted_classes = (preds > 0.5).astype('int32').flatten()

true_classes = validation_generator.classes
class_labels = list(validation_generator.class_indices.keys())

report = classification_report(true_classes, predicted_classes, target_names=class_labels)
print(report)

cm = confusion_matrix(true_classes, predicted_classes)
plt.figure(figsize=(6, 6))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_labels, yticklabels=class_labels, cmap='Blues')
plt.title('Матрица ошибок')
plt.xlabel('Предсказанные классы')
plt.ylabel('Настоящие классы')
plt.show()

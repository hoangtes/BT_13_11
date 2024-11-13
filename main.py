import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt

# Định nghĩa các thư mục chứa dữ liệu huấn luyện và xác nhận
train_dir = 'train'
validation_dir = 'validation'

# 1. Tiền xử lý dữ liệu
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

validation_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary'
)

validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary'
)

# 2. Xây dựng mô hình CNN
model = Sequential([
    Input(shape=(150, 150, 3)),  # Kích thước đầu vào của ảnh (150x150 pixels, 3 kênh màu RGB)

    # Lớp Convolutional đầu tiên với 32 bộ lọc
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),  # Max pooling với kích thước 2x2

    # Lớp Convolutional thứ hai với 64 bộ lọc
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),

    # Lớp Convolutional thứ ba với 128 bộ lọc
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),

    Flatten(),  # Chuyển đổi tensor 3D thành 1D để kết nối với các lớp Dense
    Dense(512, activation='relu'),  # Lớp Fully Connected với 512 nút
    Dropout(0.5),  # Dropout để giảm overfitting
    Dense(1, activation='sigmoid')  # Lớp đầu ra với 1 nút và hàm kích hoạt sigmoid cho phân loại nhị phân
])

# Biên dịch mô hình với optimizer Adam và hàm mất mát binary_crossentropy
model.compile(optimizer=Adam(learning_rate=0.001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# 3. Huấn luyện mô hình
history = model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),  # Đảm bảo steps_per_epoch là đúng
    epochs=20,
    validation_data=validation_generator,
    validation_steps=len(validation_generator)  # Đảm bảo validation_steps là đúng
)

# 4. Dự đoán ảnh trong thư mục validation
def predict_image_from_validation():
    for img_batch, labels in validation_generator:
        # Dự đoán cho batch ảnh đầu tiên
        predictions = model.predict(img_batch)
        for i, prediction in enumerate(predictions):
            # Kiểm tra xem mô hình dự đoán Mèo (1) hay Chó (0)
            if prediction > 0.5:
                print(f"Dự đoán: Mèo, Nhãn thực tế: {'Mèo' if labels[i] == 1 else 'Chó'}")
            else:
                print(f"Dự đoán: Chó, Nhãn thực tế: {'Chó' if labels[i] == 0 else 'Mèo'}")
        break  # Dự đoán chỉ cho batch đầu tiên

# Gọi hàm để dự đoán ảnh
predict_image_from_validation()

# 5. Đánh giá mô hình (vẽ biểu đồ độ chính xác và mất mát)
# Vẽ biểu đồ độ chính xác
plt.plot(history.history['accuracy'], label='Độ chính xác huấn luyện')
plt.plot(history.history['val_accuracy'], label='Độ chính xác xác nhận')
plt.xlabel('Epoch')
plt.ylabel('Độ chính xác')
plt.legend()
plt.title('Biểu đồ Độ chính xác')
plt.show()

# Vẽ biểu đồ mất mát
plt.plot(history.history['loss'], label='Mất mát huấn luyện')
plt.plot(history.history['val_loss'], label='Mất mát xác nhận')
plt.xlabel('Epoch')
plt.ylabel('Mất mát')
plt.legend()
plt.title('Biểu đồ Mất mát')
plt.show()

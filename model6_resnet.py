from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, BatchNormalization, Activation, Add, GlobalAveragePooling1D, Dense

def resnet_block(x, filters, kernel_size, stride=1):
    shortcut = x

    x = Conv1D(filters, kernel_size, padding='same', strides=stride)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv1D(filters, kernel_size, padding='same', strides=1)(x)
    x = BatchNormalization()(x)

    # Add shortcut connection
    x = Add()([shortcut, x])
    x = Activation('relu')(x)
    return x

def build_resnet1d(input_shape=(3000, 1), num_classes=1):
    inputs = Input(shape=input_shape)
    x = Conv1D(64, kernel_size=7, strides=2, padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # Residual blocks
    for _ in range(3):
        x = resnet_block(x, filters=64, kernel_size=3)

    x = GlobalAveragePooling1D()(x)
    outputs = Dense(num_classes, activation='sigmoid')(x)

    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


# Replace with your real ECG dataset
X_train = np.random.rand(100, 3000, 1)
y_train = np.random.randint(0, 2, 100)
X_val = np.random.rand(20, 3000, 1)
y_val = np.random.randint(0, 2, 20)

model = build_resnet1d(input_shape=(3000, 1))  # or build_inception_time
model.summary()

model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=32)


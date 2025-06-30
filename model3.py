import keras_tuner as kt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv1D, MaxPooling1D, BatchNormalization, Dropout,
    LSTM, Dense, Input, Activation
)
from tensorflow.keras.optimizers import Adam
import numpy as np

# ------------------------------
# Conv1D Block (VGG-style)
# ------------------------------
def conv1d_block(model, filters, convs=2, kernel_size=3):
    for _ in range(convs):
        model.add(Conv1D(filters=filters, kernel_size=kernel_size, activation='relu', padding='same'))
        model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2))
    return model

# ------------------------------
# LSTM Block
# ------------------------------
def lstm_block(model, units, return_seq=True, dropout=0.3):
    model.add(LSTM(units, return_sequences=return_seq))
    model.add(Dropout(dropout))
    return model

# ------------------------------
# Dense Block (1 layer per call)
# ------------------------------
def dense_block(model, units=64, dropout=0.3):
    model.add(Dense(units))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(dropout))
    return model

# ------------------------------
# Hyperparameter Build Function
# ------------------------------
def build_model(hp):
    model = Sequential()
    model.add(Input(shape=(3000, 1)))  # ECG input shape

    # 🔁 Number of Conv1D blocks
    num_blocks = hp.Choice('num_conv_blocks', values=[3, 4, 5])

    for i in range(num_blocks):
        filters = hp.Choice(f'filters_{i}', values=[32, 64, 128, 512])
        convs = hp.Choice(f'conv_layers_{i}', values=[2, 3, 4])
        kernel_size = hp.Choice(f'kernel_size_{i}', values=[3, 5, 7])
        conv1d_block(model, filters=filters, convs=convs, kernel_size=kernel_size)

    model.add(Dropout(0.4))

    # LSTM Blocks
    lstm_block(model, units=64, return_seq=True)
    lstm_block(model, units=64, return_seq=True)
    lstm_block(model, units=32, return_seq=False)

    # Dense Blocks (each one has 1 layer)
    dense_block(model, units=64, dropout=0.3)
    dense_block(model, units=64, dropout=0.3)
    dense_block(model, units=64, dropout=0.3)

    # Output Layer
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer=Adam(1e-3), loss='binary_crossentropy', metrics=['accuracy'])
    return model

# ------------------------------
# Run Tuner
# ------------------------------
if __name__ == "__main__":
    # Replace with actual data
    X_train = np.random.rand(100, 3000, 1)
    y_train = np.random.randint(0, 2, size=(100,))
    X_val = np.random.rand(20, 3000, 1)
    y_val = np.random.randint(0, 2, size=(20,))

    tuner = kt.RandomSearch(
        build_model,
        objective='val_accuracy',
        max_trials=10,  # Try 10 combinations
        executions_per_trial=1,
        directory='tuner_logs',
        project_name='ecg_tuning'
    )

    tuner.search(X_train, y_train,
                 validation_data=(X_val, y_val),
                 epochs=10,
                 batch_size=32)

    # Best Model + Hyperparams
    best_model = tuner.get_best_models(1)[0]
    best_hp = tuner.get_best_hyperparameters(1)[0]

    print("\n✅ Best Hyperparameters:")
    for key, val in best_hp.values.items():
        print(f"{key}: {val}")

    print("\n🧠 Best Model Summary:")
    best_model.summary()

import keras_tuner as kt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv1D, MaxPooling1D, BatchNormalization, Dropout,
    LSTM, Dense, Input, Activation
)
from tensorflow.keras.optimizers import Adam
import numpy as np

# Conv Block
def conv1d_block(model, filters, convs=2, kernel_size=3):
    for _ in range(convs):
        model.add(Conv1D(filters=filters, kernel_size=kernel_size, activation='relu', padding='same'))
        model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2))
    return model

# LSTM Block
def lstm_block(model, units, return_seq=True, dropout=0.3):
    model.add(LSTM(units, return_sequences=return_seq))
    model.add(Dropout(dropout))
    return model

# Dense Block
def dense_block(model, layers=2, units=64, dropout=0.3):
    for _ in range(layers):
        model.add(Dense(units))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
    model.add(Dropout(dropout))
    return model

# Build the model
def build_model(hp):
    model = Sequential()
    model.add(Input(shape=(3000, 1)))

    # 🔧 Conv1D Blocks
    num_conv_blocks = hp.Choice('num_conv_blocks', values=[3, 4, 5])
    for i in range(num_conv_blocks):
        filters = hp.Choice(f'filters_{i}', values=[32, 64, 128, 512])
        convs = hp.Choice(f'conv_layers_{i}', values=[2, 3, 4])
        kernel_size = hp.Choice(f'kernel_size_{i}', values=[3, 5, 7])
        conv1d_block(model, filters=filters, convs=convs, kernel_size=kernel_size)

    model.add(Dropout(0.4))

    # 🔧 LSTM Blocks
    num_lstm_blocks = hp.Choice('num_lstm_blocks', values=[2, 3, 4])
    for j in range(num_lstm_blocks):
        units = hp.Choice(f'lstm_units_{j}', values=[32, 64, 128, 256])
        return_seq = True if j < num_lstm_blocks - 1 else False
        lstm_block(model, units=units, return_seq=return_seq)

    # 🔧 Dense Blocks
    num_dense_blocks = hp.Choice('num_dense_blocks', values=[1, 2, 3])
    for k in range(num_dense_blocks):
        units = hp.Choice(f'dense_units_{k}', values=[32, 64, 128, 256])
        layers = hp.Choice(f'dense_layers_{k}', values=[2, 3, 4])
        dropout = hp.Choice(f'dense_layers_dropout_ratio_{k}', values=[0.3, 0.4, 0.5])
        dense_block(model, layers=layers, units=units, dropout=dropout)

    # Output
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer=Adam(1e-3), loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Tuning Execution
if __name__ == "__main__":
    # Dummy ECG-like data
    X_train = np.random.rand(100, 3000, 1)
    y_train = np.random.randint(0, 2, 100)
    X_val = np.random.rand(20, 3000, 1)
    y_val = np.random.randint(0, 2, 20)

    tuner = kt.RandomSearch(
        build_model,
        objective='val_accuracy',
        max_trials=10,
        executions_per_trial=1,
        directory='tuner_logs',
        project_name='ecg_full_blockwise_tuning'
    )

    tuner.search(X_train, y_train,
                 validation_data=(X_val, y_val),
                 epochs=10,
                 batch_size=32)

    best_model = tuner.get_best_models(1)[0]
    best_hp = tuner.get_best_hyperparameters(1)[0]

    print("\n✅ Best Hyperparameters:")
    for key, val in best_hp.values.items():
        print(f"{key}: {val}")

    print("\n🧠 Best Model Summary:")
    best_model.summary()

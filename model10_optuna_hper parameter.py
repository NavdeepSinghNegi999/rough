import optuna
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv1D, MaxPooling1D, BatchNormalization, Dropout,
    LSTM, Dense, Input, Activation
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------
# Block Definitions
# ---------------------------------------------------------

def conv1d_block(model, filters, convs=2, kernel_size=3):
    for _ in range(convs):
        model.add(Conv1D(filters=filters, kernel_size=kernel_size, activation='relu', padding='same'))
        model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2))
    return model

def lstm_block(model, units, return_seq=True, dropout=0.3):
    model.add(LSTM(units, return_sequences=return_seq))
    model.add(Dropout(dropout))
    return model

def dense_block(model, layers=2, units=64, dropout=0.3):
    for _ in range(layers):
        model.add(Dense(units))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
    model.add(Dropout(dropout))
    return model

# ---------------------------------------------------------
# Build Model Function for Optuna
# ---------------------------------------------------------

def build_model(trial):
    model = Sequential()
    model.add(Input(shape=(3000, 1)))

    # Conv1D Blocks
    num_conv_blocks = trial.suggest_int('num_conv_blocks', 3, 5)
    for i in range(num_conv_blocks):
        filters = trial.suggest_categorical(f'filters_{i}', [32, 64, 128, 512])
        convs = trial.suggest_int(f'conv_layers_{i}', 2, 4)
        kernel_size = trial.suggest_categorical(f'kernel_size_{i}', [3, 5, 7])
        conv1d_block(model, filters, convs, kernel_size)

    model.add(Dropout(0.4))

    # LSTM Blocks
    num_lstm_blocks = trial.suggest_int('num_lstm_blocks', 2, 4)
    for j in range(num_lstm_blocks):
        units = trial.suggest_categorical(f'lstm_units_{j}', [32, 64, 128, 256])
        return_seq = j < num_lstm_blocks - 1
        lstm_block(model, units=units, return_seq=return_seq)

    # Dense Blocks
    num_dense_blocks = trial.suggest_int('num_dense_blocks', 1, 3)
    for k in range(num_dense_blocks):
        units = trial.suggest_categorical(f'dense_units_{k}', [32, 64, 128, 256])
        layers = trial.suggest_int(f'dense_layers_{k}', 2, 4)
        dropout = trial.suggest_float(f'dense_dropout_{k}', 0.3, 0.5)
        dense_block(model, layers=layers, units=units, dropout=dropout)

    model.add(Dense(1, activation='sigmoid'))

    # Optimizer
    lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
    model.compile(optimizer=Adam(learning_rate=lr), loss='binary_crossentropy', metrics=['accuracy'])
    return model

# ---------------------------------------------------------
# Objective Function
# ---------------------------------------------------------

def objective(trial):
    model = build_model(trial)
    early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True, verbose=0)

    history = model.fit(X_train, y_train,
                        validation_data=(X_val, y_val),
                        epochs=20,
                        batch_size=32,
                        verbose=0,
                        callbacks=[early_stop])
    
    return max(history.history['val_accuracy'])

# ---------------------------------------------------------
# Data Setup
# ---------------------------------------------------------

X = np.random.rand(120, 3000, 1)
y = np.random.randint(0, 2, 120)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)

# ---------------------------------------------------------
# Optuna Search
# ---------------------------------------------------------

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=20)

# ---------------------------------------------------------
# Show Best Result
# ---------------------------------------------------------

print("\n✅ Best Hyperparameters:")
for key, val in study.best_params.items():
    print(f"{key}: {val}")

print(f"\n🎯 Best Validation Accuracy: {study.best_value:.4f}")

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv1D, MaxPooling1D, BatchNormalization, Dropout,
    LSTM, Dense, Input, Activation
)

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
# Full Model Function
# ------------------------------
def create_ecg_model_with_3_lstm_blocks(input_shape=(3000, 1)):
    model = Sequential()
    model.add(Input(shape=input_shape))

    # Conv1D Blocks (VGG-style)
    conv1d_block(model, filters=32, convs=2)
    conv1d_block(model, filters=64, convs=2)
    conv1d_block(model, filters=128, convs=3)
    model.add(Dropout(0.4))

    # LSTM Blocks
    lstm_block(model, units=64, return_seq=True)
    lstm_block(model, units=64, return_seq=True)
    lstm_block(model, units=32, return_seq=False)

    # Dense Blocks (each one has 1 layer)
    dense_block(model, units=64, dropout=0.3)
    dense_block(model, units=64, dropout=0.3)
    dense_block(model, units=64, dropout=0.3)

    # Output Layer (Binary Classification)
    model.add(Dense(1, activation='sigmoid'))

    # Compile Model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return model

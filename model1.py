from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, BatchNormalization, Dropout, LSTM, Dense, Input

# Define Conv1D Block
def conv1d_block(model, filters, convs=2, kernel_size=3):
    for _ in range(convs):
        model.add(Conv1D(filters=filters, kernel_size=kernel_size, activation='relu', padding='same'))
        model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2))
    return model

# Define LSTM Block
def lstm_block(model, lstm_units=[64, 32], dropout=0.3):
    for i, units in enumerate(lstm_units):
        return_seq = i < len(lstm_units) - 1  # True for all but last layer
        model.add(LSTM(units=units, return_sequences=return_seq))
        model.add(Dropout(dropout))
    return model

# Create Full Model
def create_vgg1d_lstm_block_model(input_shape=(3000, 1)):
    model = Sequential()
    model.add(Input(shape=input_shape))

    # VGG-style Conv1D Blocks
    conv1d_block(model, filters=32, convs=2)   # Block 1
    conv1d_block(model, filters=64, convs=2)   # Block 2
    conv1d_block(model, filters=128, convs=3)  # Block 3

    model.add(Dropout(0.4))

    # LSTM Block
    lstm_block(model, lstm_units=[64, 32])  

    # Dense Layers
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(1, activation='sigmoid'))  

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Dense, Dropout, LayerNormalization,
    MultiHeadAttention, GlobalAveragePooling1D
)
from tensorflow.keras.models import Model


def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0.1):
    # Self-attention
    x = MultiHeadAttention(num_heads=num_heads, key_dim=head_size, dropout=dropout)(inputs, inputs)
    x = Dropout(dropout)(x)
    x = LayerNormalization(epsilon=1e-6)(x + inputs)

    # Feed-forward
    x_ff = Dense(ff_dim, activation='relu')(x)
    x_ff = Dense(inputs.shape[-1])(x_ff)
    x_ff = Dropout(dropout)(x_ff)

    x_out = LayerNormalization(epsilon=1e-6)(x + x_ff)
    return x_out


def build_ecg_transformer(input_shape=(4096, 1), num_classes=1):
    inputs = Input(shape=input_shape)

    x = transformer_encoder(inputs, head_size=64, num_heads=4, ff_dim=128, dropout=0.1)
    x = transformer_encoder(x, head_size=64, num_heads=4, ff_dim=128, dropout=0.1)
    x = transformer_encoder(x, head_size=64, num_heads=4, ff_dim=128, dropout=0.1)

    x = GlobalAveragePooling1D()(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.3)(x)
    outputs = Dense(num_classes, activation='sigmoid')(x)

    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


import numpy as np

# Simulate ECG data
X_train = np.random.rand(100, 4096, 1)   # (samples, length, 1)
y_train = np.random.randint(0, 2, 100)

X_val = np.random.rand(20, 4096, 1)
y_val = np.random.randint(0, 2, 20)

model = build_ecg_transformer(input_shape=(4096, 1))
model.summary()

model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=16)

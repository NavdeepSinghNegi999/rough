import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Dense, Dropout, LayerNormalization,
    MultiHeadAttention, GlobalAveragePooling1D
)
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
import numpy as np
import datetime
import matplotlib.pyplot as plt

# Transformer Encoder with optional return of attention scores
def transformer_encoder_with_attention(inputs, head_size, num_heads, ff_dim, dropout=0.1):
    mha = MultiHeadAttention(num_heads=num_heads, key_dim=head_size, dropout=dropout)
    attn_output, attn_weights = mha(inputs, inputs, return_attention_scores=True)
    x = Dropout(dropout)(attn_output)
    x = LayerNormalization(epsilon=1e-6)(x + inputs)

    # Feed-forward network
    x_ff = Dense(ff_dim, activation='relu')(x)
    x_ff = Dense(inputs.shape[-1])(x_ff)
    x_ff = Dropout(dropout)(x_ff)

    x_out = LayerNormalization(epsilon=1e-6)(x + x_ff)
    return x_out, attn_weights

# Build main model and attention extractor model
def build_ecg_transformer(input_shape=(4096, 1), num_classes=1):
    inputs = Input(shape=input_shape)

    x1, attn1 = transformer_encoder_with_attention(inputs, head_size=64, num_heads=4, ff_dim=128)
    x2, attn2 = transformer_encoder_with_attention(x1, head_size=64, num_heads=4, ff_dim=128)
    x3, attn3 = transformer_encoder_with_attention(x2, head_size=64, num_heads=4, ff_dim=128)

    x = GlobalAveragePooling1D()(x3)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.3)(x)
    outputs = Dense(num_classes, activation='sigmoid')(x)

    model = Model(inputs, outputs)
    attention_model = Model(inputs, attn1)  # Return first layer's attention weights

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model, attention_model

# Generate dummy ECG data
X_train = np.random.rand(100, 4096, 1)
y_train = np.random.randint(0, 2, 100)

X_val = np.random.rand(20, 4096, 1)
y_val = np.random.randint(0, 2, 20)

# Callbacks
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
log_dir = "logs/ecg_transformer/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

# Build model
model, attn_model = build_ecg_transformer(input_shape=(4096, 1))
model.summary()

# Train
model.fit(X_train, y_train,
          validation_data=(X_val, y_val),
          epochs=50,
          batch_size=16,
          callbacks=[early_stop, tensorboard_callback])

# Visualize attention weights
sample = np.expand_dims(X_val[0], axis=0)  # shape: (1, 4096, 1)
attn_weights = attn_model.predict(sample)[0]  # (4096, 4096)

# Mean attention across all timesteps
attention_map = np.mean(attn_weights, axis=0)

# Plot
plt.figure(figsize=(15, 4))
plt.plot(attention_map)
plt.title("Attention Map for Sample ECG")
plt.xlabel("Time Step")
plt.ylabel("Attention")
plt.grid(True)
plt.tight_layout()
plt.show()

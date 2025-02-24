import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    MaxPooling2D,
    UpSampling2D,
    Flatten,
    Dense,
    Reshape,
)
from tensorflow.keras.models import Model


def build_autoencoder(input_shape=(224, 224, 3), latent_dim=128):
    # Entrada
    input_img = Input(shape=input_shape)

    # -----------------
    #    ENCODER
    # -----------------
    x = Conv2D(32, (3, 3), activation="relu", padding="same")(input_img)
    x = MaxPooling2D((2, 2), padding="same")(x)  # 112x112x32

    x = Conv2D(64, (3, 3), activation="relu", padding="same")(x)
    x = MaxPooling2D((2, 2), padding="same")(x)  # 56x56x64

    x = Conv2D(128, (3, 3), activation="relu", padding="same")(x)
    x = MaxPooling2D((2, 2), padding="same")(x)  # 28x28x128

    # Guardar la forma para el decoder
    shape_before_flattening = x.shape[1:]
    x = Flatten()(x)
    latent = Dense(latent_dim, name="latent_vector")(x)  # Representación latente

    # -----------------
    #    DECODER
    # -----------------
    x = Dense(np.prod(shape_before_flattening), activation="relu")(latent)
    x = Reshape(shape_before_flattening)(x)

    x = UpSampling2D((2, 2))(x)  # 56x56x128
    x = Conv2D(128, (3, 3), activation="relu", padding="same")(x)

    x = UpSampling2D((2, 2))(x)  # 112x112x128
    x = Conv2D(64, (3, 3), activation="relu", padding="same")(x)

    x = UpSampling2D((2, 2))(x)  # 224x224x64
    decoded = Conv2D(3, (3, 3), activation="sigmoid", padding="same")(x)

    # Modelo autoencoder completo
    autoencoder = Model(input_img, decoded)

    # Modelo encoder: desde la entrada hasta la representación latente
    encoder = Model(input_img, latent)

    return autoencoder, encoder


def get_autoencoder(x_train, validation):
    # Construir el autoencoder
    autoencoder, encoder = build_autoencoder(input_shape=(224, 224, 3), latent_dim=128)

    # Compilar el autoencoder
    autoencoder.compile(optimizer="adam", loss="mse")

    # Entrenar el autoencoder
    autoencoder.fit(x_train, epochs=50, validation_data=validation)

    return autoencoder

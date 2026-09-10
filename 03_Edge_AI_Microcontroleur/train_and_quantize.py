"""
Entraîne un modèle de classification d'images léger (CNN) sur le jeu de
données MNIST, puis le convertit et le quantifie en TensorFlow Lite
(int8) pour un déploiement embarqué sur microcontrôleur (ESP32).

NB : ce script nécessite tensorflow (non installé dans cet environnement
de génération). Il est fourni prêt à l'exécution sur une machine disposant
de TensorFlow/Keras — c'est le livrable technique du projet.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

IMG_SIZE = 28
NUM_CLASSES = 10


def build_tiny_cnn():
    """CNN volontairement très léger pour tenir sur un microcontrôleur
    (quelques dizaines de Ko une fois quantifié en int8)."""
    model = models.Sequential([
        layers.Input(shape=(IMG_SIZE, IMG_SIZE, 1)),
        layers.Conv2D(8, 3, activation="relu", padding="same"),
        layers.MaxPooling2D(2),
        layers.Conv2D(16, 3, activation="relu", padding="same"),
        layers.MaxPooling2D(2),
        layers.Flatten(),
        layers.Dense(32, activation="relu"),
        layers.Dense(NUM_CLASSES, activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def representative_dataset_gen(x_train, num_samples=200):
    """Nécessaire pour la quantification full-integer : fournit des
    exemples représentatifs des données d'entrée réelles."""
    for i in range(num_samples):
        sample = x_train[i:i + 1].astype(np.float32)
        yield [sample]


def main():
    print("Chargement de MNIST...")
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = (x_train / 255.0).astype(np.float32)[..., np.newaxis]
    x_test = (x_test / 255.0).astype(np.float32)[..., np.newaxis]

    print("Entraînement du CNN léger...")
    model = build_tiny_cnn()
    model.summary()
    model.fit(x_train, y_train, epochs=5, batch_size=64, validation_split=0.1)

    test_loss, test_acc = model.evaluate(x_test, y_test)
    print(f"Précision sur le jeu de test (modèle float32) : {test_acc:.3f}")

    model.save("modele_cnn_leger.h5")

    print("Conversion et quantification int8 pour l'embarqué (TensorFlow Lite)...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = lambda: representative_dataset_gen(x_train)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    tflite_quant_model = converter.convert()
    with open("modele_quantifie.tflite", "wb") as f:
        f.write(tflite_quant_model)

    import os
    taille_ko = os.path.getsize("modele_quantifie.tflite") / 1024
    print(f"Modèle quantifié sauvegardé : modele_quantifie.tflite ({taille_ko:.1f} Ko)")
    print("Ce fichier .tflite est ensuite converti en tableau C (xxd -i) "
          "pour être embarqué dans le firmware ESP32 via TensorFlow Lite for Microcontrollers.")


if __name__ == "__main__":
    main()

#  Projet 3 — Reconnaissance d'objets embarquée sur microcontrôleur (Edge AI)

## Objectif
Déployer un modèle de classification d'images allégé directement sur un microcontrôleur, pour une inférence locale en temps réel, sans connexion cloud.

## Fonctionnalités
- Entraînement d'un CNN volontairement très léger (peu de couches/filtres)
- Conversion et quantification int8 pour l'embarqué (TensorFlow Lite)
- Inférence locale en temps réel sur ESP32 (TensorFlow Lite for Microcontrollers)

## Stack technique
TensorFlow/Keras · TensorFlow Lite · microcontrôleur ESP32

## Concepts démontrés
Edge computing, quantification de modèles, contraintes de ressources embarquées (mémoire, puissance de calcul, énergie).

## Structure du projet
```
03_Edge_AI_Microcontroleur/
├── train_and_quantize.py       # entraînement CNN + conversion .tflite quantifiée int8
└── firmware_esp32/
    └── inference_esp32.ino     # sketch Arduino pour l'inférence embarquée sur ESP32
```

## Comment lancer le projet
1. **Entraînement et quantification (sur une machine avec TensorFlow) :**
   ```bash
   pip install tensorflow
   python train_and_quantize.py
   ```
   Génère `modele_quantifie.tflite` (quelques dizaines de Ko).

2. **Génération du header C pour l'ESP32 :**
   ```bash
   xxd -i modele_quantifie.tflite > firmware_esp32/model_data.h
   ```

3. **Flash sur l'ESP32 :**
   Ouvrir `firmware_esp32/inference_esp32.ino` dans l'IDE Arduino (bibliothèque `Arduino_TensorFlowLite`), sélectionner la carte ESP32, compiler et téléverser.

## Pourquoi la quantification int8 ?
Un modèle en float32 pèse généralement 4x plus lourd et est plus lent à exécuter sur un microcontrôleur qui ne dispose ni de GPU, ni de beaucoup de RAM (souvent < 520 Ko sur un ESP32). La quantification int8 réduit la taille du modèle et accélère l'inférence, au prix d'une légère perte de précision, généralement négligeable pour des tâches de classification simples.

## Ligne CV
« Edge AI embarquée — TensorFlow Lite, ESP32, quantification de modèle. »

## Présentation GitHub
Démonstration vidéo de l'inférence embarquée (à ajouter), README expliquant les contraintes de quantification.

## Question d'entretien possible
Quelles techniques permettent de réduire la taille d'un modèle pour le déployer sur un microcontrôleur ?

*(Réponse : quantification (float32 → int8), pruning (suppression des connexions/poids peu significatifs), distillation de connaissance (un petit modèle apprend à imiter un grand modèle), et conception d'architectures nativement compactes comme MobileNet ou les CNN "tiny" utilisés ici.)*

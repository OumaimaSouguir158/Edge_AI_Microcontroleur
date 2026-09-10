/*
  Inférence embarquée sur ESP32 avec TensorFlow Lite for Microcontrollers.

  Ce sketch charge le modèle quantifié (converti en tableau C, voir
  model_data.h généré à partir de modele_quantifie.tflite via :
      xxd -i modele_quantifie.tflite > model_data.h
  ) et effectue une inférence locale, sans connexion réseau, sur une
  image 28x28 reçue (ex. capteur de vision ou image de test envoyée en
  série).

  Bibliothèque requise : Arduino_TensorFlowLite (ou tflite-micro-arduino)
*/

#include <TensorFlowLite_ESP32.h>
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "model_data.h"  // contient le tableau g_model[] généré par xxd

namespace {
tflite::ErrorReporter* error_reporter = nullptr;
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

constexpr int kTensorArenaSize = 60 * 1024;  // arène mémoire pour le graphe
uint8_t tensor_arena[kTensorArenaSize];
}

void setup() {
  Serial.begin(115200);

  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;

  model = tflite::GetModel(g_model);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    error_reporter->Report("Version de schéma du modèle incompatible.");
    return;
  }

  static tflite::AllOpsResolver resolver;
  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, kTensorArenaSize, error_reporter);
  interpreter = &static_interpreter;

  interpreter->AllocateTensors();
  input = interpreter->input(0);
  output = interpreter->output(0);

  Serial.println("Modèle chargé. Prêt pour l'inférence locale (offline).");
}

void loop() {
  // 1. Récupérer une image 28x28 (ex. capteur caméra, ou test via Serial)
  //    et la copier quantifiée (int8) dans input->data.int8

  // 2. Lancer l'inférence
  TfLiteStatus invoke_status = interpreter->Invoke();
  if (invoke_status != kTfLiteOk) {
    Serial.println("Échec de l'inférence.");
    return;
  }

  // 3. Lire la classe prédite (argmax sur les 10 sorties)
  int predicted_class = 0;
  int8_t best_score = output->data.int8[0];
  for (int i = 1; i < 10; i++) {
    if (output->data.int8[i] > best_score) {
      best_score = output->data.int8[i];
      predicted_class = i;
    }
  }

  Serial.print("Classe prédite : ");
  Serial.println(predicted_class);

  delay(1000);
}

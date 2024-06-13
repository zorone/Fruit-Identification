
#include <TensorFlowLite.h>

#include <tensorflow/lite/micro/all_ops_resolver.h>
#include <tensorflow/lite/micro/micro_error_reporter.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/schema/schema_generated.h>
#include <tensorflow/lite/version.h>
#include <Arduino_APDS9960.h>
#include "model.h"

#include <Wire.h>
#include <LiquidCrystal_I2C.h>
LiquidCrystal_I2C lcd(0x27, 16, 2);

// global variables used for TensorFlow Lite (Micro)
tflite::MicroErrorReporter tflErrorReporter;

// pull in all the TFLM ops, you can remove this line and
// only pull in the TFLM ops you need, if would like to reduce
// the compiled size of the sketch.
tflite::AllOpsResolver tflOpsResolver;

const tflite::Model* tflModel = nullptr;
tflite::MicroInterpreter* tflInterpreter = nullptr;
TfLiteTensor* tflInputTensor = nullptr;
TfLiteTensor* tflOutputTensor = nullptr;

// Create a static memory buffer for TFLM, the size may need to
// be adjusted based on the model you are using
constexpr int tensorArenaSize = 8 * 1024;
byte tensorArena[tensorArenaSize];

// array to map gesture index to a name
const char* CLASSES[] = {
  "Apple",
  "Banana", 
  "Green Apple", 
  "Kiwi",
  "Mangosteen",
  "Orange",
  "Trolley",
};

#define NUM_CLASSES (sizeof(CLASSES) / sizeof(CLASSES[0]))

void setup() {
  lcd.init();         
  lcd.backlight();
  lcd.print("Color Clssify");
  lcd.setCursor(0, 1);
  
  //Serial.begin(9600);
  //pinMode(2, OUTPUT);
  //digitalWrite(2, HIGH);
  //while (!Serial) {};

  //Serial.println("Object classification using RGB color sensor");
  //Serial.println("--------------------------------------------");
  //Serial.println("Arduino Nano 33 BLE Sense running TensorFlow Lite Micro");
  //Serial.println("");

  if (!APDS.begin()) {
    lcd.println("Error initializing APDS9960 sensor.");
  }

  // get the TFL representation of the model byte array
  tflModel = tflite::GetModel(model);
  if (tflModel->version() != TFLITE_SCHEMA_VERSION) {
    lcd.println("Model schema mismatch!");
    while (1);
  }

  // Create an interpreter to run the model
  tflInterpreter = new tflite::MicroInterpreter(tflModel, tflOpsResolver, tensorArena, tensorArenaSize, &tflErrorReporter);

  // Allocate memory for the model's input and output tensors
  tflInterpreter->AllocateTensors();

  // Get pointers for the model's input and output tensors
  tflInputTensor = tflInterpreter->input(0);
  tflOutputTensor = tflInterpreter->output(0);
}

void loop() {
  int r, g, b, p, c;
  float sum;

  // check if both color and proximity data is available to sample
  while (!APDS.colorAvailable() || !APDS.proximityAvailable()) {}

  // read the color and proximity sensor
  APDS.readColor(r, g, b, c);
  p = APDS.readProximity();
  sum = r + g + b;

  // check if there's an object close and well illuminated enough
  if (p == 0 && c > 10 && sum > 0) {

    float redRatio = r / sum;
    float greenRatio = g / sum;
    float blueRatio = b / sum;

    // input sensor data to model
    tflInputTensor->data.f[0] = redRatio;
    tflInputTensor->data.f[1] = greenRatio;
    tflInputTensor->data.f[2] = blueRatio;

    // Run inferencing
    TfLiteStatus invokeStatus = tflInterpreter->Invoke();
    if (invokeStatus != kTfLiteOk) {
      lcd.println("Invoke failed!");
      while (1);
      return;
    }

    // Output results
    float max = -1;
    int maxi = -1;
    
    for (int i = 0; i < NUM_CLASSES; i++) {
      if(tflOutputTensor->data.f[i] > max){
        max = tflOutputTensor->data.f[i];
        maxi = i;
      }
    }
    
    lcd.clear();
    lcd.print(CLASSES[maxi]);
    lcd.print(" ");
    lcd.print(int(tflOutputTensor->data.f[maxi] * 100));
    lcd.print("%");
    lcd.setCursor(0, 1);

    // Wait for the object to be moved away
    while (!APDS.proximityAvailable() || (APDS.readProximity() == 0)) {}
  }

}
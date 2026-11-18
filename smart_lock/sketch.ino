
#include <WiFi.h>
#include <PubSubClient.h>
#include <ESP32Servo.h>

// ---------- Wi-Fi ----------
const char* WIFI_SSID = "Wokwi-GUEST";
const char* WIFI_PASSWORD = "";

// ---------- MQTT ----------
const char* MQTT_BROKER = "broker.hivemq.com";
const int MQTT_PORT = 1883;
const char* MQTT_TOPIC = "aryan/p300eeg/smartlock/control";

// ---------- Servo ----------
const int SERVO_PIN = 18;
const int ANGLE_LOCKED = 0;
const int ANGLE_UNLOCKED = 90;

enum LockState {
  LOCKED,
  UNLOCKED
};

Servo lockServo;
LockState lockState = LOCKED;

// ---------- MQTT Client ----------
WiFiClient espClient;
PubSubClient mqttClient(espClient);

// ---------- Serial input ----------
String inputLine = "";

const char* stateName(LockState state) {
  return state == LOCKED ? "LOCKED" : "UNLOCKED";
}

// ---------- Lock control ----------
void setLock(LockState target) {
  lockServo.write(target == LOCKED ? ANGLE_LOCKED : ANGLE_UNLOCKED);
  lockState = target;

  Serial.print("[LOCK] Door is now ");
  Serial.print(stateName(lockState));
  Serial.print(" (servo ");
  Serial.print(target == LOCKED ? ANGLE_LOCKED : ANGLE_UNLOCKED);
  Serial.println(" deg)");
}

void printStatus() {
  Serial.print("[STATUS] Door is ");
  Serial.println(stateName(lockState));
}

// ---------- MQTT message handler ----------
void mqttCallback(char* topic, byte* payload, unsigned int length) {
  String message = "";

  for (unsigned int i = 0; i < length; i++) {
    message += (char)payload[i];
  }

  message.trim();
  message.toUpperCase();

  Serial.print("[MQTT] Received on ");
  Serial.print(topic);
  Serial.print(": ");
  Serial.println(message);

  if (message == "ACCEPT" || message == "UNLOCK") {
    setLock(UNLOCKED);
  }
  else if (message == "REJECT" || message == "LOCK") {
    setLock(LOCKED);
  }
  else if (message == "STATUS") {
    printStatus();
  }
  else {
    Serial.println("[MQTT] Unknown command");
  }
}

// ---------- Wi-Fi ----------
void connectWiFi() {
  Serial.println("[WiFi] Connecting to Wokwi-GUEST...");

  WiFi.begin(WIFI_SSID, WIFI_PASSWORD, 6);

  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }

  Serial.println();
  Serial.println("[WiFi] Connected!");
  Serial.print("[WiFi] IP: ");
  Serial.println(WiFi.localIP());
}

// ---------- MQTT ----------
void reconnectMQTT() {
  while (!mqttClient.connected()) {
    Serial.print("[MQTT] Connecting to broker...");

    String clientId = "P300EEG-ESP32-";
    clientId += String(random(0xffff), HEX);

    if (mqttClient.connect(clientId.c_str())) {
      Serial.println(" connected!");

      if (mqttClient.subscribe(MQTT_TOPIC)) {
        Serial.print("[MQTT] Subscribed to: ");
        Serial.println(MQTT_TOPIC);
      }
      else {
        Serial.println("[MQTT] Subscription failed");
      }
    }
    else {
      Serial.print("[MQTT] Failed, state=");
      Serial.print(mqttClient.state());
      Serial.println(" | Retrying in 3 seconds...");
      delay(3000);
    }
  }
}

// ---------- Serial commands ----------
void handleCommand(String command) {
  command.trim();
  command.toUpperCase();

  if (command.length() == 0) return;

  if (command == "LOCK") {
    setLock(LOCKED);
  }
  else if (command == "UNLOCK") {
    setLock(UNLOCKED);
  }
  else if (command == "STATUS") {
    printStatus();
  }
  else if (command == "HELP") {
    Serial.println("Commands: LOCK | UNLOCK | STATUS | HELP");
  }
  else {
    Serial.print("[ERROR] Unknown command: ");
    Serial.println(command);
  }
}

// ---------- Setup ----------
void setup() {
  Serial.begin(115200);
  delay(500);

  lockServo.setPeriodHertz(50);
  lockServo.attach(SERVO_PIN, 500, 2400);

  Serial.println();
  Serial.println("=== EEG Smart Lock - MQTT ===");

  setLock(LOCKED);

  connectWiFi();

  mqttClient.setServer(MQTT_BROKER, MQTT_PORT);
  mqttClient.setCallback(mqttCallback);

  Serial.println("[SYSTEM] Setup complete");
  Serial.println("[SYSTEM] Serial: LOCK | UNLOCK | STATUS | HELP");
}

// ---------- Main loop ----------
void loop() {
  if (WiFi.status() != WL_CONNECTED) {
    connectWiFi();
  }

  if (!mqttClient.connected()) {
    reconnectMQTT();
  }

  mqttClient.loop();

  while (Serial.available() > 0) {
    char c = (char)Serial.read();

    if (c == '\n' || c == '\r') {
      handleCommand(inputLine);
      inputLine = "";
    }
    else if (inputLine.length() < 32) {
      inputLine += c;
    }
  }
}

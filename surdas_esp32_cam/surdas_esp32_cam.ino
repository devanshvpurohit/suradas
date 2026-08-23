#include "esp_camera.h"
#include <WiFi.h>
#include <WebServer.h>

// =====================================================
// WIFI CONFIGURATION (AP MODE)
// =====================================================
const char* AP_SSID = "SURDAS_EYES";
const char* AP_PASSWORD = "12345678";

WebServer server(80);

// =====================================================
// AI-THINKER ESP32-CAM PIN DEFINITIONS
// =====================================================
#define PWDN_GPIO_NUM     32
#define RESET_GPIO_NUM    -1
#define XCLK_GPIO_NUM      0
#define SIOD_GPIO_NUM     26
#define SIOC_GPIO_NUM     27
#define Y9_GPIO_NUM       35
#define Y8_GPIO_NUM       34
#define Y7_GPIO_NUM       39
#define Y6_GPIO_NUM       36
#define Y5_GPIO_NUM       21
#define Y4_GPIO_NUM       19
#define Y3_GPIO_NUM       18
#define Y2_GPIO_NUM        5
#define VSYNC_GPIO_NUM    25
#define HREF_GPIO_NUM     23
#define PCLK_GPIO_NUM     22

#define FLASH_LED_PIN      4  // Onboard high-power flash LED

// =====================================================
// CAMERA INITIALIZATION
// =====================================================
bool initCamera() {
  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer   = LEDC_TIMER_0;
  config.pin_d0 = Y2_GPIO_NUM;
  config.pin_d1 = Y3_GPIO_NUM;
  config.pin_d2 = Y4_GPIO_NUM;
  config.pin_d3 = Y5_GPIO_NUM;
  config.pin_d4 = Y6_GPIO_NUM;
  config.pin_d5 = Y7_GPIO_NUM;
  config.pin_d6 = Y8_GPIO_NUM;
  config.pin_d7 = Y9_GPIO_NUM;
  config.pin_xclk = XCLK_GPIO_NUM;
  config.pin_pclk = PCLK_GPIO_NUM;
  config.pin_vsync = VSYNC_GPIO_NUM;
  config.pin_href = HREF_GPIO_NUM;
  config.pin_sscb_sda = SIOD_GPIO_NUM;
  config.pin_sscb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn = PWDN_GPIO_NUM;
  config.pin_reset = RESET_GPIO_NUM;
  config.xclk_freq_hz = 20000000;
  config.pixel_format = PIXFORMAT_JPEG;

  if (psramFound()) {
    config.frame_size = FRAMESIZE_QVGA;  // 320x240 for 25-30 FPS streaming
    config.jpeg_quality = 10;            // Lower number = higher quality (10-63)
    config.fb_count = 2;
    config.grab_mode = CAMERA_GRAB_LATEST;
    config.fb_location = CAMERA_FB_IN_PSRAM;
  } else {
    config.frame_size = FRAMESIZE_QQVGA;
    config.jpeg_quality = 15;
    config.fb_count = 1;
  }

  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("Camera initialization failed: 0x%x\n", err);
    return false;
  }

  sensor_t *s = esp_camera_sensor_get();
  // Image tuning for computer vision
  s->set_brightness(s, 1);
  s->set_contrast(s, 1);
  s->set_saturation(s, 0);
  s->set_whitebal(s, 1);       // Auto White Balance
  s->set_gain_ctrl(s, 1);      // Auto Gain Control
  s->set_exposure_ctrl(s, 1);  // Auto Exposure

  return true;
}

// =====================================================
// SERVER ENDPOINTS
// =====================================================

// 1. Live MJPEG Stream for YOLO + MiDaS
void handleStream() {
  WiFiClient client = server.client();
  client.println("HTTP/1.1 200 OK");
  client.println("Content-Type: multipart/x-mixed-replace; boundary=frame");
  client.println("Access-Control-Allow-Origin: *");
  client.println();

  while (client.connected()) {
    camera_fb_t *fb = esp_camera_fb_get();
    if (!fb) break;

    client.printf("--frame\r\nContent-Type: image/jpeg\r\nContent-Length: %u\r\n\r\n", fb->len);
    client.write(fb->buf, fb->len);
    client.print("\r\n");
    esp_camera_fb_return(fb);

    delay(20);  // Frame rate throttling
  }
}

// 2. High-Quality Snapshot for OCR / Text Recognition
void handleCapture() {
  camera_fb_t *fb = esp_camera_fb_get();
  if (!fb) {
    server.send(500, "text/plain", "Capture failed");
    return;
  }
  server.sendHeader("Content-Type", "image/jpeg");
  server.sendHeader("Content-Length", String(fb->len));
  server.send_P(200, "image/jpeg", (const char *)fb->buf, fb->len);
  esp_camera_fb_return(fb);
}

// 3. Flashlight Control (/led?state=on /led?state=off)
void handleLed() {
  if (server.hasArg("state")) {
    String state = server.arg("state");
    if (state == "on") {
      digitalWrite(FLASH_LED_PIN, HIGH);
      server.send(200, "text/plain", "LED ON");
      return;
    } else if (state == "off") {
      digitalWrite(FLASH_LED_PIN, LOW);
      server.send(200, "text/plain", "LED OFF");
      return;
    }
  }
  server.send(400, "text/plain", "Invalid state param");
}

// 4. Status Check
void handleStatus() {
  String json = "{\"status\":\"online\",\"model\":\"AI-Thinker\",\"ip\":\"" + WiFi.softAPIP().toString() + "\"}";
  server.send(200, "application/json", json);
}

// =====================================================
// SETUP & LOOP
// =====================================================
void setup() {
  Serial.begin(115200);
  pinMode(FLASH_LED_PIN, OUTPUT);
  digitalWrite(FLASH_LED_PIN, LOW);

  Serial.println("\n[SURDAS] Booting Optical Transmitter...");
  if (!initCamera()) {
    Serial.println("[SURDAS] FATAL: Camera initialization failed.");
    while (true) delay(1000);
  }

  WiFi.mode(WIFI_AP);
  WiFi.softAP(AP_SSID, AP_PASSWORD);

  server.on("/stream", HTTP_GET, handleStream);
  server.on("/capture", HTTP_GET, handleCapture);
  server.on("/led", HTTP_GET, handleLed);
  server.on("/status", HTTP_GET, handleStatus);
  server.begin();

  Serial.println("[SURDAS] Ready!");
  Serial.print("Wi-Fi SSID: "); Serial.println(AP_SSID);
  Serial.print("Stream URL: http://"); Serial.print(WiFi.softAPIP()); Serial.println("/stream");
}

void loop() {
  server.handleClient();
  delay(1);
}

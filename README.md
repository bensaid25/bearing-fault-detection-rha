# Intelligent Bearing Fault Detection and Remaining Health Assessment

A predictive maintenance system for industrial machines, combining three AI models (fault classification, remaining useful life prediction, anomaly detection), a real-time IoT pipeline, and a monitoring dashboard.

## 🎯 Objective

Detect bearing faults, estimate a motor's Remaining Useful Life (RUL), and detect vibration anomalies in real time, using sensor data from an ESP32-based IoT setup.

## 🏗️ Architecture

```
Machine/Motor → Sensors (MPU6050) → ESP32 → Wi-Fi/MQTT → Backend (FastAPI) → AI Models → Frontend (dashboard)
```

![Architecture](docs/overall.png)

## 🧠 AI Components

| Component | Dataset | Task | Final model |
|---|---|---|---|
| 1 — Fault classification | CWRU (Case Western Reserve University) | Normal / Ball fault / Inner-race / Outer-race | *[name of the selected ML model]* |
| 2 — RUL prediction | NASA C-MAPSS (FD001) | Remaining Useful Life | *[name of the selected DL model]* |
| 3 — Anomaly detection | NASA IMS | Unsupervised anomaly detection (vibration signals) | Isolation Forest (trained on healthy data only) |


## 📁 Project structure

```
├── final-notebooks/         # Final training/analysis notebooks for the 3 components
│   ├── cwru/
│   ├── cmapss/
│   │   ├── ML/
│   │   ├── DL/
│   │   └── model_comparison.ipynb
│   └── ims/
├── models/
│   └── production/          # Final trained models used by the API
├── src/                     # Core source code
├── api/                     # FastAPI backend
│   ├── app.py
│   ├── model_loader.py
│   ├── inference_cwru.py
│   ├── inference_cmapss.py
│   ├── inference_ims.py
│   ├── inference_sensor.py
│   └── schemas.py
├── agents/                  # Multi-agent orchestration 
├── frontend/                # Dashboard application
├── IoT equipments/          # ESP32 firmware, MPU6050 wiring, MQTT publishing
├── integration_material/    # Real hardware setup (components, wiring docs, photos)
├── vibration_test/          # Real vibration test setup and recorded results
├── tests/                   # Automated tests (pytest)
│   ├── conftest.py
│   ├── test_cwru.py
│   ├── test_cmapss.py
│   ├── test_ims.py
│   ├── test_sensor.py
│   └── test_health.py
├── tools/                   # Utility scripts (payload generation, model inspection)
│   ├── generate_test_payloads.py
│   ├── generate_real_test_payloads.py
│   ├── inspect_models.py
│   ├── test_api.py
│   └── sample_payloads/
├── docs/
├── pytest.ini
└── requirements.txt
```

## ⚙️ Installation

```bash
git clone https://github.com/bensaid25/bearing-fault-detection-rha.git
cd bearing-fault-detection-rha
python -m venv venv
source venv/bin/activate   # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

## 🚀 Usage

**1. Run the API (FastAPI)**
```bash
cd api
uvicorn app:app --reload
```

**2. Run the IoT sensor stream**
```bash
# See IoT equipments/README.md for ESP32 flashing and MQTT setup
```

**3. Run the frontend**
```bash
cd frontend
streamlit run app.py   # adjust filename to match the actual entry point
```

## ✅ Tests

25 automated tests covering the three prediction endpoints:
```bash
pytest tests/
```

Utility scripts are available in `tools/` to generate test payloads (`generate_test_payloads.py`, `generate_real_test_payloads.py`) and inspect loaded models (`inspect_models.py`).

## 📦 Models

Final trained models are stored in `models/production/`. If a model exceeds GitHub's file size limit, it is hosted externally: *[GitHub Release / Hugging Face / Drive link]*.

Verify models load correctly with:
```bash
python tools/inspect_models.py
```

## 🔧 Hardware setup

- `IoT equipments/` — ESP32 + MPU6050 firmware and wiring
- `integration_material/` — full real-world hardware integration (components used, wiring diagrams, setup photos)
- `vibration_test/` — real vibration test bench setup and recorded results

## 📊 Demo

![Dashboard](docs/dashboard.png)


## 🏢 Context

Internship project carried out at **ELYOS DIGITAL**, under the supervision of Salma KALLELA — ENSI.

## 📜 License

This project is licensed under the MIT License.

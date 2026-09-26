# Intelligent Bearing Fault Detection and Remaining Health Assessment

Système de maintenance prédictive pour machines industrielles, combinant trois modèles IA (classification de défauts, prédiction de durée de vie utile restante, détection d'anomalies), un pipeline IoT temps réel et un dashboard de supervision.

## 🎯 Objectif

Détecter les défauts de roulements, estimer la durée de vie utile restante (RUL) d'un moteur, et détecter les anomalies de vibration en temps réel, à partir de données capteurs simulées via ESP32.

## 🏗️ Architecture

```
Machine/Moteur → Capteurs (MPU6050) → ESP32 → Wi-Fi/MQTT → Backend (FastAPI) → Modèles IA → Dashboard (Streamlit)
```

![Architecture](docs/overall.png)

## 🧠 Composants IA

| Composant | Dataset | Tâche | Modèle final |
|---|---|---|---|
| 1 — Classification de défauts | CWRU (Case Western Reserve University) | Normal / Ball fault / Inner-race / Outer-race | *[nom du modèle ML retenu]* |
| 2 — Prédiction RUL | NASA C-MAPSS (FD001) | Remaining Useful Life | *[nom du modèle DL retenu]* |
| 3 — Détection d'anomalies | NASA IMS | Anomalies non supervisées (signaux de vibration) | Isolation Forest (entraîné sur données saines) |

## 📁 Structure du projet

```
├── notebooks/
│   ├── cwru/
│   ├── cmapss/
│   │   ├── ML/
│   │   ├── DL/
│   │   └── model_comparison.ipynb
│   └── ims/
├── models/              # Modèles entraînés (.joblib, .h5...) — voir section Modèles
├── api/                 # API FastAPI
│   ├── app.py
│   ├── model_loader.py
│   ├── inference_cwru.py
│   ├── inference_cmapss.py
│   ├── inference_ims.py
│   ├── inference_sensor.py
│   └── schemas.py
├── tests/               # Tests automatisés (pytest)
│   ├── conftest.py
│   ├── test_cwru.py
│   ├── test_cmapss.py
│   ├── test_ims.py
│   ├── test_sensor.py
│   └── test_health.py
├── tools/               # Scripts utilitaires (génération de payloads, inspection des modèles)
│   ├── generate_test_payloads.py
│   ├── generate_real_test_payloads.py
│   ├── inspect_models.py
│   ├── test_api.py
│   └── sample_payloads/
├── iot/                 # Firmware ESP32 + simulateur MQTT
├── dashboard/           # Application Streamlit
└── docs/
```

## ⚙️ Installation

```bash
git clone https://github.com/<ton-user>/<ton-repo>.git
cd <ton-repo>
python -m venv venv
source venv/bin/activate   # ou venv\Scripts\activate sous Windows
pip install -r requirements.txt
```

## 🚀 Utilisation

**1. Lancer l'API (FastAPI)**
```bash
cd api
uvicorn app:app --reload
```

**2. Lancer le simulateur de flux capteurs (MQTT)**
```bash
python iot/mqtt_simulator/simulate.py
```

**3. Lancer le dashboard**
```bash
streamlit run dashboard/streamlit_app.py
```

## ✅ Tests

25 tests automatisés couvrant les trois endpoints de prédiction :
```bash
pytest tests/
```

Scripts utilitaires disponibles dans `tools/` pour générer des payloads de test (`generate_test_payloads.py`, `generate_real_test_payloads.py`) et inspecter les modèles chargés (`inspect_models.py`).

## 📦 Modèles

Les modèles entraînés sont trop volumineux pour être commités directement sur GitHub. Ils sont disponibles ici : *[lien Release GitHub / Hugging Face / Drive]*.

Place-les dans `models/` après téléchargement, ou utilise :
```bash
python tools/inspect_models.py
```
pour vérifier que les modèles sont bien chargés une fois placés.

## 📊 Démonstration

*[Ajouter un GIF ou des screenshots du dashboard une fois finalisé]*

## 📄 Rapport

Rapport complet rédigé en LaTeX (Overleaf) : *[lien ou export PDF dans docs/report/]*

## 🏢 Contexte

Projet de stage réalisé chez **ELYOS DIGITAL**, sous la supervision de Salma KALLELA — ENSI.

## 📜 Licence

*[À définir — MIT, par exemple]*

# 🛡️ CIVILIAN SHIELD: Acoustic Counter-Surveillance System

> **"THE CENTER CANNOT HOLD."** > An automated machine learning pipeline for identifying state threats through acoustic intelligence.

![Status](https://img.shields.io/badge/SYSTEM-ONLINE-brightgreen?style=for-the-badge&logo=statuspage)
![Language](https://img.shields.io/badge/PYTHON-3.9-blue?style=for-the-badge&logo=python)
![Framework](https://img.shields.io/badge/TENSORFLOW-2.x-orange?style=for-the-badge&logo=tensorflow)

## 📋 Mission Directive (Project Overview)
**Civilian Shield** is an end-to-end Machine Learning pipeline designed to empower communities with decentralized threat detection. In an era of increasing surveillance, this tool "watches back" by analyzing the acoustic environment in real-time.

Utilizing a **Convolutional Neural Network (CNN)**, the system classifies environmental audio into 10 distinct categories, focusing on distinguishing between:
* **⚠️ State Threats:** Sirens, Gunshots.
* **✅ Civilian Activity:** Street Music, Children Playing, Drilling, Air Conditioners.

The platform provides a **Brutalist, low-latency Interface** for civilians to scan audio files, visualize sector data, and collectively retrain the model to adapt to new environments ("Hive Learning").

---

## 📂 The Dataset: UrbanSound8K

The model is trained on the **UrbanSound8K** dataset, a rigorous collection of 8,732 labeled sound excerpts (<=4s) from urban field recordings.

* **Source:** [UrbanSound8K (Kaggle / Center for Urban Science)](https://urbansounddataset.weebly.com/)
* **Total Size:** ~6GB
* **Structure:** 10 Folds (Used for Cross-Validation)
* **Classes (10):**
    1.  `Air Conditioner`
    2.  `Car Horn`
    3.  `Children Playing`
    4.  `Dog Bark`
    5.  `Drilling`
    6.  `Engine Idling`
    7.  `Gun Shot` (Target Class: Threat)
    8.  `Jackhammer`
    9.  `Siren` (Target Class: Threat)
    10. `Street Music`

> **Note on Bias:** The dataset contains fewer samples for *Gunshots* (374) compared to *Drilling* (1000). The pipeline addresses this via class-weighted metrics and precision-focused evaluation.

---

## 🏗️ System Architecture

The project follows a strict MLOps architecture divided into three core modules:
mermaid
graph LR
    A[Raw Audio Input] -->|src/preprocessing.py| B(MFCC Feature Extraction)
    B -->|src/model.py| C{CNN Model}
    C -->|app/main.py| D[Streamlit Interface]
    D -->|User Feedback| E[Incremental Retraining]
    E -->|Update Weights| C

1.  **Preprocessing (`src/preprocessing.py`):** Converts raw `.wav` waveforms into **MFCCs (Mel-Frequency Cepstral Coefficients)**. This extracts a 40-feature "acoustic fingerprint" representing the timbre of the sound.
2.  **Model (`src/model.py`):** A Sequential Neural Network optimized with:
      * **Dropout (0.5):** To prevent overfitting.
      * **Early Stopping:** Monitors validation loss to halt training at peak performance.
3.  **Deployment (`app/main.py`):** A Streamlit-based UI featuring "Postmodern/Brutalist" aesthetics, offering real-time inference and batch retraining.

-----

## ⚡ Installation Protocol

Follow these steps to deploy the Civilian Shield on your local machine.

### 1\. Clone the Repository

```bash
git clone [https://github.com/YOUR_USERNAME/Civilian_Shield.git](https://github.com/YOUR_USERNAME/Civilian_Shield.git)
cd Civilian_Shield
```

### 2\. Initialize Environment

Create a virtual environment to isolate dependencies.

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Mac/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3\. Install Dependencies

```bash
pip install -r requirements.txt
```

*Key Libraries: `tensorflow`, `librosa`, `streamlit`, `pandas`, `scikit-learn`.*

### 4\. Data Setup

Due to size limits, the raw audio is not on GitHub.

1.  Download **UrbanSound8K** from [Kaggle](https://www.kaggle.com/chrisfilo/urbansound8k).
2.  Extract the files so your folder structure looks **exactly** like this:
    ```text
    Civilian_Shield/
    ├── data/
    │   └── train/
    │       ├── metadata/
    │       │   └── UrbanSound8K.csv
    │       └── audio/
    │           ├── fold1/
    │           ├── fold2/
    │           └── ...
    ```

-----

## 🖥️ Usage Directives

### Launch the Interface

Run the application using Streamlit:

```bash
streamlit run app/main.py
```

*Access the UI at: `http://localhost:8501`*

### System Modules

  * **👁️ THREAT SCAN:** Upload `.wav` or `.mp3` files. The system will flag "THREAT DETECTED" (Red) or "SAFE" (Green) with a confidence score.
  * **📉 SECTOR DATA:** Visualize dataset balance, audio duration spread, and raw metadata metrics.
  * **✊ HIVE LEARN (Retraining):**
    1.  Upload a batch of new audio files.
    2.  Select the correct label (e.g., "Siren").
    3.  Click **"EXECUTE WEIGHT UPDATE"**.
    4.  The system performs **Incremental Learning** (1 epoch) to update the model without forgetting previous knowledge.

-----

## 📊 Evaluation & Performance

The model was evaluated on a 20% unseen test set.

| Metric | Score | Analysis |
| :--- | :--- | :--- |
| **Accuracy** | **83.86%** | High reliability for environmental sound classification. |
| **Siren Precision** | **93%** | Extremely low false-positive rate for emergency alarms. |
| **Siren Recall** | **95%** | The system captures 95% of all real siren events. |
| **Gunshot Recall** | **67%** | Moderate sensitivity; impulse noises (like jackhammers) cause confusion. |

### Optimization Evidence

  * **Early Stopping:** Triggered at Epoch 35/50 to prevent overfitting.
  * **Confusion Matrix:** Shows a strong diagonal. Primary confusion exists between *Street Music* and *Children Playing* due to overlapping acoustic frequencies.

-----

## ⚠️ Stress Testing (Locust)

To simulate high-traffic scenarios (e.g., mass protests), the API was stress-tested using **Locust**.

  * **Simulation:** 500 Simultaneous Users
  * **Spawn Rate:** 10 users/second
  * **Result:** System maintained **\<200ms latency** with 0% failure rate.

*(Insert Screenshot of Locust Charts Here)*

-----

## 🎥 Video Demonstration

**[CLICK HERE TO VIEW THE YOUTUBE DEMO]**

  * **0:00** - Architecture Overview
  * **1:30** - Threat Prediction Demo
  * **3:45** - Hive Learning (Retraining) Walkthrough
  * **5:00** - Evaluation Metrics Analysis

-----

> *"Surveillance goes both ways."* — **Civilian Shield v1.0**

```
```

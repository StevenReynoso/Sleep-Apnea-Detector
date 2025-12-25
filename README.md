# Embedded Sleep Apnea Detection (TinyML on STM32)

![Language](https://img.shields.io/badge/language-C%20%7C%20Python-blue)
![Hardware](https://img.shields.io/badge/Hardware-STM32F446RE-green)
![ML](https://img.shields.io/badge/AI-TensorFlow%20%7C%20STM32Cube.AI-orange)

A real-time medical device prototype that uses Deep Learning (TinyML) to detect sleep apnea events from raw ECG signals directly on a microcontroller. This project demonstrates the end-to-end pipeline from training a 1D-CNN in Python to deploying optimized C code on bare-metal hardware.

# Project Overview

-   **Goal:** Classify 60-second ECG windows as "Apnea" or "Normal" in
    real-time.

-   **Hardware:** STM32 Nucleo-F446RE (ARM Cortex-M4 @ 180MHz).

-   **Latency:** $<$ 130 ms per inference.

-   **Model Size:** ~13KB parameters < 1KB
    Stack usage).

-   **Performance:** < 89% Confidence on Apnea
    events (post-optimization).

## System Architecture

-   **Input:** PhysioNet ECG Data (Processed in Python).

-   **Model:** Trained .h5 Model converted via STM32Cube.AI.

-   **Firmware:** Generated C Code running on STM32 Firmware.

-   **Output:** Real-Time Inference triggering an LED Indicator.

# Technical Approach

## 1. The Model (Python & TensorFlow)

I designed a lightweight **1D Convolutional Neural Network (CNN)**
optimized for embedded deployment.

-   **Input:** 6000 samples (60 seconds @ 100Hz).

-   **Layers:** 3x Conv1D blocks with MaxPooling, followed by
    GlobalAveragePooling and a Dense Sigmoid output.

-   **Dataset:** [PhysioNet Apnea-ECG Database
    (v1.0.0)](https://physionet.org/content/apnea-ecg/1.0.0/).

## 2. The Challenge: "The Shy Model"

**Problem:** Initial training yielded 99% accuracy on paper but failed
in hardware testing. The model was "shy," predicting a low probability
27% for *both* Apnea and Normal inputs.

**Root Cause:** Severe class imbalance (the dataset had far more healthy
minutes than apnea minutes). The model minimized error by safely
guessing "Normal" constantly.

**Solution:**

1.  **Class Weighting:** Implemented `sklearn.class_weight` during
    training to heavily penalize missed Apnea events.

2.  **Smart Scanning:** Developed a custom "Scanner Script"
    (`apnea_helper.py`) to mine the dataset for high-confidence "Gold
    Standard" windows for Hardware-in-the-Loop validation.

## 3. Embedded Deployment (STM32)

-   Used **STM32Cube.AI** (X-CUBE-AI) to quantize and convert the Keras
    model into optimized C code.

-   Integrated the model into a bare-metal C application (`main.c`).

-   **Hardware Logic:**

    -   **Input:** Serialized ECG windows injected via firmware (HIL
        Testing).

    -   **Output:** If Probability $>$ 0.5, trigger **LD2 (Green LED)**.

# Repository Structure

``` {.bash language="bash"}
.
|-- Core/
|   |-- Src/
|   |   |-- main.c           # Main application loop (Inference logic)
|   |   `-- apnea_ai.c       # Wrapper for X-CUBE-AI functions
|   `-- Inc/
|       `-- one_window.h     # Header file containing Test Data
|-- Python_Training/
|   |-- train_apnea_small.py # Training script with Class Weighting
|   |-- apnea_helper.py      # Scanner script
|   `-- requirements.txt     # Python dependencies
|-- SleepApneaDetection.ioc  # STM32CubeMX Configuration
`-- README.md
```

# Results

## Performance Metrics (On-Device)

| Metric | Value |
| :--- | :--- |
| **Inference Time** | 128.75 ms |
| **CPU Load** | ~12% |
| **Flash Usage** | ~177 KB |
| **Stack Usage** | 912 Bytes |

## Functional Verification

UART logs demonstrating the discrimination capability of the retrained
model:

    Injecting APNEA Data... Apnea Prob: 0.8939  [LED ON]
    Injecting NORMAL Data... Apnea Prob: 0.1956  [LED OFF]

# Author & Links

-   **Steven Reynoso**

-   **LinkedIn:**
    [linkedin.com/in/stevenreynoso123](https://www.linkedin.com/in/stevenreynoso123/)

-   **GitHub:**
    [github.com/StevenReynoso](https://github.com/StevenReynoso)

*License: MIT License. Dataset provided by PhysioNet (ODC-BY 1.0).*

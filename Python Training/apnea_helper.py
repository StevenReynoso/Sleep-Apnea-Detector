import os
import numpy as np
import wfdb
import tensorflow as tf

# ========= CONFIG =========
DATA_DIR = r"E:\EmbeddedWork\apnea-ecg-database-1.0.0"
OUT_H    = r"E:\EmbeddedWork\SleepApneaDetection\Core\Inc\one_window.h" # Ensure this matches your C path
MODEL_PATH = r"E:\EmbeddedWork\saved_models\apnea_cnn_small.h5"
FS = 100
WINDOW_SEC = 60
SAMPLES_PER_MIN = FS * WINDOW_SEC
# ==========================

def get_window(record, minute_idx):
    try:
        rec_path = os.path.join(DATA_DIR, record)
        rec = wfdb.rdrecord(rec_path)
        ecg = rec.p_signal[:, 0].astype(np.float32)
        start = minute_idx * SAMPLES_PER_MIN
        end = start + SAMPLES_PER_MIN
        if end > len(ecg): return None
        window = ecg[start:end]
        # Preprocessing
        mean = np.mean(window)
        std = np.std(window)
        if std < 1e-6: return None
        return (window - mean) / std
    except:
        return None

# 1. Load Model
print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)

# 2. SCAN for the Best "Apnea" Window (Highest Probability)
print("Scanning 'a01' for the strongest Apnea signal...")
best_apnea_prob = -1.0
best_apnea_window = None
best_apnea_idx = -1

# Scan first 60 minutes of record a01
for i in range(60):
    win = get_window("a01", i)
    if win is not None:
        p = model.predict(win[np.newaxis, :, np.newaxis], verbose=0)[0, 0]
        if p > best_apnea_prob:
            best_apnea_prob = p
            best_apnea_window = win
            best_apnea_idx = i

print(f"Found best APNEA candidate at minute {best_apnea_idx} with Prob: {best_apnea_prob:.4f}")

# 3. SCAN for the Best "Normal" Window (Lowest Probability)
print("Scanning 'c01' for the clearest Normal signal...")
best_normal_prob = 2.0
best_normal_window = None
best_normal_idx = -1

# Scan first 60 minutes of record c01
for i in range(60):
    win = get_window("c01", i)
    if win is not None:
        p = model.predict(win[np.newaxis, :, np.newaxis], verbose=0)[0, 0]
        if p < best_normal_prob:
            best_normal_prob = p
            best_normal_window = win
            best_normal_idx = i

print(f"Found best NORMAL candidate at minute {best_normal_idx} with Prob: {best_normal_prob:.4f}")

# 4. CRITICAL CHECK for Demo
threshold_suggestion = (best_apnea_prob + best_normal_prob) / 2
print("-" * 40)
print(f"SUGGESTED THRESHOLD FOR MAIN.C: {threshold_suggestion:.4f}")
print("-" * 40)

if best_apnea_prob < best_normal_prob + 0.1:
    print("WARNING: Model cannot distinguish well. Demo might look weak.")
else:
    print("SUCCESS: Clear distinction found.")

# 5. Write to Header
with open(OUT_H, "w") as f:
    f.write("#ifndef ONE_WINDOW_H\n#define ONE_WINDOW_H\n\n")
    f.write("#include <stdint.h>\n\n")
    f.write(f"#define ONE_WINDOW_LEN {SAMPLES_PER_MIN}\n\n")

    f.write(f"/* Best Apnea Candidate (Minute {best_apnea_idx}, Prob {best_apnea_prob:.4f}) */\n")
    f.write("static const float apnea_window[ONE_WINDOW_LEN] = {\n")
    for i, v in enumerate(best_apnea_window):
        f.write(f"  {v:.7f}f,")
        if (i + 1) % 8 == 0: f.write("\n")
    f.write("\n};\n\n")

    f.write(f"/* Best Normal Candidate (Minute {best_normal_idx}, Prob {best_normal_prob:.4f}) */\n")
    f.write("static const float normal_window[ONE_WINDOW_LEN] = {\n")
    for i, v in enumerate(best_normal_window):
        f.write(f"  {v:.7f}f,")
        if (i + 1) % 8 == 0: f.write("\n")
    f.write("\n};\n\n")

    f.write("#endif // ONE_WINDOW_H\n")
print(f"Header file updated at {OUT_H}")
import os
import glob
import numpy as np
import wfdb
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.utils import class_weight  # <--- Essential import

# ========= CONFIG =========
DATA_DIR = r"E:\EmbeddedWork\apnea-ecg-database-1.0.0" # TODO: Verify path
RECORD_LIMIT = 69               # Use all available records
MAX_MINUTES_PER_RECORD = 1000   # Use full duration of records
WINDOW_SEC = 60
FS_EXPECTED = 100
# ==========================

if not os.path.isdir(DATA_DIR):
    raise FileNotFoundError(f"DATA_DIR does not exist: {DATA_DIR}")


def list_records(data_dir):
    """Return base record names (no extension) for all .dat files."""
    dat_files = glob.glob(os.path.join(data_dir, "*.dat"))
    base_names = sorted({os.path.splitext(os.path.basename(f))[0] for f in dat_files})
    return base_names


def load_record_windows(record_name, data_dir):
    """
    Load one record, slice into 60 s windows, attach apnea labels using .apn.
    Returns:
        X_windows: (num_minutes, 6000) float32
        y_labels:  (num_minutes,) int (1=apnea, 0=normal)
    """
    rec_path = os.path.join(data_dir, record_name)

    # Load ECG
    rec = wfdb.rdrecord(rec_path)
    fs = int(rec.fs)
    if fs != FS_EXPECTED:
        raise ValueError(f"Unexpected fs {fs} for {record_name}, expected {FS_EXPECTED}")
    
    ecg = rec.p_signal[:, 0].astype(np.float32)

    # Load apnea annotations (.apn)
    try:
        ann = wfdb.rdann(rec_path, "apn")
    except FileNotFoundError:
        # If .apn is missing, skip this record entirely
        return np.array([]), np.array([])

    symbols = np.array(ann.symbol)
    samples = np.array(ann.sample)

    samples_per_min = FS_EXPECTED * WINDOW_SEC
    num_minutes_possible = len(ecg) // samples_per_min

    # Build minute-level labels
    minute_labels = []
    for m in range(num_minutes_possible):
        start = m * samples_per_min
        end = start + samples_per_min
        
        idx = np.where((samples >= start) & (samples < end))[0]
        if len(idx) == 0:
            minute_labels.append(None)
        else:
            sym = symbols[idx[0]]
            if sym not in ("A", "N"):
                minute_labels.append(None)
            else:
                minute_labels.append(sym)

    X_windows = []
    y_labels = []

    for m, sym in enumerate(minute_labels):
        if sym is None:
            continue
        if m >= MAX_MINUTES_PER_RECORD:
            break

        start = m * samples_per_min
        end = start + samples_per_min
        window = ecg[start:end]

        # Safety checks
        if window.shape[0] != samples_per_min:
            continue
        if not np.isfinite(window).all():
            continue

        # Normalization
        mean = np.mean(window)
        std = np.std(window)
        if not np.isfinite(mean) or not np.isfinite(std) or std < 1e-6:
            continue

        window = (window - mean) / std

        X_windows.append(window)
        y_labels.append(1 if sym == "A" else 0)

    return np.array(X_windows, dtype=np.float32), np.array(y_labels, dtype=np.int32)


def build_small_cnn(input_len):
    inputs = layers.Input(shape=(input_len, 1))
    x = layers.Conv1D(8, 7, strides=2, padding="same", activation="relu")(inputs)
    x = layers.MaxPool1D(2)(x)
    x = layers.Conv1D(16, 5, strides=2, padding="same", activation="relu")(x)
    x = layers.MaxPool1D(2)(x)
    x = layers.Conv1D(32, 3, strides=2, padding="same", activation="relu")(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(32, activation="relu")(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = models.Model(inputs, outputs)

    opt = tf.keras.optimizers.Adam(learning_rate=1e-4, clipnorm=1.0)
    model.compile(optimizer=opt, loss="binary_crossentropy", metrics=["accuracy"])
    return model


####
## Function for HIL Simulator
####
def save_golden_data(X, y):
    """
    Saves one Normal and one Apnea window to both C header and text files.
    This ensures Firmware and HIL Simulator use IDENTICAL data.
    """
    print("\n💾 Saving Golden Data...")
    
    # Try finding the 10th or 20th Normal sample instead of the 1st
    idx_normal = np.where(y == 0)[0][20] 
    idx_apnea = np.where(y == 1)[0][20]
    
    # Extract the actual windows (already normalized)
    win_normal = X[idx_normal].flatten()
    win_apnea = X[idx_apnea].flatten()
    
    # --- 1. Save C Header (for STM32) ---
    with open("one_window.h", "w") as f:
        f.write("#ifndef ONE_WINDOW_H\n")
        f.write("#define ONE_WINDOW_H\n\n")
        f.write("#include <stdint.h>\n\n")
        f.write(f"#define ONE_WINDOW_LEN {len(win_normal)}\n\n")
        
        f.write("static const float normal_window[ONE_WINDOW_LEN] = {\n")
        f.write(", ".join(f"{x:.8f}f" for x in win_normal))
        f.write("\n};\n\n")
        
        f.write("static const float apnea_window[ONE_WINDOW_LEN] = {\n")
        f.write(", ".join(f"{x:.8f}f" for x in win_apnea))
        f.write("\n};\n\n")
        f.write("#endif // ONE_WINDOW_H\n")
    
    # --- 2. Save Text Files (for Python HIL) ---
    # We save them as simple lists of numbers
    np.savetxt("normal_data.txt", win_normal, fmt="%.8f", newline=",")
    np.savetxt("apnea_data.txt", win_apnea, fmt="%.8f", newline=",")
    
    print("✅ Created: one_window.h, normal_data.txt, apnea_data.txt")

def main():
    # 1. FIND RECORDS
    records = list_records(DATA_DIR)
    # Filter out 'x' records (test set with no labels)
    records = [r for r in records if not r.startswith('x')]
    
    if len(records) == 0:
        raise RuntimeError(f"No .dat files found in {DATA_DIR}")

    records = records[:RECORD_LIMIT]
    print("Using records:", records)

    # 2. LOAD DATA
    X_list = []
    y_list = []

    for r in records:
        Xr, yr = load_record_windows(r, DATA_DIR)
        print(f"{r}: got {len(yr)} labeled minutes")
        if len(yr) == 0:
            continue
        X_list.append(Xr)
        y_list.append(yr)

    if not X_list:
        raise RuntimeError("No labeled data collected.")

    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    print("Total windows collected:", X.shape[0])

    ### HIL FUNCTION CALL
    # === ADD THIS LINE ===
    save_golden_data(X, y)
    # =====================

    # 3. PREPARE DATA (Reshape & Split)
    X = X[..., np.newaxis]
    input_len = X.shape[1]

    n = X.shape[0]
    n_train = int(0.8 * n)
    X_train, X_val = X[:n_train], X[n_train:]
    y_train, y_val = y[:n_train], y[n_train:]

    # 4. COMPUTE CLASS WEIGHTS (Fixes imbalance issue)
    # This must happen AFTER y_train is defined!
    weights = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    class_weights_dict = dict(enumerate(weights))
    print("Computed Class Weights:", class_weights_dict)

    # 5. BUILD & TRAIN MODEL
    model = build_small_cnn(input_len)
    model.summary()

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=5,
        batch_size=16,
        class_weight=class_weights_dict  # <--- Apply weights here
    )

    # 6. SAVE MODEL
    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
    print(f"Validation accuracy: {val_acc:.3f}")

    os.makedirs("saved_models", exist_ok=True)
    out_path = os.path.join("saved_models", "apnea_cnn_small.h5")
    model.save(out_path, include_optimizer=False)
    print("Saved model to", out_path)

if __name__ == "__main__":
    main()
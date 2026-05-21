import argparse
import json
from pathlib import Path

try:
    import numpy as np
    import pandas as pd
    import tensorflow as tf
    from sklearn.metrics import classification_report, confusion_matrix, recall_score
except ModuleNotFoundError as exc:
    missing_package = exc.name
    raise SystemExit(
        f"Missing Python package: {missing_package}\n"
        "Install the required packages, then run this script again:\n"
        "  pip install numpy pandas scikit-learn tensorflow\n"
        "If your terminal does not have a 'python' command on macOS, use python3/pip3."
    ) from exc


# Sensor columns used as one timestep feature vector: (ax, ay, az, gx, gy, gz)
SENSOR_COLS = ["ax", "ay", "az", "gx", "gy", "gz"]

LABEL_MAP = {
    "Upright": 0,
    "Mild Forward Flexion": 1,
    "Severe Looking-Down": 2,
}
INV_LABEL_MAP = {value: key for key, value in LABEL_MAP.items()}

SAMPLING_RATE = 50
WINDOW_SEC = 2
WINDOW_SIZE = SAMPLING_RATE * WINDOW_SEC  # 100 rows
HOP_SIZE = 50
BATCH_SIZE = 32

REQUIRED_COLS = ["timestamp_ns", "label"] + SENSOR_COLS
OPTIONAL_COLS = ["session_id", "day_id"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a 1D-CNN posture classifier from smartphone IMU CSV files."
    )
    parser.add_argument(
        "--data_dir",
        default=".",
        help="Directory containing posture_session_*.csv files. Default: current folder.",
    )
    parser.add_argument(
        "--model_dir",
        default="models",
        help="Directory where trained models and reports will be saved. Default: models.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Maximum number of training epochs. Default: 50.",
    )
    parser.add_argument(
        "--transition_margin_sec",
        type=float,
        default=0.25,
        help=(
            "Seconds to remove from both start and end of each same-label segment "
            "before windowing. Default: 0.25."
        ),
    )
    return parser.parse_args()


def find_csv_files(data_dir):
    """Find only real session CSV files and ignore dataset.zip / macOS metadata files."""
    data_path = Path(data_dir)
    csv_files = sorted(
        path
        for path in data_path.glob("posture_session_*.csv")
        if path.is_file() and not path.name.startswith("._")
    )

    if not csv_files:
        raise FileNotFoundError(
            f"No posture_session_*.csv files were found in: {data_path.resolve()}\n"
            "Run this script from PostureIMUData or pass --data_dir with the CSV folder."
        )

    return csv_files


def normalize_column_names(df, source_file):
    """
    Accept minor column-name differences such as leading/trailing spaces or case changes.
    The returned DataFrame uses the expected lowercase names.
    """
    original_columns = list(df.columns)
    normalized_lookup = {str(col).strip().lower(): col for col in original_columns}

    rename_map = {}
    for expected in REQUIRED_COLS + OPTIONAL_COLS:
        actual = normalized_lookup.get(expected.lower())
        if actual is not None:
            rename_map[actual] = expected

    df = df.rename(columns=rename_map)

    missing = [col for col in REQUIRED_COLS if col not in df.columns]
    if missing:
        raise ValueError(
            f"{source_file} is missing required columns: {missing}\n"
            f"Expected required columns: {REQUIRED_COLS}\n"
            f"Actual columns in this CSV: {original_columns}\n"
            "Tip: check spelling, extra spaces, or whether accelerometer/gyroscope "
            "columns were exported with different names."
        )

    return df


def clean_label(label):
    """Map labels after trimming spaces and tolerating simple case differences."""
    if pd.isna(label):
        return None

    stripped = str(label).strip()
    for canonical_label in LABEL_MAP:
        if stripped.lower() == canonical_label.lower():
            return canonical_label
    return None


def estimate_sampling_rate(df):
    """Estimate Hz from median adjacent timestamp difference in nanoseconds."""
    timestamps = pd.to_numeric(df["timestamp_ns"], errors="coerce").dropna().sort_values()
    diffs = timestamps.diff().dropna()
    diffs = diffs[diffs > 0]

    if diffs.empty:
        return np.nan

    median_diff_ns = float(diffs.median())
    if median_diff_ns <= 0:
        return np.nan

    return 1e9 / median_diff_ns


def load_all_csv(data_dir):
    csv_files = find_csv_files(data_dir)
    frames = []
    file_row_counts = {}
    file_sampling_rates = {}
    nan_counts = pd.Series(dtype="int64")

    for csv_path in csv_files:
        try:
            df = pd.read_csv(csv_path)
        except Exception as exc:
            raise RuntimeError(f"Failed to read CSV file {csv_path}: {exc}") from exc

        df = normalize_column_names(df, csv_path.name)
        file_row_counts[csv_path.name] = len(df)
        file_sampling_rates[csv_path.name] = estimate_sampling_rate(df)

        # Optional columns are useful for debugging and future split strategies.
        if "session_id" not in df.columns:
            df["session_id"] = csv_path.stem.replace("posture_session_", "")
        if "day_id" not in df.columns:
            df["day_id"] = "unknown"

        keep_cols = ["timestamp_ns", "session_id", "day_id", "label"] + SENSOR_COLS
        df = df[keep_cols].copy()
        df["source_file"] = csv_path.name

        raw_label_values = df["label"].dropna().unique()
        df["label"] = df["label"].apply(clean_label)
        unknown_labels = df["label"].isna().sum()
        if unknown_labels:
            raise ValueError(
                f"{csv_path.name} contains {unknown_labels} rows with labels not in LABEL_MAP.\n"
                f"Expected labels: {list(LABEL_MAP.keys())}\n"
                f"Labels found in file: {list(raw_label_values)}"
            )

        for col in ["timestamp_ns"] + SENSOR_COLS:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        nan_counts = nan_counts.add(df.isna().sum(), fill_value=0).astype("int64")
        frames.append(df)

    data = pd.concat(frames, ignore_index=True)
    before_drop = len(data)
    data = data.dropna(subset=["timestamp_ns", "label"] + SENSOR_COLS).copy()
    dropped = before_drop - len(data)

    data = data.sort_values(["source_file", "timestamp_ns"]).reset_index(drop=True)

    quality = {
        "csv_files": csv_files,
        "file_row_counts": file_row_counts,
        "file_sampling_rates": file_sampling_rates,
        "nan_counts": nan_counts,
        "dropped_rows": dropped,
    }
    return data, quality


def print_data_quality(df, quality):
    total_raw_rows = sum(quality["file_row_counts"].values())

    print("\n=== Data Quality Check ===")
    print(f"CSV files read: {len(quality['csv_files'])}")
    print(f"Total raw rows: {total_raw_rows}")
    print(f"Total usable rows: {len(df)}")

    if quality["dropped_rows"]:
        print(f"Rows dropped because of NaN in required fields: {quality['dropped_rows']}")

    print("\nRows by label:")
    print(df["label"].value_counts().reindex(LABEL_MAP.keys(), fill_value=0))

    print("\nRows by file:")
    for file_name, row_count in quality["file_row_counts"].items():
        print(f"  {file_name}: {row_count}")

    print("\nEstimated sampling rate by file:")
    for file_name, rate in quality["file_sampling_rates"].items():
        if np.isnan(rate):
            print(f"  {file_name}: could not estimate")
        else:
            print(f"  {file_name}: {rate:.2f} Hz")

    print("\nNaN count by column before required-row drop:")
    print(quality["nan_counts"].sort_index())


def create_windows_from_dataframe(
    df,
    window_size=WINDOW_SIZE,
    hop_size=HOP_SIZE,
    transition_margin_rows=0,
):
    """
    Create windows only inside contiguous same-label regions.

    This avoids windows crossing transition rows where posture labels change.
    transition_margin_rows also removes rows near each label-change boundary because
    the phone may still be moving while the user settles into the next posture.
    Output X shape is (num_windows, 100, 6), y shape is (num_windows,).
    """
    X_windows = []
    y_windows = []
    source_files = []
    session_ids = []

    # Process each recording independently. A session never leaks into another session.
    for (source_file, session_id), session_df in df.groupby(["source_file", "session_id"], sort=False):
        session_df = session_df.sort_values("timestamp_ns").copy()

        # segment_id increments whenever the label changes.
        label_changed = session_df["label"].ne(session_df["label"].shift())
        session_df["segment_id"] = label_changed.cumsum()

        for (_, label), group in session_df.groupby(["segment_id", "label"], sort=False):
            if transition_margin_rows > 0:
                # Drop unstable rows near posture transitions. If a segment becomes
                # too short, it simply contributes no windows.
                group = group.iloc[transition_margin_rows:-transition_margin_rows]

            values = group[SENSOR_COLS].to_numpy(dtype=np.float32)

            if len(values) < window_size:
                continue

            for start in range(0, len(values) - window_size + 1, hop_size):
                end = start + window_size
                X_windows.append(values[start:end])
                y_windows.append(LABEL_MAP[label])
                source_files.append(source_file)
                session_ids.append(session_id)

    X = np.asarray(X_windows, dtype=np.float32)
    y = np.asarray(y_windows, dtype=np.int64)
    source_files = np.asarray(source_files)
    session_ids = np.asarray(session_ids)

    return X, y, source_files, session_ids


def split_by_source_file(X, y, source_files, sorted_csv_files):
    """Use file/session split: first 12 train, next 2 validation, last 2 test."""
    sorted_file_names = [path.name for path in sorted_csv_files]

    if len(sorted_file_names) < 3:
        raise ValueError("At least 3 CSV files are needed for train/validation/test split.")

    if len(sorted_file_names) >= 16:
        train_files = sorted_file_names[:12]
        val_files = sorted_file_names[12:14]
        test_files = sorted_file_names[14:16]
    else:
        train_end = max(1, int(len(sorted_file_names) * 0.75))
        val_end = max(train_end + 1, int(len(sorted_file_names) * 0.875))
        val_end = min(val_end, len(sorted_file_names) - 1)
        train_files = sorted_file_names[:train_end]
        val_files = sorted_file_names[train_end:val_end]
        test_files = sorted_file_names[val_end:]

    train_mask = np.isin(source_files, train_files)
    val_mask = np.isin(source_files, val_files)
    test_mask = np.isin(source_files, test_files)

    splits = {
        "train": (X[train_mask], y[train_mask], train_files),
        "val": (X[val_mask], y[val_mask], val_files),
        "test": (X[test_mask], y[test_mask], test_files),
    }

    for split_name, (split_X, _, split_files) in splits.items():
        if len(split_X) == 0:
            raise ValueError(
                f"{split_name} split has 0 windows.\n"
                f"Files assigned to this split: {split_files}\n"
                "Check whether each file has at least 100 consecutive rows with the same label."
            )

    print("\n=== Source File Split ===")
    for split_name, (split_X, split_y, split_files) in splits.items():
        print(f"\n{split_name.upper()} files ({len(split_files)}):")
        for file_name in split_files:
            file_windows = int(np.sum(source_files == file_name))
            print(f"  {file_name}: {file_windows} windows")
        print(f"{split_name.upper()} total windows: {len(split_X)}")
        print(
            pd.Series(split_y)
            .map(INV_LABEL_MAP)
            .value_counts()
            .reindex(LABEL_MAP.keys(), fill_value=0)
        )

    return splits


def fit_normalize_train_only(X_train, X_val, X_test):
    """
    Fit mean/std only on train data, then apply the same values to val/test.
    Mean and std are calculated independently for each sensor channel.
    """
    train_flat = X_train.reshape(-1, X_train.shape[-1])
    mean = train_flat.mean(axis=0)
    std = train_flat.std(axis=0)

    # Avoid division by zero if a channel is constant.
    std = np.where(std < 1e-8, 1.0, std)

    def transform(X):
        return ((X - mean) / std).astype(np.float32)

    return transform(X_train), transform(X_val), transform(X_test), mean, std


def save_normalization(model_dir, mean, std, transition_margin_sec, transition_margin_rows):
    normalization = {
        "sensor_cols": SENSOR_COLS,
        "mean": mean.tolist(),
        "std": std.tolist(),
        "window_size": WINDOW_SIZE,
        "hop_size": HOP_SIZE,
        "transition_margin_sec": transition_margin_sec,
        "transition_margin_rows": transition_margin_rows,
        "sampling_rate": SAMPLING_RATE,
        "label_map": LABEL_MAP,
    }

    output_path = model_dir / "normalization.json"
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(normalization, f, indent=2, ensure_ascii=False)

    print(f"\nSaved normalization stats: {output_path}")


def build_1d_cnn(input_shape=(WINDOW_SIZE, len(SENSOR_COLS)), num_classes=3):
    inputs = tf.keras.Input(shape=input_shape)

    x = tf.keras.layers.Conv1D(32, kernel_size=5, padding="same", activation="relu")(inputs)
    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.Conv1D(64, kernel_size=5, padding="same", activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling1D(pool_size=2)(x)

    x = tf.keras.layers.Conv1D(128, kernel_size=3, padding="same", activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)

    x = tf.keras.layers.Dense(64, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.3)(x)

    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def save_training_history(history, model_dir):
    history_path = model_dir / "training_history.csv"
    pd.DataFrame(history.history).to_csv(history_path, index=False)
    print(f"Saved training history: {history_path}")


def evaluate_and_save_reports(model, X_test, y_test, model_dir):
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print("\n=== Test Evaluation ===")
    print(f"Test loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_accuracy:.4f}")

    probabilities = model.predict(X_test, verbose=0)
    y_pred = np.argmax(probabilities, axis=1)

    target_names = [INV_LABEL_MAP[i] for i in range(len(LABEL_MAP))]
    report_text = classification_report(
        y_test,
        y_pred,
        labels=list(range(len(LABEL_MAP))),
        target_names=target_names,
        digits=4,
        zero_division=0,
    )
    cm = confusion_matrix(y_test, y_pred, labels=list(range(len(LABEL_MAP))))
    recalls = recall_score(
        y_test,
        y_pred,
        labels=list(range(len(LABEL_MAP))),
        average=None,
        zero_division=0,
    )

    print("\nClassification report:")
    print(report_text)

    print("Confusion matrix:")
    print(pd.DataFrame(cm, index=target_names, columns=target_names))

    print("\nRecall by class:")
    for class_id, recall in enumerate(recalls):
        print(f"  {INV_LABEL_MAP[class_id]}: {recall:.4f}")

    severe_id = LABEL_MAP["Severe Looking-Down"]
    print(f"\n*** Severe Looking-Down recall: {recalls[severe_id]:.4f} ***")

    report_path = model_dir / "classification_report.txt"
    with report_path.open("w", encoding="utf-8") as f:
        f.write(f"Test loss: {test_loss:.6f}\n")
        f.write(f"Test accuracy: {test_accuracy:.6f}\n\n")
        f.write(report_text)
        f.write("\nRecall by class:\n")
        for class_id, recall in enumerate(recalls):
            f.write(f"{INV_LABEL_MAP[class_id]}: {recall:.6f}\n")
        f.write(f"\nSevere Looking-Down recall: {recalls[severe_id]:.6f}\n")

    cm_path = model_dir / "confusion_matrix.csv"
    pd.DataFrame(cm, index=target_names, columns=target_names).to_csv(cm_path)

    print(f"\nSaved classification report: {report_path}")
    print(f"Saved confusion matrix: {cm_path}")


def convert_to_tflite_float32(model, model_dir):
    output_path = model_dir / "posture_1dcnn_float32.tflite"
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    output_path.write_bytes(tflite_model)
    print(f"Saved float32 TFLite model: {output_path}")


def convert_to_tflite_int8(model, X_train, model_dir):
    output_path = model_dir / "posture_1dcnn_int8.tflite"

    def representative_dataset():
        sample_count = min(200, len(X_train))
        for i in range(sample_count):
            yield [X_train[i : i + 1].astype(np.float32)]

    try:
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_dataset
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8

        tflite_model = converter.convert()
        output_path.write_bytes(tflite_model)
        print(f"Saved int8 TFLite model: {output_path}")
    except Exception as exc:
        print("\nINT8 TFLite conversion failed.")
        print(f"Reason: {exc}")
        print("Float32 TFLite model was still saved successfully.")


def print_final_usage(model_dir):
    expected_files = [
        "best_posture_1dcnn.keras",
        "posture_1dcnn_final.keras",
        "normalization.json",
        "training_history.csv",
        "confusion_matrix.csv",
        "classification_report.txt",
        "posture_1dcnn_float32.tflite",
        "posture_1dcnn_int8.tflite",
    ]

    print("\n=== How to run ===")
    print("cd PostureIMUData")
    print("python train_posture_1dcnn.py")
    print("\nGenerated files:")
    for file_name in expected_files:
        path = model_dir / file_name
        status = "created" if path.exists() else "not created"
        print(f"  {path} ({status})")


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)
    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    print("Loading CSV files...")
    df, quality = load_all_csv(data_dir)
    print_data_quality(df, quality)

    transition_margin_rows = int(round(args.transition_margin_sec * SAMPLING_RATE))
    print("\nCreating windows...")
    print(
        "Transition margin removal: "
        f"{args.transition_margin_sec:.2f} sec = {transition_margin_rows} rows "
        "from both ends of each same-label segment"
    )
    X, y, source_files, session_ids = create_windows_from_dataframe(
        df,
        transition_margin_rows=transition_margin_rows,
    )

    if len(X) == 0:
        raise ValueError(
            "No windows were created. Each sample needs 100 consecutive rows with the same label."
        )

    print(f"Generated windows: {len(X)}")
    print("Windows by label:")
    print(pd.Series(y).map(INV_LABEL_MAP).value_counts().reindex(LABEL_MAP.keys(), fill_value=0))
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print(f"Session metadata shape: {session_ids.shape}")

    splits = split_by_source_file(X, y, source_files, quality["csv_files"])
    X_train, y_train, _ = splits["train"]
    X_val, y_val, _ = splits["val"]
    X_test, y_test, _ = splits["test"]

    print("\nNormalizing sensor channels with train-set mean/std only...")
    X_train, X_val, X_test, mean, std = fit_normalize_train_only(X_train, X_val, X_test)
    save_normalization(model_dir, mean, std, args.transition_margin_sec, transition_margin_rows)

    print("\nBuilding model...")
    model = build_1d_cnn(input_shape=(WINDOW_SIZE, len(SENSOR_COLS)), num_classes=len(LABEL_MAP))
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=8,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            model_dir / "best_posture_1dcnn.keras",
            monitor="val_accuracy",
            save_best_only=True,
        ),
    ]

    print("\nTraining model...")
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=args.epochs,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1,
    )
    save_training_history(history, model_dir)

    final_model_path = model_dir / "posture_1dcnn_final.keras"
    model.save(final_model_path)
    print(f"Saved final Keras model: {final_model_path}")

    evaluate_and_save_reports(model, X_test, y_test, model_dir)

    print("\nConverting trained model to TensorFlow Lite...")
    convert_to_tflite_float32(model, model_dir)
    convert_to_tflite_int8(model, X_train, model_dir)

    print_final_usage(model_dir)


if __name__ == "__main__":
    main()

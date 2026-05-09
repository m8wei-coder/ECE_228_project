import os
import pickle
import argparse
import joblib
import numpy as np
import pandas as pd
import torch
from scipy.signal import medfilt

from config import get_config
from data_preprocessing import CMAPSSDataset
from network import CMAPSS_LSTM


def calculate_metrics(y_pred, y_true):
    """
    Calculate RMSE and PHM score.
    y_pred and y_true should be torch tensors.
    """
    error = y_pred - y_true

    rmse = torch.sqrt(torch.mean(error ** 2)).item()

    score = 0.0
    for e in error:
        e = e.item()
        if e < 0:
            score += np.exp(-e / 13.0) - 1.0
        else:
            score += np.exp(e / 10.0) - 1.0

    return rmse, score


def load_true_rul(rul_file):
    """
    Load true RUL file.
    Compatible with both:
    1. CSV with header, e.g. true_rul
    2. CSV / txt without header
    """
    try:
        rul_df = pd.read_csv(rul_file)
        y_true = rul_df.iloc[:, 0].astype(np.float32).values
    except ValueError:
        rul_df = pd.read_csv(rul_file, header=None)
        y_true = rul_df.iloc[:, 0].astype(np.float32).values

    return y_true


def load_saved_preprocessing_artifact(scaler_path):
    """
    Load saved preprocessing artifact from training stage.
    Required keys:
        kmeans
        scalers
        features
        initial_rul
    """
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(
            f"Preprocessing artifact not found: {scaler_path}. "
            f"Please train the model first."
        )

    saved = joblib.load(scaler_path)

    required_keys = ["kmeans", "scalers", "features", "initial_rul"]
    missing_keys = [key for key in required_keys if key not in saved]

    if missing_keys:
        raise KeyError(
            f"Missing keys in preprocessing artifact: {missing_keys}. "
            f"Please make sure training stage saved kmeans, scalers, features, and initial_rul."
        )

    return saved


def add_absolute_rul_for_train(df):
    """
    For training data, full run-to-failure trajectories are available.
    RUL_absolute = max_cycle_of_engine - current_cycle.
    """
    rul = pd.DataFrame(
        df.groupby("unit_number")["time_cycles"].max()
    ).reset_index()

    rul.columns = ["unit_number", "max_cycle"]

    df = df.merge(rul, on="unit_number", how="left")
    df["RUL_absolute"] = df["max_cycle"] - df["time_cycles"]
    df.drop("max_cycle", axis=1, inplace=True)

    return df


def apply_saved_preprocessing(df, saved):
    """
    Apply saved preprocessing to raw dataframe.

    Important:
    - settings are only used for kmeans.predict()
    - features are only sensor columns saved during training
    - scalers are applied condition-wise to sensor features only
    """
    settings_cols = ["setting_1", "setting_2", "setting_3"]

    kmeans = saved["kmeans"]
    scalers = saved["scalers"]
    features = saved["features"]

    median_window = saved.get("median_window", 5)

    df = df.copy()

    # Median filter on selected sensor features only
    for eng_id in df["unit_number"].unique():
        idx = df["unit_number"] == eng_id
        for col in features:
            df.loc[idx, col] = medfilt(
                df.loc[idx, col].values,
                kernel_size=median_window
            )

    df[features] = df[features].astype(float)

    # KMeans condition prediction uses settings only
    df["condition"] = kmeans.predict(df[settings_cols])

    # Condition-wise scaling on selected sensor features only
    for condition in range(6):
        idx = df["condition"] == condition

        if idx.sum() == 0:
            continue

        if condition not in scalers:
            raise KeyError(
                f"Scaler for condition {condition} not found in artifact."
            )

        df.loc[idx, features] = scalers[condition].transform(
            df.loc[idx, features]
        )

    df.drop("condition", axis=1, inplace=True)

    return df, features


def prepare_train_data_from_saved_artifact(train_file, sequence_length, scaler_path):
    """
    Prepare X_train and y_train using saved preprocessing artifact.
    This function does NOT refit KMeans or StandardScaler.
    """
    saved = load_saved_preprocessing_artifact(scaler_path)

    features = saved["features"]
    initial_rul = saved["initial_rul"]

    df_train = pd.read_csv(train_file)

    df_train = add_absolute_rul_for_train(df_train)

    df_train, features = apply_saved_preprocessing(df_train, saved)

    # Use saved initial RUL from training stage
    df_train["RUL_piecewise"] = df_train["RUL_absolute"].clip(
        upper=initial_rul
    )

    train_dataset = CMAPSSDataset(
        df_train,
        sequence_length=sequence_length,
        features=features
    )

    X_train = train_dataset.sequences.astype(np.float32)
    y_train = train_dataset.labels.astype(np.float32)

    return X_train, y_train, features, initial_rul


def prepare_test_data_from_saved_artifact(test_file, rul_file, sequence_length, scaler_path):
    """
    Prepare X_test and y_test using saved preprocessing artifact.
    One sequence is generated for each test engine using the last sequence_length cycles.
    """
    saved = load_saved_preprocessing_artifact(scaler_path)

    features = saved["features"]
    initial_rul = saved["initial_rul"]

    df_test = pd.read_csv(test_file)

    df_test, features = apply_saved_preprocessing(df_test, saved)

    X_test = []

    for eng_id in df_test["unit_number"].unique():
        eng_data = df_test[df_test["unit_number"] == eng_id].sort_values(
            "time_cycles"
        )

        X = eng_data[features].values.astype(np.float32)

        if len(X) >= sequence_length:
            X_test.append(X[-sequence_length:])
        else:
            pad_len = sequence_length - len(X)
            pad = np.repeat(X[:1], pad_len, axis=0)
            X_test.append(np.vstack([pad, X]))

    X_test = np.array(X_test, dtype=np.float32)

    y_test = load_true_rul(rul_file)

    input_size = len(features)

    return X_test, y_test, input_size, initial_rul, features


def compute_corr_matrix(train_file, features, output_dir):
    """
    Compute Pearson correlation matrix using raw train data and RUL_absolute.
    This is mainly for graph construction / edge weighting.
    """
    df = pd.read_csv(train_file)

    df = add_absolute_rul_for_train(df)

    corr_cols = features + ["RUL_absolute"]

    corr_matrix = df[corr_cols].corr(method="pearson")

    np.save(
        os.path.join(output_dir, "corr_matrix.npy"),
        corr_matrix.values
    )

    corr_matrix.to_csv(
        os.path.join(output_dir, "corr_matrix.csv"),
        index=True
    )

    return corr_matrix


def export_train_arrays(cfg, output_dir):
    X_train, y_train, feature_cols, initial_rul = prepare_train_data_from_saved_artifact(
        train_file=cfg["train_file"],
        sequence_length=cfg["seq_length"],
        scaler_path=cfg["scaler_path"]
    )

    np.save(os.path.join(output_dir, "X_train.npy"), X_train)
    np.save(os.path.join(output_dir, "y_train.npy"), y_train)

    with open(os.path.join(output_dir, "retained_sensors.pkl"), "wb") as f:
        pickle.dump(feature_cols, f)

    return X_train, y_train, feature_cols, initial_rul


def export_test_arrays(cfg, output_dir):
    X_test, y_test, input_size, initial_rul, feature_cols = prepare_test_data_from_saved_artifact(
        test_file=cfg["test_file"],
        rul_file=cfg["rul_file"],
        sequence_length=cfg["seq_length"],
        scaler_path=cfg["scaler_path"]
    )

    np.save(os.path.join(output_dir, "X_test.npy"), X_test)
    np.save(os.path.join(output_dir, "y_test.npy"), y_test)

    return X_test, y_test, input_size, initial_rul, feature_cols


def evaluate_baseline_lstm(cfg, X_test, y_test, input_size, initial_rul, output_dir):
    """
    Load trained baseline LSTM and evaluate RMSE / Score on test set.
    """
    if not os.path.exists(cfg["model_path"]):
        print(f"Model file not found: {cfg['model_path']}")
        print("Skipped baseline LSTM evaluation.")
        return None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CMAPSS_LSTM(
        input_size=input_size,
        hidden_size=cfg["hidden_size"],
        num_layers=cfg["num_layers"],
        dropout_prob=cfg["dropout"],
    ).to(device)

    model.load_state_dict(
        torch.load(cfg["model_path"], map_location=device)
    )

    model.eval()

    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).to(device)

    with torch.no_grad():
        y_pred = model(X_test_tensor).squeeze()

    # Use training-stage initial RUL as prediction upper bound
    y_pred = torch.clamp(
        y_pred,
        min=0.0,
        max=float(initial_rul)
    )

    y_test_tensor = torch.clamp(y_test_tensor, min=0.0, max=float(initial_rul))
    
    rmse, score = calculate_metrics(y_pred, y_test_tensor)

    y_pred_np = y_pred.detach().cpu().numpy()

    np.save(
        os.path.join(output_dir, "y_pred_lstm.npy"),
        y_pred_np
    )

    metrics = {
        "rmse": float(rmse),
        "score": float(score),
        "model_path": cfg["model_path"],
        "initial_rul_upper_bound": float(initial_rul),
    }

    with open(os.path.join(output_dir, "baseline_lstm_metrics.pkl"), "wb") as f:
        pickle.dump(metrics, f)

    pd.DataFrame([metrics]).to_csv(
        os.path.join(output_dir, "baseline_lstm_metrics.csv"),
        index=False
    )

    return metrics


def export_metadata(cfg, dataset_name, feature_cols, initial_rul, output_dir):
    metadata = {
        "dataset": dataset_name,
        "train_file": cfg["train_file"],
        "test_file": cfg["test_file"],
        "rul_file": cfg["rul_file"],
        "features": feature_cols,
        "num_features": len(feature_cols),
        "initial_rul": initial_rul,
        "seq_length": cfg["seq_length"],
        "window_size": cfg["window_size"],
        "threshold": cfg["threshold"],
        "patience": cfg["patience"],
        "drop_cols": cfg["drop_cols"],
        "model_path": cfg["model_path"],
        "scaler_path": cfg["scaler_path"],
    }

    with open(os.path.join(output_dir, "metadata.pkl"), "wb") as f:
        pickle.dump(metadata, f)

    metadata_for_csv = {}

    for key, value in metadata.items():
        if isinstance(value, list):
            metadata_for_csv[key] = ",".join(map(str, value))
        else:
            metadata_for_csv[key] = value

    pd.DataFrame([metadata_for_csv]).to_csv(
        os.path.join(output_dir, "metadata.csv"),
        index=False
    )

    return metadata


def main(dataset_name="FD001", evaluate_model=True):
    dataset_name = dataset_name.upper()
    cfg = get_config(dataset_name)

    output_dir = os.path.join("./exports", dataset_name)
    os.makedirs(output_dir, exist_ok=True)

    print(f"Exporting artifacts for {dataset_name}")
    print(f"Using saved preprocessing artifact: {cfg['scaler_path']}")
    print(f"Output directory: {output_dir}")

    X_train, y_train, feature_cols, initial_rul = export_train_arrays(
        cfg,
        output_dir
    )

    X_test, y_test, input_size, test_initial_rul, test_feature_cols = export_test_arrays(
        cfg,
        output_dir
    )

    if feature_cols != test_feature_cols:
        raise ValueError(
            "Train and test feature lists do not match.\n"
            f"Train features: {feature_cols}\n"
            f"Test features: {test_feature_cols}"
        )

    corr_matrix = compute_corr_matrix(
        train_file=cfg["train_file"],
        features=feature_cols,
        output_dir=output_dir
    )

    export_metadata(
        cfg=cfg,
        dataset_name=dataset_name,
        feature_cols=feature_cols,
        initial_rul=initial_rul,
        output_dir=output_dir
    )

    print("\nExport summary:")
    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_test shape: {y_test.shape}")
    print(f"Retained sensors ({len(feature_cols)}): {feature_cols}")
    print(f"Initial RUL upper bound: {initial_rul}")
    print(f"Correlation matrix shape: {corr_matrix.shape}")

    if evaluate_model:
        metrics = evaluate_baseline_lstm(
            cfg=cfg,
            X_test=X_test,
            y_test=y_test,
            input_size=input_size,
            initial_rul=test_initial_rul,
            output_dir=output_dir
        )

        if metrics is not None:
            print("\nBaseline LSTM metrics:")
            print(f"RMSE: {metrics['rmse']:.4f}")
            print(f"Score: {metrics['score']:.4e}")

    print("\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        type=str,
        default="FD001",
        choices=["FD001", "FD002", "FD003", "FD004"],
        help="Dataset name."
    )

    parser.add_argument(
        "--no-eval",
        action="store_true",
        help="Skip baseline LSTM evaluation."
    )

    args = parser.parse_args()

    main(
        dataset_name=args.dataset,
        evaluate_model=not args.no_eval
    )
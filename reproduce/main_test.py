import pandas as pd
import numpy as np
from scipy.signal import medfilt
from sklearn.preprocessing import StandardScaler
import torch
from network import *
from config import get_config
import joblib
import argparse

def prepare_test_data(test_file, rul_file, sequence_length, scaler_path):
    df_test = pd.read_csv(test_file)

    saved = joblib.load(scaler_path)
    kmeans = saved['kmeans']
    scalers = saved['scalers']
    features = saved['features']
    initial_rul = saved["initial_rul"]

    settings_cols = ['setting_1', 'setting_2', 'setting_3']

    df_test[features] = df_test[features].astype(float)

    # KMeans uses settings only
    df_test['condition'] = kmeans.predict(df_test[settings_cols])

    # Scalers transform sensor features only
    for condition in range(6):
        idx = df_test['condition'] == condition

        if idx.sum() == 0:
            continue

        df_test.loc[idx, features] = scalers[condition].transform(
            df_test.loc[idx, features]
        )

    df_test.drop('condition', axis=1, inplace=True)

    X_test = []

    for eng_id in df_test['unit_number'].unique():
        eng_data = df_test[df_test['unit_number'] == eng_id].sort_values('time_cycles')
        X = eng_data[features].values

        if len(X) >= sequence_length:
            X_test.append(X[-sequence_length:])
        else:
            pad_len = sequence_length - len(X)
            pad = np.repeat(X[:1], pad_len, axis=0)
            X_test.append(np.vstack([pad, X]))

    X_test = np.array(X_test)

    rul_df = pd.read_csv(rul_file)
    y_true = rul_df.iloc[:, 0].astype(np.float32).values

    input_size = len(features)

    return X_test, y_true, input_size, initial_rul

def main(dataset):
    cfg = get_config(dataset)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_test, y_true, input_size, initial_rul = prepare_test_data(
        test_file=cfg["test_file"],
        rul_file=cfg["rul_file"],
        sequence_length=cfg["seq_length"],
        scaler_path=cfg["scaler_path"]
    )

    model = CMAPSS_LSTM(
        input_size=input_size,
        hidden_size=cfg["hidden_size"],
        num_layers=cfg["num_layers"],
        dropout_prob=cfg["dropout"]
    ).to(device)

    model.load_state_dict(torch.load(cfg["model_path"], map_location=device))
    model.eval()

    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_true_tensor = torch.tensor(y_true, dtype=torch.float32).to(device)

    with torch.no_grad():
        y_pred = model(X_test_tensor).squeeze()

    y_pred = torch.clamp(y_pred, min=0.0, max=float(initial_rul))
    y_true_tensor = torch.clamp(y_true_tensor, min=0.0, max=float(initial_rul))
    rmse, score = calculate_metrics(y_pred, y_true_tensor)

    print(f"Dataset: {dataset}")
    print(f"Test RMSE: {rmse:.2f}")
    print(f"Test Score: {score:.2E}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="FD001")
    args = parser.parse_args()

    main(args.dataset)
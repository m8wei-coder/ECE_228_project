import pandas as pd
import numpy as np
from scipy.signal import medfilt
from sklearn.preprocessing import StandardScaler
from scipy.ndimage import median_filter
from sklearn.cluster import KMeans
import torch
from torch.utils.data import Dataset, DataLoader
import joblib

def update_preprocessing_metadata(scaler_path, **kwargs):
    saved = joblib.load(scaler_path)
    saved.update(kwargs)
    joblib.dump(saved, scaler_path)

def load_and_filter_data(file_path, drop_cols, scaler_path):
    df = pd.read_csv(file_path)

    # Calculate absolute RUL
    rul = pd.DataFrame(df.groupby('unit_number')['time_cycles'].max()).reset_index()
    rul.columns = ['unit_number', 'max_cycle']
    df = df.merge(rul, on='unit_number', how='left')
    df['RUL_absolute'] = df['max_cycle'] - df['time_cycles']
    df.drop('max_cycle', axis=1, inplace=True)

    settings_cols = ['setting_1', 'setting_2', 'setting_3']

    sensor_cols = [
        c for c in df.columns
        if c.startswith('sensor_')
    ]

    selected_sensor_cols = [
        c for c in sensor_cols
        if c not in drop_cols
    ]

    # Median filter only on selected sensors
    window_size = 5
    for eng_id in df['unit_number'].unique():
        idx = df['unit_number'] == eng_id
        for col in selected_sensor_cols:
            df.loc[idx, col] = medfilt(
                df.loc[idx, col].values,
                kernel_size=window_size
            )

    df[selected_sensor_cols] = df[selected_sensor_cols].astype(float)

    # KMeans uses settings only
    kmeans = KMeans(n_clusters=6, random_state=42, n_init=10)
    df['condition'] = kmeans.fit_predict(df[settings_cols])

    # Standardize sensors condition-wise
    scalers = {}
    for condition in range(6):
        idx = df['condition'] == condition

        if idx.sum() == 0:
            continue

        scaler = StandardScaler()
        df.loc[idx, selected_sensor_cols] = scaler.fit_transform(
            df.loc[idx, selected_sensor_cols]
        )
        scalers[condition] = scaler

    # Save preprocessing objects and feature list
    joblib.dump(
        {
            'kmeans': kmeans,
            'scalers': scalers,
            'features': selected_sensor_cols,
            'drop_cols': drop_cols,
        },
        scaler_path
    )

    df.drop('condition', axis=1, inplace=True)

    return df, selected_sensor_cols

def calculate_piecewise_rul(df, features, w=12, th=0.2, patience=1):
    irul_list = []
    irul_records = []

    for eng_id in df['unit_number'].unique():
        eng_data = df[df['unit_number'] == eng_id].sort_values('time_cycles')
        n = len(eng_data)
        g = n // w

        if g < 2:
            irul = eng_data['RUL_absolute'].max()
            irul_list.append(irul)
            continue

        X = eng_data[features].values

        centroids = np.array([
            np.mean(X[i*w : (i+1)*w], axis=0)
            for i in range(g)
        ])

        found_knee = False
        consecutive_count = 0

        for i in range(1, g):
            prev_mean = np.mean(centroids[:i], axis=0)

            dist = np.linalg.norm(prev_mean - centroids[i]) / (
                np.linalg.norm(prev_mean) + 1e-8
            )

            if dist > th:
                consecutive_count += 1
            else:
                consecutive_count = 0

            if consecutive_count >= patience:
                # backtracking to the first window in the confirmed segment
                knee_i = i - patience + 1
                knee_point_cycle = (knee_i + 1) * w

                max_cycle = eng_data['time_cycles'].max()
                irul = max_cycle - knee_point_cycle

                irul_list.append(irul)
                irul_records.append({
                    "engine": eng_id,
                    "irul": irul,
                    "max_cycle": max_cycle,
                    "knee_i": knee_i,
                    "knee_cycle": knee_point_cycle,
                    "dist": dist,
                    "consecutive_count": consecutive_count
                })

                found_knee = True
                break

        if not found_knee:
            irul = eng_data['RUL_absolute'].max()
            irul_list.append(irul)

    global_initial_rul = int(np.min(irul_list))

    print("IRUL stats:")
    print("min:", np.min(irul_list))
    print("percentiles:", np.percentile(irul_list, [0, 5, 10, 25, 50]))
    print(f"Calculated Global Initial RUL: {global_initial_rul}")

    df['RUL_piecewise'] = df['RUL_absolute'].clip(upper=global_initial_rul)

    return df, global_initial_rul

class CMAPSSDataset(Dataset):
    def __init__(self, df, sequence_length, features):
        self.sequence_length = sequence_length
        self.features = features
        self.sequences, self.labels = self._generate_sequences(df)

    def _generate_sequences(self, df):
        seqs = []
        labels = []
        
        for eng_id in df['unit_number'].unique():
            eng_data = df[df['unit_number'] == eng_id].sort_values('time_cycles')
            X = eng_data[self.features].values
            y = eng_data['RUL_piecewise'].values
            
            for i in range(len(eng_data) - self.sequence_length + 1):
                seqs.append(X[i : i + self.sequence_length])
                labels.append(y[i + self.sequence_length - 1])
            
        return np.array(seqs), np.array(labels)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq_tensor = torch.tensor(self.sequences[idx], dtype=torch.float32)
        label_tensor = torch.tensor(self.labels[idx], dtype=torch.float32)
        return seq_tensor, label_tensor
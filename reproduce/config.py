DATASET_CONFIGS = {
    "FD001": {
        "train_file": "./csv/train/train_FD001.csv",
        "test_file": "./csv/test/test_FD001.csv",
        "rul_file": "./csv/rul/RUL_FD001.csv",

        "drop_cols": [
            "sensor_1", "sensor_5", "sensor_6", "sensor_10",
            "sensor_16", "sensor_18", "sensor_19"
        ],

        "window_size": 12,
        "threshold": 0.2,
        "patience": 1,

        "seq_length": 30,
        "batch_size": 15,
        "learning_rate": 2e-3,
        "epochs": 150,

        "hidden_size": 60,
        "num_layers": 4,
        "dropout": 0.1,

        "scaler_path": "./logs/cmapss_condition_scalers_fd001.gz",
        "model_path": "./logs/cmapss_lstm_fd001.pth",
    },

    "FD002": {
        "train_file": "./csv/train/train_FD002.csv",
        "test_file": "./csv/test/test_FD002.csv",
        "rul_file": "./csv/rul/RUL_FD002.csv",

        "drop_cols": [],

        "window_size": 12,
        "threshold": 0.2,
        "patience": 1,

        "seq_length": 30,
        "batch_size": 15,
        "learning_rate": 2e-3,
        "epochs": 150,

        "hidden_size": 60,
        "num_layers": 4,
        "dropout": 0.1,

        "scaler_path": "./logs/cmapss_condition_scalers_fd002.gz",
        "model_path": "./logs/cmapss_lstm_fd002.pth",
    },

    "FD003": {
        "train_file": "./csv/train/train_FD003.csv",
        "test_file": "./csv/test/test_FD003.csv",
        "rul_file": "./csv/rul/RUL_FD003.csv",

        "drop_cols": [
            "sensor_1", "sensor_5",
            "sensor_16", "sensor_18", "sensor_19"
        ],

        "window_size": 12,
        "threshold": 0.2,
        "patience": 2,

        "seq_length": 30,
        "batch_size": 20,
        "learning_rate": 1e-3,
        "epochs": 250,

        "hidden_size": 90,
        "num_layers": 6,
        "dropout": 0.1,

        "scaler_path": "./logs/cmapss_condition_scalers_fd003.gz",
        "model_path": "./logs/cmapss_lstm_fd003.pth",
    },

    "FD004": {
        "train_file": "./csv/train/train_FD004.csv",
        "test_file": "./csv/test/test_FD004.csv",
        "rul_file": "./csv/rul/RUL_FD004.csv",

        "drop_cols": [],

        "window_size": 12,
        "threshold": 0.3,
        "patience": 3,

        "seq_length": 30,
        "batch_size": 10,
        "learning_rate": 1e-3,
        "epochs": 20,

        "hidden_size": 30,
        "num_layers": 2,
        "dropout": 0.1,

        "scaler_path": "./logs/cmapss_condition_scalers_fd004.gz",
        "model_path": "./logs/cmapss_lstm_fd004.pth",
    },
}


def get_config(dataset_name):
    dataset_name = dataset_name.upper()

    if dataset_name not in DATASET_CONFIGS:
        raise ValueError(
            f"Unknown dataset: {dataset_name}. "
            f"Choose from {list(DATASET_CONFIGS.keys())}."
        )

    return DATASET_CONFIGS[dataset_name]
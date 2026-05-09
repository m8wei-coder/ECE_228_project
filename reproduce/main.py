import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from config import get_config
from data_preprocessing import *
from network import *
import argparse

def main(dataset):
    cfg = get_config(dataset)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    print(f"--- Stage 1: Data Preprocessing ---")

    df_train, feature_cols = load_and_filter_data(
        file_path=cfg["train_file"],
        drop_cols=cfg["drop_cols"],
        scaler_path=cfg["scaler_path"]
    )

    df_train, initial_rul = calculate_piecewise_rul(
        df_train,
        feature_cols,
        w=cfg["window_size"],
        th=cfg["threshold"],
        patience=cfg["patience"]
    )

    update_preprocessing_metadata(
        cfg["scaler_path"],
        initial_rul=initial_rul,
        window_size=cfg["window_size"],
        threshold=cfg["threshold"],
        patience=cfg["patience"]
    )
    
    input_size = len(feature_cols)
    print(f"Input feature size: {input_size}")
    print(f"Initial RUL: {initial_rul}\n")

    print("--- Stage 2: Building DataLoader ---")

    train_dataset = CMAPSSDataset(
        df_train,
        sequence_length=cfg["seq_length"],
        features=feature_cols
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg["batch_size"],
        shuffle=True,
        drop_last=True
    )

    print(f"Total training sequences generated: {len(train_dataset)}\n")

    print("--- Stage 3: Initializing Model ---")

    model = CMAPSS_LSTM(
        input_size=input_size,
        hidden_size=cfg["hidden_size"],
        num_layers=cfg["num_layers"],
        dropout_prob=cfg["dropout"]
    ).to(device)

    criterion = nn.MSELoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=cfg["learning_rate"],
        weight_decay=1e-5
    )

    print("--- Stage 4: Starting Training ---")

    best_loss = float('inf')

    for epoch in range(cfg["epochs"]):
        model.train()
        train_loss = 0.0

        all_preds = []
        all_labels = []

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()

            y_pred = model(X_batch).squeeze()
            loss = criterion(y_pred, y_batch)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            all_preds.append(y_pred.detach())
            all_labels.append(y_batch.detach())

        avg_loss = train_loss / len(train_loader)

        if (epoch + 1) % 10 == 0:
            preds_tensor = torch.cat(all_preds)
            labels_tensor = torch.cat(all_labels)

            rmse, score = calculate_metrics(preds_tensor, labels_tensor)

            print(
                f"Epoch [{epoch+1:03d}/{cfg['epochs']}] | "
                f"MSE Loss: {avg_loss:.2f} | "
                f"RMSE: {rmse:.2f} | "
                f"Score: {score:.2E}"
            )

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), cfg["model_path"])

    print("\nTraining complete!")
    print(f"Best model saved to: {cfg['model_path']}")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="FD001")
    args = parser.parse_args()

    main(args.dataset)
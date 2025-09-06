import argparse
import os
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
import mlflow
import random

EXPERIMENT_NAME = "sepsis_challenge"

def main():
    # Reproducibilidad
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Argumentos
    parser = argparse.ArgumentParser(description="Entrenamiento LSTM para Sepsis")
    parser.add_argument('--sample_frac', type=float, default=1.0, help='Porcentaje del dataset de entrenamiento a usar (0.0-1.0)')
    parser.add_argument('--mlflow_uri', type=str, default="http://54.86.215.8:8050", help='URI de MLflow')
    parser.add_argument('--hidden_size', type=int, default=128)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--num_epochs', type=int, default=7)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=64)
    args = parser.parse_args()

    # Nombre de la corrida dinámico
    RUN_NAME = f"lstm_ep-{args.num_epochs}_lr-{args.learning_rate}_size-{int(args.sample_frac*100)}pct"

    # Cargar datos procesados
    processed_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')
    X_train = np.load(os.path.join(processed_dir, 'X_train.npy'), mmap_mode='r')
    y_train = np.load(os.path.join(processed_dir, 'y_train.npy'), mmap_mode='r')
    X_val = np.load(os.path.join(processed_dir, 'X_val.npy'), mmap_mode='r')
    y_val = np.load(os.path.join(processed_dir, 'y_val.npy'), mmap_mode='r')

    num_samples = int(X_train.shape[0] * args.sample_frac)
    X_train_small = np.array(X_train[:num_samples])
    y_train_small = np.array(y_train[:num_samples])
    # Imprimite la cantidad de positivos en el conjunto reducido
    print(f"Positivos en entrenamiento reducido: {y_train_small.sum()} de {len(y_train_small)}")
    print(f"Positivos en validación: {y_val.sum()} de {len(y_val)}")

    X_train_tensor = torch.tensor(X_train_small, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train_small, dtype=torch.float32).unsqueeze(1)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)

    num_workers = os.cpu_count()
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=num_workers)

    class SepsisLSTM(nn.Module):
        def __init__(self, input_size, hidden_size, num_layers, output_size):
            super(SepsisLSTM, self).__init__()
            self.hidden_size = hidden_size
            self.num_layers = num_layers
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
            self.fc = nn.Linear(hidden_size, output_size)
            # self.sigmoid = nn.Sigmoid()  <-- LINEA ELIMINADA
        def forward(self, x):
            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
            c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
            out, _ = self.lstm(x, (h0, c0))
            out = self.fc(out[:, -1, :])
            # out = self.sigmoid(out)  <-- LINEA ELIMINADA
            return out

    input_size = X_train_tensor.shape[2]
    output_size = 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SepsisLSTM(input_size, args.hidden_size, args.num_layers, output_size).to(device)

    # Calcula el peso de la clase positiva
    pos_weight = torch.tensor([(len(y_train_small) - y_train_small.sum()) / y_train_small.sum()]).to(device)
    # Cambio de BCELoss por BCEWithLogitsLoss para mayor estabilidad numérica en desbalance de clases
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    # MLflow
    mlflow.set_tracking_uri(args.mlflow_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)
    with mlflow.start_run(run_name=RUN_NAME) as run:
        # Registrar todos los hiperparámetros
        mlflow.log_params({
            "hidden_size": args.hidden_size,
            "num_layers": args.num_layers,
            "epochs": args.num_epochs,
            "train_frac": args.sample_frac,
            "learning_rate": args.learning_rate,
            "batch_size": args.batch_size,
            "input_size": input_size,
            "output_size": output_size
        })

        for epoch in range(args.num_epochs):
            model.train()
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # Validación
            model.eval()
            all_preds = []
            all_labels = []
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                    outputs = model(batch_X)
                    all_preds.extend(outputs.cpu().numpy())
                    all_labels.extend(batch_y.cpu().numpy())
            # Convertir a binario para F1 y accuracy
            preds_bin = (np.array(all_preds) > 0.5).astype(int)
            labels_bin = (np.array(all_labels) > 0.5).astype(int)
            auc = roc_auc_score(all_labels, all_preds)
            f1 = f1_score(labels_bin, preds_bin)
            acc = accuracy_score(labels_bin, preds_bin)
            mlflow.log_metric("val_auc", auc, step=epoch)
            mlflow.log_metric("val_f1", f1, step=epoch)
            mlflow.log_metric("val_acc", acc, step=epoch)
            mlflow.log_metric("val_loss", loss.item(), step=epoch)
            print(f"Epoch [{epoch+1}/{args.num_epochs}], Loss: {loss.item():.4f}, Val AUC: {auc:.4f}, F1: {f1:.4f}, Acc: {acc:.4f}")
        
        # Reporte adicional
        print("Positivos reales en validación:", labels_bin.sum())
        print("Positivos predichos:", preds_bin.sum())

        # Guardar modelo
        model_path = os.path.join(os.path.dirname(__file__), '..', 'models', '{RUN_NAME}.pth')
        torch.save(model.state_dict(), model_path)
        mlflow.log_artifact(model_path)

if __name__ == "__main__":
    main()
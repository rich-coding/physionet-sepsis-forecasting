import argparse
import os
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
import mlflow
import random
# <-- OPTIMIZACIÓN: Importar las herramientas para Precisión Mixta Automática (AMP)
from torch.cuda.amp import GradScaler, autocast

EXPERIMENT_NAME = "sepsis_challenge"

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        # Usamos sigmoid en los logits para obtener probabilidades
        probs = torch.sigmoid(logits)
        
        # Aplanar etiquetas y probabilidades
        probs = probs.view(-1)
        targets = targets.view(-1)
        
        # Calcular la intersección y la suma
        intersection = (probs * targets).sum()                            
        dice_coeff = (2. * intersection + self.smooth) / (probs.sum() + targets.sum() + self.smooth)
        
        # Devolver el loss
        return 1 - dice_coeff

class SepsisLSTM(nn.Module):
        def __init__(self, input_size, hidden_size, num_layers, output_size):
            super(SepsisLSTM, self).__init__()
            self.hidden_size = hidden_size
            self.num_layers = num_layers
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
            self.fc = nn.Linear(hidden_size, output_size)
        def forward(self, x):
            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
            c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
            out, _ = self.lstm(x, (h0, c0))
            out = self.fc(out[:, -1, :])
            return out
        
def main():
    # --- Reproducibilidad vs Rendimiento ---
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # <-- OPTIMIZACIÓN: Se cambia la configuración para priorizar rendimiento sobre determinismo.
        # Esto permite a cuDNN encontrar los algoritmos más rápidos para tu hardware.
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

    # Argumentos
    parser = argparse.ArgumentParser(description="Entrenamiento LSTM para Sepsis (Optimizado para A100)")
    parser.add_argument('--sample_frac', type=float, default=1.0, help='Porcentaje del dataset de entrenamiento a usar (0.0-1.0)')
    parser.add_argument('--mlflow_uri', type=str, default="http://204.236.203.19:5000", help='URI de MLflow')
    parser.add_argument('--hidden_size', type=int, default=128)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--num_epochs', type=int, default=7)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    # <-- OPTIMIZACIÓN: Se recomienda aumentar el batch_size para saturar la GPU. Empieza con 256 o 512.
    parser.add_argument('--batch_size', type=int, default=256, help='Tamaño del lote. Aumentar para A100.')
    args = parser.parse_args()

    # Nombre dinámico de la ejecución 
    RUN_NAME = f"DL_lstm_optimized_ep-{args.num_epochs}_lr-{args.learning_rate}_bs-{args.batch_size}_size-{int(args.sample_frac*100)}pct"

    # Cargar datos procesados
    processed_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')
    X_train = np.load(os.path.join(processed_dir, 'X_train.npy'), mmap_mode='r')
    y_train = np.load(os.path.join(processed_dir, 'y_train.npy'), mmap_mode='r')
    X_val = np.load(os.path.join(processed_dir, 'X_val.npy'), mmap_mode='r')
    y_val = np.load(os.path.join(processed_dir, 'y_val.npy'), mmap_mode='r')

    num_samples = int(X_train.shape[0] * args.sample_frac)
    X_train_small = np.array(X_train[:num_samples])
    y_train_small = np.array(y_train[:num_samples])
    
    # ---- Balanced Sampling para manejar desbalanceo de clases------------
    class_sample_count = np.array([len(np.where(y_train_small == t)[0]) for t in np.unique(y_train_small)])
    weight = 1. / class_sample_count
    samples_weight = np.array([weight[t] for t in y_train_small.astype(int)])
    samples_weight = torch.from_numpy(samples_weight)

    sampler = WeightedRandomSampler(samples_weight, len(samples_weight))

    X_train_tensor = torch.tensor(X_train_small, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train_small, dtype=torch.float32).unsqueeze(1)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)

    num_workers = os.cpu_count()
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    
    # <-- OPTIMIZACIÓN: Se añade pin_memory=True para acelerar la transferencia de datos CPU -> GPU.
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    #Muestra cantidad de positivos y negativos en entrenamiento y validación
    print(f"Entrenamiento - Positivos: {y_train_small.sum()}, Negativos: {len(y_train_small) - y_train_small.sum()}")
    print(f"Validación - Positivos: {y_val.sum()}, Negativos: {len(y_val) - y_val.sum()}")

    input_size = X_train_tensor.shape[2]
    output_size = 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SepsisLSTM(input_size, args.hidden_size, args.num_layers, output_size).to(device)
    
    # <-- OPTIMIZACIÓN: Compilar el modelo con PyTorch 2.0+ para una gran aceleración.
    # PyTorch optimizará el grafo de computación del modelo.
    print("Compilando el modelo con torch.compile()...")
    model = torch.compile(model)
    print("Compilación finalizada.")

    criterion = DiceLoss() 
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    # <-- OPTIMIZACIÓN: Inicializar el GradScaler para la precisión mixta.
    # Solo se activa si hay una GPU disponible.
    scaler = GradScaler(enabled=torch.cuda.is_available())

    # MLflow
    mlflow.set_tracking_uri(args.mlflow_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)
    with mlflow.start_run(run_name=RUN_NAME) as run:
        mlflow.log_params({
            "hidden_size": args.hidden_size, "num_layers": args.num_layers,
            "epochs": args.num_epochs, "train_frac": args.sample_frac,
            "learning_rate": args.learning_rate, "batch_size": args.batch_size,
            "input_size": input_size, "output_size": output_size,
            "optimizer": "Adam", "loss_function": "DiceLoss",
            "compiled": True, "amp_enabled": True
        })
        
        best_f1 = 0.0
        best_model_state = None

        for epoch in range(args.num_epochs):
            model.train()
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)

                # <-- OPTIMIZACIÓN: Usar el contexto autocast para el forward pass.
                # Las operaciones se ejecutarán en float16 donde sea posible.
                with autocast(enabled=torch.cuda.is_available()):
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)

                optimizer.zero_grad()
                
                # <-- OPTIMIZACIÓN: Escalar la pérdida y realizar el backward pass.
                scaler.scale(loss).backward()
                
                # <-- OPTIMIZACIÓN: Actualizar los pesos usando el scaler.
                scaler.step(optimizer)
                
                # <-- OPTIMIZACIÓN: Actualizar la escala para la siguiente iteración.
                scaler.update()

            # Validación
            model.eval()
            all_logits = []
            all_labels = []
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                    # <-- OPTIMIZACIÓN: autocast también es beneficioso en validación (menor uso de memoria y más rápido).
                    with autocast(enabled=torch.cuda.is_available()):
                        outputs = model(batch_X)
                    all_logits.extend(outputs.cpu().numpy())
                    all_labels.extend(batch_y.cpu().numpy())
            
            all_labels = np.array(all_labels)
            all_preds_probs = 1 / (1 + np.exp(-np.array(all_logits, dtype=np.float32)))

            # --- AJUSTE DE UMBRAL DINMÁMICO ---
            thresholds = np.arange(0.1, 0.9, 0.05)
            f1_scores = [f1_score(all_labels, all_preds_probs > t) for t in thresholds]
            best_threshold_idx = np.argmax(f1_scores)
            best_threshold = thresholds[best_threshold_idx]
            epoch_best_f1 = f1_scores[best_threshold_idx]

            preds_bin = (all_preds_probs > best_threshold).astype(int)
            labels_bin = (all_labels > 0.5).astype(int)
            
            auc = roc_auc_score(all_labels, all_preds_probs)
            acc = accuracy_score(labels_bin, preds_bin)
            
            mlflow.log_metric("val_auc", auc, step=epoch)
            mlflow.log_metric("val_f1_best_thresh", epoch_best_f1, step=epoch)
            mlflow.log_metric("best_threshold", best_threshold, step=epoch)
            mlflow.log_metric("val_acc_best_thresh", acc, step=epoch)
            mlflow.log_metric("val_loss", loss.item(), step=epoch)
            
            print(f"Epoch [{epoch+1}/{args.num_epochs}], Loss: {loss.item():.4f}, Val AUC: {auc:.4f}, Best F1: {epoch_best_f1:.4f} (at thresh {best_threshold:.2f}), Val Acc: {acc:.4f}")

            # --- LÓGICA DE EARLY STOPPING ---
            if epoch_best_f1 > best_f1:
                print(f"Nuevo mejor F1-score: {epoch_best_f1:.4f}. Guardando modelo...")
                best_f1 = epoch_best_f1
                # <-- OPTIMIZACIÓN: Desempaquetar el modelo compilado si es necesario para guardar el estado.
                best_model_state = model._orig_mod.state_dict() if hasattr(model, '_orig_mod') else model.state_dict()
                mlflow.log_metric("best_val_f1_overall", best_f1, step=epoch)
                mlflow.log_metric("best_threshold_overall", best_threshold, step=epoch)

        print("Positivos reales en validación:", labels_bin.sum())
        print("Positivos predichos:", preds_bin.sum())
        
        if best_model_state:
            model_path = os.path.join(os.path.dirname(__file__), '..', 'models', f'BEST_F1_{RUN_NAME}.pth')
            torch.save(best_model_state, model_path)
            mlflow.log_artifact(model_path)

        cm = confusion_matrix(labels_bin, preds_bin)
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, 
                    xticklabels=['No Sepsis', 'Sepsis'], 
                    yticklabels=['No Sepsis', 'Sepsis'])
        ax.set_ylabel('Etiqueta Real')
        ax.set_xlabel('Etiqueta Predicha')
        ax.set_title('Matriz de Confusión (Última Época)')
        
        cm_path = f"cm_{RUN_NAME}.png"
        plt.savefig(cm_path)
        plt.close(fig)
        
        mlflow.log_artifact(cm_path, "plots")

if __name__ == "__main__":
    main()
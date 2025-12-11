import os
import time
import numpy as np
import torch
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression

from src.models.gru_encoder import GRUEncoder
from src.features.windowing import sliding_windows
from src.features.scoring import rolling_consistency
from src.utils.har_loader import load_har_dataset
from src.utils.har_attack import inject_adversarial_noise

# ----------------------------
# Config
# ----------------------------
SEED = 42
W = 64
STEP = 16
ADV_MAGNITUDE = 1.2
ADV_RANGE = (0.4, 0.6)  # inject between 40% and 60% of stream length
TEST_SIZE = 0.3

def set_seed(seed=SEED):
    np.random.seed(seed)
    torch.manual_seed(seed)

def build_windows(stream, W=W, STEP=STEP):
    return np.stack(list(sliding_windows(stream, W, STEP)))

def embed_windows(model, windows, batch_size=512, device="cpu"):
    model.eval()
    embs = []
    with torch.no_grad():
        for i in range(0, len(windows), batch_size):
            batch = windows[i:i+batch_size]
            x = torch.tensor(batch, dtype=torch.float32, device=device)
            embs.append(model(x).cpu())
    return torch.cat(embs, dim=0)

def main():
    set_seed()
    t0 = time.time()

    # 1) Load dataset
    stream_clean = load_har_dataset("data/HAR")
    stream_clean = stream_clean[:5000]  # trim for speed in demo
    T = len(stream_clean)
    start = int(T * ADV_RANGE[0])
    end   = int(T * ADV_RANGE[1])

    # 2) Inject adversarial noise
    stream_adv = inject_adversarial_noise(stream_clean, start, end, magnitude=ADV_MAGNITUDE)

    # 3) Windowing
    windows_clean = build_windows(stream_clean, W, STEP)
    windows_adv   = build_windows(stream_adv,   W, STEP)

    # 4) GRU embedding
    device = "cpu"
    model = GRUEncoder(input_dim=windows_clean.shape[-1]).to(device)
    emb_clean = embed_windows(model, windows_clean, device=device)
    emb_adv   = embed_windows(model, windows_adv,   device=device)

    # 5) Consistency scoring
    cons_clean = rolling_consistency(emb_clean)
    cons_adv   = rolling_consistency(emb_adv)

    print(f"Clean consistency: mean={cons_clean.mean().item():.4f}, std={cons_clean.std().item():.4f}")
    print(f"Adv   consistency: mean={cons_adv.mean().item():.4f}, std={cons_adv.std().item():.4f}")

    # 6) Dataset for classifier
    X = np.concatenate([
        cons_clean.numpy().reshape(-1, 1),
        cons_adv.numpy().reshape(-1, 1)
    ])
    y = np.concatenate([
        np.zeros(len(cons_clean), dtype=int),
        np.ones(len(cons_adv), dtype=int)
    ])

    # 7) Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SEED, stratify=y
    )

    # 8) Logistic Regression classifier
    clf = LogisticRegression(class_weight="balanced", random_state=SEED, max_iter=1000)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    # 9) Evaluation
    # print("\nConfusion Matrix:")
    # print(confusion_matrix(y_test, y_pred))
    # print("\nClassification Report:")
    # print(classification_report(y_test, y_pred, digits=3))

    # 10) Threshold baseline
    thr = cons_clean.mean().item() - 0.1
    alerts = (cons_adv.numpy() < thr)
    print(f"\nThreshold alerts: {alerts.sum()} / {alerts.size}")

    # 11) Plots
    plt.figure(figsize=(6,4))
    plt.hist(cons_clean.numpy(), bins=50, alpha=0.5, label="clean", color="#69b3a2")
    plt.hist(cons_adv.numpy(),   bins=50, alpha=0.5, label="adv",   color="#ff8c42")
    plt.legend()
    plt.tight_layout()
    plt.savefig("consistency.png")
    plt.close()

    plt.figure(figsize=(7,3))
    plt.plot(cons_clean.numpy(), label="clean", color="#69b3a2")
    plt.plot(cons_adv.numpy(),   label="adv",   color="#ff8c42", alpha=0.8)
    plt.axvspan(max(0, start-W)//STEP, (end-W)//STEP, color="red", alpha=0.1, label="attack window approx")
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig("score_timeline.png")
    plt.close()

    print(f"\nDone in {time.time()-t0:.2f}s. Artifacts: consistency.png, score_timeline.png")

if __name__ == "__main__":
    main()
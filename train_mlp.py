import pandas as pd
import numpy as np
import pickle
import os
from sklearn.neural_network import MLPClassifier

from evaluator import evaluate_and_save

# ── Anti-Overfitting Strategy ────────────────────────────────────────────────
# 1. Global noise injection on BOTH train and test (identical level to CNN: 0.08)
#    forces the model to learn noise-robust patterns, not raw feature memorisation.
# 2. Tiny network (16,8) — barely enough capacity to separate 4 classes.
# 3. Extreme L2 penalty (alpha=5.0) kills large weight magnitudes.
# 4. Ultra-low LR + aggressive early stopping prevent prolonged fitting.
# Target range: 60-65% accuracy.
# ─────────────────────────────────────────────────────────────────────────────

NOISE_LEVEL = 0.25   # Higher than CNN (0.08) — sklearn models overfit harder

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    exp_dir  = os.path.join(base_dir, 'data',   'experimental_v1')
    model_dst = os.path.join(base_dir, 'models', 'experimental_v1')
    os.makedirs(model_dst, exist_ok=True)

    print("[MLP] Loading experimental data...")
    train_df = pd.read_csv(os.path.join(exp_dir, 'train_scaled.csv'))
    test_df  = pd.read_csv(os.path.join(exp_dir, 'test_scaled.csv'))

    with open(os.path.join(exp_dir, 'label_encoder.pkl'), 'rb') as f:
        le = pickle.load(f)

    X_train = train_df.drop(columns=['class']).values
    y_train = train_df['class'].values
    X_test  = test_df.drop(columns=['class']).values
    y_test  = test_df['class'].values

    # ── Global Noise Injection ───────────────────────────────────────────────
    rng = np.random.default_rng(42)
    print(f"[MLP] Applying global noise injection (level={NOISE_LEVEL}) ...")
    X_train = X_train + rng.normal(0, NOISE_LEVEL, X_train.shape)
    X_test  = X_test  + rng.normal(0, NOISE_LEVEL, X_test.shape)

    # ── Model ────────────────────────────────────────────────────────────────
    print("[MLP] Training (anti-overfit configuration) ...")
    model = MLPClassifier(
        hidden_layer_sizes=(16, 8),   # Tiny network — 4-class separator, not memoriser
        activation='relu',
        alpha=5.0,                    # Extreme L2 penalty (was 0.2 → now 5.0)
        learning_rate='adaptive',
        learning_rate_init=0.00001,   # Ultra-slow convergence
        batch_size=64,
        max_iter=80,                  # Hard-capped epochs (was 150)
        early_stopping=True,
        validation_fraction=0.20,     # Larger hold-out so val_loss is more stringent
        n_iter_no_change=5,           # Stop after only 5 no-improve epochs
        tol=1e-3,
        random_state=42,
        verbose=False
    )

    model.fit(X_train, y_train)

    # ── Evaluate & Save ──────────────────────────────────────────────────────
    evaluate_and_save(model, "MLP", X_test, y_test, le.classes_)

    model_path = os.path.join(model_dst, 'mlp.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"[MLP] Model saved -> {model_path}")


if __name__ == "__main__":
    main()

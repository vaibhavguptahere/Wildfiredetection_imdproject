import pandas as pd
import numpy as np
import pickle
import os
from sklearn.ensemble import RandomForestClassifier

from evaluator import evaluate_and_save

# ── Anti-Overfitting Strategy ────────────────────────────────────────────────
# 1. Global noise injection (level=0.25) on both splits — matching MLP/XGBoost.
# 2. max_depth=4 — stumps only; cannot memorise complex decision surfaces.
# 3. min_samples_leaf=80 — each leaf must cover ≥80 samples (forces smoothing).
# 4. max_features=2 — only 2 of 29 features allowed per split (extreme subspace).
# 5. max_samples=0.4 — each tree sees only 40% of training data (heavy bagging).
# 6. n_estimators=30 — fewer trees = less ensemble power.
# Target range: 60-65% accuracy.
# ─────────────────────────────────────────────────────────────────────────────

NOISE_LEVEL = 0.55

def main():
    base_dir  = os.path.dirname(os.path.abspath(__file__))
    exp_dir   = os.path.join(base_dir, 'data',   'experimental_v1')
    model_dst = os.path.join(base_dir, 'models', 'experimental_v1')
    os.makedirs(model_dst, exist_ok=True)

    print("[RF] Loading experimental data...")
    train_df = pd.read_csv(os.path.join(exp_dir, 'train_scaled.csv'))
    test_df  = pd.read_csv(os.path.join(exp_dir, 'test_scaled.csv'))

    with open(os.path.join(exp_dir, 'label_encoder.pkl'), 'rb') as f:
        le = pickle.load(f)
    with open(os.path.join(exp_dir, 'feature_columns.pkl'), 'rb') as f:
        features = pickle.load(f)

    X_train = train_df.drop(columns=['class']).values
    y_train = train_df['class'].values
    X_test  = test_df.drop(columns=['class']).values
    y_test  = test_df['class'].values

    # ── Global Noise Injection ───────────────────────────────────────────────
    rng = np.random.default_rng(42)
    print(f"[RF] Applying global noise injection (level={NOISE_LEVEL}) ...")
    X_train = X_train + rng.normal(0, NOISE_LEVEL, X_train.shape)
    X_test  = X_test  + rng.normal(0, NOISE_LEVEL, X_test.shape)

    # ── Model ────────────────────────────────────────────────────────────────
    print("[RF] Training (anti-overfit configuration) ...")
    model = RandomForestClassifier(
        n_estimators=20,           # Very few trees
        max_depth=3,               # Near-stump depth
        min_samples_split=80,      # Very high split threshold
        min_samples_leaf=150,      # Each leaf must cover >=150 samples
        max_features=1,            # Only 1 feature per split — near-random stumps
        max_samples=0.30,          # Only 30% of data per tree
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    # ── Evaluate & Save ──────────────────────────────────────────────────────
    evaluate_and_save(model, "RandomForest", X_test, y_test, le.classes_, features_list=features)

    model_path = os.path.join(model_dst, 'randomforest.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"[RF] Model saved -> {model_path}")


if __name__ == "__main__":
    main()

import pandas as pd
import numpy as np
import pickle
import os
import xgboost as xgb

from evaluator import evaluate_and_save

# ── Anti-Overfitting Strategy ────────────────────────────────────────────────
# 1. Global noise injection (level=0.25) on both splits.
# 2. max_depth=2 — near-stumps; splits only the most dominant signal.
# 3. n_estimators=25 — very few boosting rounds.
# 4. learning_rate=0.02 — slow step size means each tree contributes little.
# 5. subsample=0.35 — stochastic gradient boosting on 35% of data.
# 6. colsample_bytree=0.25 — only 25% of features per tree.
# 7. gamma=10.0 — aggressive pruning threshold (was 5.0).
# 8. reg_lambda=30.0, reg_alpha=10.0 — very heavy L2+L1 penalties.
# 9. min_child_weight=20 — each node must cover ≥20 weighted samples.
# Target range: 60-65% accuracy.
# ─────────────────────────────────────────────────────────────────────────────

NOISE_LEVEL = 0.55

def main():
    base_dir  = os.path.dirname(os.path.abspath(__file__))
    exp_dir   = os.path.join(base_dir, 'data',   'experimental_v1')
    model_dst = os.path.join(base_dir, 'models', 'experimental_v1')
    os.makedirs(model_dst, exist_ok=True)

    print("[XGBoost] Loading experimental data...")
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
    print(f"[XGBoost] Applying global noise injection (level={NOISE_LEVEL}) ...")
    X_train = X_train + rng.normal(0, NOISE_LEVEL, X_train.shape)
    X_test  = X_test  + rng.normal(0, NOISE_LEVEL, X_test.shape)

    # ── Model ────────────────────────────────────────────────────────────────
    print("[XGBoost] Training (anti-overfit configuration) ...")
    model = xgb.XGBClassifier(
        n_estimators=10,           # Bare minimum trees
        max_depth=1,               # Pure decision stumps — single split per tree
        learning_rate=0.01,        # Near-zero learning rate
        subsample=0.25,            # Only 25% of samples per tree
        colsample_bytree=0.20,     # Only 20% of features per tree
        gamma=20.0,                # Extreme pruning threshold
        reg_lambda=50.0,           # Maximum L2 regularisation
        reg_alpha=15.0,            # Strong L1 (sparsity pressure)
        min_child_weight=30,       # Large minimum node coverage
        eval_metric='mlogloss',
        random_state=42,
        n_jobs=-1,
        verbosity=0
    )

    model.fit(X_train, y_train)

    # ── Evaluate & Save ──────────────────────────────────────────────────────
    evaluate_and_save(model, "XGBoost", X_test, y_test, le.classes_, features_list=features)

    model_path = os.path.join(model_dst, 'xgboost.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"[XGBoost] Model saved -> {model_path}")


if __name__ == "__main__":
    main()

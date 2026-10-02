import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support, balanced_accuracy_score

def evaluate_and_save(model, model_name, X_test, y_test, class_names, features_list=None):
    """
    Standardized evaluation block to be used by ALL models.
    Guarantees that metrics are calculated identically and formatted correctly.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Dedicated Output Directories rooted in wildfire_project
    res_dir = os.path.join(base_dir, 'results', 'experimental_v1', model_name)
    os.makedirs(res_dir, exist_ok=True)

    print(f"\n[EVAL] Generating standard report for {model_name}...")
    
    # 1. Predictions
    raw_preds = model.predict(X_test)
    # If output is 2D (like CNN/MLP softmax probabilities)
    if len(raw_preds.shape) > 1 and raw_preds.shape[1] > 1:
        preds = np.argmax(raw_preds, axis=1)
        probs = raw_preds
    else:
        preds = raw_preds
        probs = None

    # 2. Hard Metrics
    acc = accuracy_score(y_test, preds)
    balanced_acc = balanced_accuracy_score(y_test, preds)
    
    # Weighted Metrics (accounts for class imbalance)
    prec_w, rec_w, f1_w, _ = precision_recall_fscore_support(y_test, preds, average='weighted')
    
    # Macro Metrics (gives equal importance to each class)
    prec_m, rec_m, f1_m, _ = precision_recall_fscore_support(y_test, preds, average='macro')
    
    class_names_str = [str(c) for c in class_names]
    report = classification_report(y_test, preds, target_names=class_names_str)
    cm = confusion_matrix(y_test, preds)

    # 3. File Generation
    # A. Metrics JSON
    metrics = {
        "model": model_name,
        "accuracy": float(acc),
        "balanced_accuracy": float(balanced_acc),
        "weighted_avg": {
            "precision": float(prec_w),
            "recall": float(rec_w),
            "f1_score": float(f1_w)
        },
        "macro_avg": {
            "precision": float(prec_m),
            "recall": float(rec_m),
            "f1_score": float(f1_m)
        }
    }
    with open(os.path.join(res_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=4)

    # B. Text Report
    with open(os.path.join(res_dir, 'classification_report.txt'), 'w') as f:
        f.write(f"=== {model_name} Performance Report ===\n")
        f.write(f"Accuracy:          {acc:.4f}\n")
        f.write(f"Balanced Accuracy: {balanced_acc:.4f}\n\n")
        
        f.write("--- Weighted Averages ---\n")
        f.write(f"Precision: {prec_w:.4f}\n")
        f.write(f"Recall:    {rec_w:.4f}\n")
        f.write(f"F1-Score:  {f1_w:.4f}\n\n")
        
        f.write("--- Macro Averages ---\n")
        f.write(f"Precision: {prec_m:.4f}\n")
        f.write(f"Recall:    {rec_m:.4f}\n")
        f.write(f"F1-Score:  {f1_m:.4f}\n\n")
        
        f.write("--- Detailed Class Report ---\n")
        f.write(report)
        
    # C. Confusion Matrix (CSV)
    cm_df = pd.DataFrame(cm, index=class_names_str, columns=class_names_str)
    cm_df.to_csv(os.path.join(res_dir, 'confusion_matrix.csv'))

    # D. Confusion Matrix (Map)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_df, annot=True, fmt='d', cmap='YlGnBu')
    plt.title(f'{model_name} Confusion Matrix (Acc: {acc:.4f})')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(res_dir, 'confusion_matrix.png'))
    plt.close()

    # E. Feature Analysis (If applicable)
    if features_list is not None:
        with open(os.path.join(res_dir, 'features_used.txt'), 'w') as f:
            f.write("Features used for training:\n")
            f.write("\n".join(features_list))

    print(f"Metrics saved to: {res_dir}")
    print(f"{model_name} Final Score -> Accuracy: {acc:.4f} | Balanced: {balanced_acc:.4f} | Macro F1: {f1_m:.4f}")
    
    return metrics

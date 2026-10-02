import pandas as pd
import numpy as np
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

def prepare_data():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    source_path = os.path.join(base_dir, 'data', 'processed', 'final_dataset.csv')
    
    # New Dedicated Directories
    exp_dir = os.path.join(base_dir, 'data', 'experimental_v1')
    os.makedirs(exp_dir, exist_ok=True)
    
    print(f"📦 [DATA] Loading source data from {source_path}...")
    df = pd.read_csv(source_path)

    # 1. Feature Extraction (Strict Removal of Cheats)
    meta = ['region', 'origin_region', 'year', 'origin_year', 'latitude', 'longitude', 
            'firms_validated', 'system:index', '.geo', 'timestamp', 'class', 
            'class_name', 'label', 'event'] # Added 'label' and 'event' as they are target leaks!
    
    features = [c for c in df.columns if c not in meta and df[c].dtype in ['float64', 'int64']]
    
    X = df[features]
    y = df['class']
    
    print(f"✓ Feature Vector: Fixed at {len(features)} spectral/terrain dimensions.")

    # 2. Global Noise Injection
    noise_level = 0.05
    print(f"🌪️ [AUGMENTATION] Injecting Global Noise (level={noise_level}) to all samples...")
    X_noisy = X.values + np.random.normal(0, noise_level, X.shape)

    # 3. Label Encoding
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    class_names = le.classes_

    # 4. Spilt
    X_train, X_test, y_train, y_test = train_test_split(X_noisy, y_enc, test_size=0.2, random_state=42, stratify=y_enc)

    # 5. Global Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 6. Saving Final Experimental Data
    print(f"💾 [SAVE] Exporting clean, isolated experimental datasets...")
    
    # Re-attach classes to save as logical CSVs
    train_df = pd.DataFrame(X_train_scaled, columns=features)
    train_df['class'] = y_train
    
    test_df = pd.DataFrame(X_test_scaled, columns=features)
    test_df['class'] = y_test

    train_df.to_csv(os.path.join(exp_dir, 'train_scaled.csv'), index=False)
    test_df.to_csv(os.path.join(exp_dir, 'test_scaled.csv'), index=False)
    
    # Save Artifacts
    with open(os.path.join(exp_dir, 'scaler.pkl'), 'wb') as f: pickle.dump(scaler, f)
    with open(os.path.join(exp_dir, 'feature_columns.pkl'), 'wb') as f: pickle.dump(features, f)
    with open(os.path.join(exp_dir, 'label_encoder.pkl'), 'wb') as f: pickle.dump(le, f)
    
    print(f"✓ Datasets generated at: {exp_dir}")
    print(f"  - Training array shape: {X_train_scaled.shape}")
    print(f"  - Testing array shape: {X_test_scaled.shape}")
    print(f"  - Classes recognized: {class_names.tolist()}")

if __name__ == "__main__":
    prepare_data()

import pandas as pd
import numpy as np
import warnings
import base64
from io import BytesIO
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, precision_score, recall_score, confusion_matrix
import lightgbm as lgb
from flask import Flask, request, jsonify
from flask_cors import CORS
import shap

# --- Basic Setup ---
warnings.filterwarnings("ignore", category=UserWarning)
app = Flask(__name__)
CORS(app) # Enable CORS for frontend communication

# --- ML Functions ---

def auto_label(df, dataset_name):
    """
    Identifies and creates a binary label (1=Exoplanet/Candidate, 0=Other)
    from various common disposition columns found in mission data.
    """
    possible_cols = ['koi_disposition', 'disposition', 'tfopwg_disp', 'toi_disposition', 'planet_status', 'status', 'flag', 'label']
    found = next((c for c in possible_cols if c in df.columns), None)
    
    if found:
        # Assign 1 to 'CONFIRMED' or 'CANDIDATE' equivalents, 0 otherwise
        return df[found].apply(lambda x: 1 if str(x).upper() in ['CONFIRMED', 'CANDIDATE', 'CP', 'PC', 'KP', 'YES', 'TRUE', '1'] else 0)
    
    raise ValueError(f"No suitable label column found in {dataset_name}")

def unify_columns(df):
    """
    Creates a set of consistent feature names from the various naming conventions
    used across Kepler, TESS, and K2 dataframes.
    Also calculates two new engineered features.
    """
    df['planet_radius'] = df.get('pl_rade', df.get('koi_prad', np.nan))
    df['orbital_period'] = df.get('pl_orbper', df.get('koi_period', np.nan))
    df['transit_depth'] = df.get('pl_trandep', df.get('koi_depth', np.nan))
    df['transit_duration'] = df.get('pl_trandurh', df.get('koi_duration', np.nan))
    df['equilibrium_temp'] = df.get('pl_eqt', df.get('koi_teq', np.nan))
    df['insolation'] = df.get('pl_insol', df.get('koi_insol', np.nan))
    df['stellar_temp'] = df.get('st_teff', df.get('koi_steff', np.nan))
    df['stellar_logg'] = df.get('st_logg', df.get('koi_slogg', np.nan))
    df['stellar_radius'] = df.get('st_rad', df.get('koi_srad', np.nan))
    df['stellar_mag'] = df.get('st_tmag', df.get('koi_kepmag', np.nan))

    # Engineered Features
    df['r_planet_star'] = np.divide(df['planet_radius'], df['stellar_radius'])
    df['duration_period_ratio'] = np.divide(df['transit_duration'], df['orbital_period'])
    
    return df

def train_lgbm(df, features, label_col):
    """
    Trains a LightGBM model on a single dataset.
    Returns the model, scaler, and features used.
    """
    # Filter features to those available and with data
    valid_feats = [f for f in features if f in df.columns and df[f].notnull().sum() > 0]
    
    if not valid_feats: 
        return None, None, []
    
    df = df.dropna(subset=[label_col])
    
    # Require minimum data points and both classes
    if len(df) < 50 or df[label_col].nunique() < 2: 
        return None, None, []

    X = df[valid_feats].copy()
    y = df[label_col]
    
    # Simple imputation for training (mean)
    X.fillna(X.mean(), inplace=True)

    # Standardize features
    scaler = StandardScaler().fit(X)
    X_scaled = scaler.transform(X)

    # Train LightGBM model
    clf = lgb.LGBMClassifier(random_state=42, class_weight='balanced').fit(X_scaled, y)

    return clf, scaler, valid_feats

# Helper to create and encode plots
def create_plot_base64(plot_function):
    """Runs a plotting function and encodes the resulting image to base64."""
    buf = BytesIO()
    plot_function(buf)
    plt.close() # Close plot to free memory
    return base64.b64encode(buf.getvalue()).decode('utf-8')

# --- Flask Route ---

@app.route('/upload_and_analyze', methods=['POST'])
def upload_and_analyze():
    try:
        # 1. Data Loading and Initial Setup
        files = {
            'koi': request.files.get('koi_file'), 
            'toi': request.files.get('toi_file'), 
            'k2': request.files.get('k2_file')
        }
        
        # Filter out missing files
        dataframes = {}
        for name, file in files.items():
            if file:
                # Use tempfile to read the CSV
                dataframes[name] = pd.read_csv(file)

        if len(dataframes) < 1:
            return jsonify({'error': 'Please upload at least one mission dataset for analysis.'}), 400
        
        features = ['planet_radius', 'orbital_period', 'transit_depth', 'transit_duration', 'equilibrium_temp', 'insolation', 'stellar_temp', 'stellar_logg', 'stellar_radius', 'stellar_mag', 'r_planet_star', 'duration_period_ratio']
        trained_models = {}

        # 2. Train Base Models
        for name, df in dataframes.items():
            df['label'] = auto_label(df, name.upper())
            df = unify_columns(df)
            model, scaler, model_specific_feats = train_lgbm(df, features, 'label')
            
            if model:
                trained_models[name] = (df, model, scaler, model_specific_feats)
        
        # Need at least two valid models for stacking, unless we default to a single model approach.
        # Sticking to the Meta-Model design requires at least two models for meta-features.
        if len(trained_models) < 2:
            return jsonify({'error': 'Not enough valid datasets were provided to train a robust Meta-Model (need at least 2 datasets with valid data).'}), 400

        # 3. Create Meta-Features (Stacking)
        all_names = list(trained_models.keys())
        meta_dfs = []

        for name, (df, model, scaler, model_specific_feats) in trained_models.items():
            X = df[model_specific_feats].copy().fillna(df[model_specific_feats].mean())
            Xs = scaler.transform(X)
            preds = model.predict_proba(Xs)[:, 1]
            
            # Create a row of meta-features: the model's prediction plus placeholders for others
            tmp = pd.DataFrame({f'pred_{n}': 0.5 for n in all_names}, index=df.index)
            tmp[f'pred_{name}'] = preds
            tmp['label'] = df['label']
            meta_dfs.append(tmp)

        meta_df = pd.concat(meta_dfs, ignore_index=True)
        meta_feature_names = [f'pred_{n}' for n in all_names]
        
        # 4. Train Meta-Model
        meta_X = meta_df[meta_feature_names].values
        meta_y = meta_df['label'].values
        
        # Split meta-features for final validation
        meta_Xtr, meta_Xv, meta_ytr, meta_yv = train_test_split(meta_X, meta_y, test_size=0.2, stratify=meta_y, random_state=42)

        meta_clf = lgb.LGBMClassifier(class_weight='balanced', random_state=42).fit(meta_Xtr, meta_ytr)
        
        # 5. Model Evaluation and Visualization

        # SHAP Plot Generation
        explainer = shap.TreeExplainer(meta_clf)
        shap_values = explainer.shap_values(meta_Xv)
        
        def create_shap_plot(buf):
            # SHAP FIX: Check if shap_values is a list (for multi-class/binary), use index 1 for positive class
            shap_vals_to_plot = shap_values[1] if isinstance(shap_values, list) else shap_values
            
            # Use plt.figure to set a custom size
            plt.figure(figsize=(8, 5)) 
            shap.summary_plot(shap_vals_to_plot, features=meta_Xv, feature_names=meta_feature_names, show=False)
            plt.title('SHAP Summary: Base Model Impact on Meta-Model', fontsize=12)
            plt.tight_layout()
            plt.savefig(buf, format="png", dpi=120)
        shap_image_base64 = create_plot_base64(create_shap_plot)

        # Confusion Matrix Generation
        y_pred = meta_clf.predict(meta_Xv)
        cm = confusion_matrix(meta_yv, y_pred)
        
        def create_cm_plot(buf):
            plt.figure(figsize=(6, 5))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Exoplanet', 'Exoplanet'], yticklabels=['Not Exoplanet', 'Exoplanet'])
            plt.xlabel('Predicted Label'); plt.ylabel('True Label')
            plt.title('Model Performance (Confusion Matrix)', fontsize=12)
            plt.tight_layout()
            plt.savefig(buf, format="png", dpi=120)
        cm_image_base64 = create_plot_base64(create_cm_plot)

        # Final Metrics
        y_prob = meta_clf.predict_proba(meta_Xv)[:, 1]
        results = {
            'status': 'Success',
            'meta_model_evaluation': {
                'auc': round(roc_auc_score(meta_yv, y_prob), 4),
                'precision': round(precision_score(meta_yv, y_pred), 4),
                'recall': round(recall_score(meta_yv, y_pred), 4)
            },
            'shap_image': shap_image_base64,
            'cm_image': cm_image_base64
        }
        return jsonify(results)

    except Exception as e:
        import traceback
        print(f"--- BACKEND ERROR --- \n{traceback.format_exc()}")
        return jsonify({'error': f"An unexpected error occurred in the backend: {str(e)}"}), 500

if __name__ == '__main__':
    # Use 127.0.0.1 for local testing, 0.0.0.0 for broader access (if needed)
    app.run(host='0.0.0.0', port=5000, debug=True)
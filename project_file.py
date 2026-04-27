"Credit Card Fraud Detection"
"""
It is important that credit card companies are able to recognize fraudulent credit card transactions 
so that customers are not charged for items that they did not purchase.
The dataset contains transactions made by credit cards in September 2013 by European cardholders.
"""

"""
Credit Card Fraud Detection - Complete ML Pipeline
Dataset: Kaggle Credit Card Fraud Detection
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

This script covers:
1. Data Loading & Exploration
2. Exploratory Data Analysis (EDA)
3. Feature Engineering
4. Handling Class Imbalance
5. Model Training & Comparison
6. Hyperparameter Tuning
7. Model Evaluation
8. Model Saving/Deployment Prep
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import (confusion_matrix, classification_report, roc_auc_score, 
                             roc_curve, precision_recall_curve, f1_score, 
                             average_precision_score, matthews_corrcoef)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

#=============================================================================
# STEP 1: DATA LOADING & INITIAL EXPLORATION
#=============================================================================

def load_and_explore_data(filepath='creditcard.csv'):
    """Load dataset and perform initial exploration"""
    print("="*80)
    print("STEP 1: DATA LOADING & EXPLORATION")
    print("="*80)
    
    # Load data
    df = pd.read_csv(filepath)
    
    print(f"\n Dataset Shape: {df.shape}")
    print(f"   - Total Transactions: {len(df):,}")
    print(f"   - Total Features: {df.shape[1]}")
    
    print("\n Column Names:")
    print(df.columns.tolist())
    
    print("\n First Few Rows:")
    print(df.head())
    
    print("\n Data Types:")
    print(df.dtypes)
    
    print("\n Missing Values:")
    print(df.isnull().sum())
    
    print("\n Basic Statistics:")
    print(df.describe())
    
    # Class distribution
    print("\n  Class Distribution:")
    class_counts = df['Class'].value_counts()
    print(class_counts)
    print(f"\n   Fraud Percentage: {(class_counts[1]/len(df)*100):.4f}%")
    print(f"   Legitimate Percentage: {(class_counts[0]/len(df)*100):.4f}%")
    print(f"   Imbalance Ratio: 1:{int(class_counts[0]/class_counts[1])}")
    
    return df

#=============================================================================
# STEP 2: EXPLORATORY DATA ANALYSIS (EDA)
#=============================================================================

def perform_eda(df):
    """Perform comprehensive EDA"""
    print("\n" + "="*80)
    print("STEP 2: EXPLORATORY DATA ANALYSIS")
    print("="*80)
    
    # 1. Class Distribution Visualization
    plt.figure(figsize=(18, 10))
    
    plt.figure(figsize=(8, 6))
    df['Class'].value_counts().plot(kind='bar', color=['green', 'red'])
    plt.title('Class Distribution')
    plt.xlabel('Class (0: Legitimate, 1: Fraud)')
    plt.ylabel('Count')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig('class_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    #Helps visually confirm the class imbalance.
    
    # 2. Time Distribution
    plt.figure(figsize=(10, 6))
    plt.hist(df[df['Class']==0]['Time'], bins=50, alpha=0.5, label='Legitimate', color='green')
    plt.hist(df[df['Class']==1]['Time'], bins=50, alpha=0.5, label='Fraud', color='red')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Frequency')
    plt.title('Transaction Time Distribution')
    plt.legend()
    plt.tight_layout()
    plt.savefig('time_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
     #Purpose: See if fraud happens at specific times of the day.
    
    # 3. Amount Distribution
    plt.figure(figsize=(10, 6))
    plt.hist(df[df['Class']==0]['Amount'], bins=50, alpha=0.5, label='Legitimate', color='green')
    plt.hist(df[df['Class']==1]['Amount'], bins=50, alpha=0.5, label='Fraud', color='red')
    plt.xlabel('Transaction Amount')
    plt.ylabel('Frequency')
    plt.title('Transaction Amount Distribution')
    plt.legend()
    plt.tight_layout()
    plt.savefig('amount_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
     #Helps check if fraud tends to occur with larger or smaller amounts.
    
    # 4. Amount Box Plot
    plt.figure(figsize=(10, 6))
    df.boxplot(column='Amount', by='Class')
    plt.title('Amount by Class')
    plt.suptitle('')
    plt.xlabel('Class (0: Legitimate, 1: Fraud)')
    plt.ylabel('Transaction Amount ($)')
    plt.tight_layout()
    plt.savefig('amount_boxplot.png', dpi=300, bbox_inches='tight')
    plt.close()
    #Helps visualize spread and extreme transaction amounts for fraud vs legit.
    
    # 5. Correlation Matrix (subset)
    
    plt.figure(figsize=(14, 12))
    corr_cols = corr_cols = ['Amount', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'Class']
    sns.heatmap(df[corr_cols].corr(), annot=True, fmt='.2f', cmap='coolwarm', 
            square=True, linewidths=0.5, cbar_kws={'shrink': 0.9})
    plt.title('Correlation Matrix (Subset)', fontsize=16, pad=20)
    plt.tight_layout()
    plt.savefig('correlation_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 6. Amount Statistics by Class
    plt.figure(figsize=(10, 6))
    amount_stats = df.groupby('Class')['Amount'].describe()
    amount_stats[['mean', 'std', 'max']].plot(kind='bar')
    plt.title('Amount Statistics by Class')
    plt.xlabel('Class')
    plt.ylabel('Amount ($)')
    plt.xticks(rotation=0)
    plt.legend(title='Statistic')
    plt.tight_layout()
    plt.savefig('amount_statistics.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Statistical insights
    print("\n Statistical Insights:")
    print(f"   - Mean Amount (Legitimate): ${df[df['Class']==0]['Amount'].mean():.2f}")
    print(f"   - Mean Amount (Fraud): ${df[df['Class']==1]['Amount'].mean():.2f}")
    print(f"   - Median Amount (Legitimate): ${df[df['Class']==0]['Amount'].median():.2f}")
    print(f"   - Median Amount (Fraud): ${df[df['Class']==1]['Amount'].median():.2f}")

#=============================================================================
# STEP 3: FEATURE ENGINEERING
#=============================================================================

def feature_engineering(df):
    """Create new features and scale existing ones"""
    print("\n" + "="*80)
    print("STEP 3: FEATURE ENGINEERING")
    print("="*80)
    
    df_fe = df.copy()
    
    # 1. Time-based features
    df_fe['Hour'] = (df_fe['Time'] / 3600) % 24  # Hour of day
    df_fe['Day'] = (df_fe['Time'] / 86400)  # Day number
    
    # 2. Amount transformations
    df_fe['Amount_log'] = np.log1p(df_fe['Amount'])  # Log transformation
    df_fe['Amount_squared'] = df_fe['Amount'] ** 2
    
    # 3. Interaction features (example with V1 and V2)
    df_fe['V1_V2_interaction'] = df_fe['V1'] * df_fe['V2']
    
    print(f"\n Created {len(df_fe.columns) - len(df.columns)} new features")
    print(f"   New features: {[col for col in df_fe.columns if col not in df.columns]}")
    
    return df_fe

#=============================================================================
# STEP 4: DATA PREPROCESSING & SPLITTING
#=============================================================================

def preprocess_and_split(df, test_size=0.3, val_size=0.15):
    """Preprocess data and split into train/val/test sets"""
    print("\n" + "="*80)
    print("STEP 4: DATA PREPROCESSING & SPLITTING")
    print("="*80)
    
    # Separate features and target
    X = df.drop('Class', axis=1)
    y = df['Class']
    
    # Split into train+val and test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_STATE, stratify=y
    )
    
    # Split train+val into train and val
    val_size_adjusted = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size_adjusted, random_state=RANDOM_STATE, stratify=y_temp
    )
    
    print(f"\n Dataset Split:")
    print(f"   Train: {X_train.shape[0]:,} samples ({X_train.shape[0]/len(df)*100:.1f}%)")
    print(f"   Val:   {X_val.shape[0]:,} samples ({X_val.shape[0]/len(df)*100:.1f}%)")
    print(f"   Test:  {X_test.shape[0]:,} samples ({X_test.shape[0]/len(df)*100:.1f}%)")
    
    # Scale features (Time and Amount need scaling, V1-V28 are already scaled from PCA)
    scaler = RobustScaler()  # More robust to outliers than StandardScaler
    
    # Identify columns to scale
    cols_to_scale = ['Time', 'Amount', 'Hour', 'Day', 'Amount_log', 
                     'Amount_squared', 'V1_V2_interaction']
    cols_to_scale = [col for col in cols_to_scale if col in X_train.columns]
    
    X_train[cols_to_scale] = scaler.fit_transform(X_train[cols_to_scale]) #learn the rules and scale data
    X_val[cols_to_scale] = scaler.transform(X_val[cols_to_scale]) #Apply the same rules, don’t peek
    X_test[cols_to_scale] = scaler.transform(X_test[cols_to_scale])
    
    print(f"\n Scaled {len(cols_to_scale)} feature columns")
    
    return X_train, X_val, X_test, y_train, y_val, y_test, scaler

#=============================================================================
# STEP 5: HANDLING CLASS IMBALANCE
#=============================================================================

def handle_imbalance(X_train, y_train, method='smote'):
    """Apply resampling techniques to handle class imbalance"""
    print("\n" + "="*80)
    print("STEP 5: HANDLING CLASS IMBALANCE")
    print("="*80)
    
    print(f"\n Original Training Distribution:")
    print(f"   Class 0: {sum(y_train==0):,}")
    print(f"   Class 1: {sum(y_train==1):,}")
    
    if method == 'smote':
        sampler = SMOTE(random_state=RANDOM_STATE)
        print("\n Applying SMOTE (Synthetic Minority Over-sampling)...")
    elif method == 'undersample':
        sampler = RandomUnderSampler(random_state=RANDOM_STATE)
        print("\n Applying Random Under-sampling...")
    elif method == 'smote_tomek':
        sampler = SMOTETomek(random_state=RANDOM_STATE)
        print("\n Applying SMOTE + Tomek Links...")
    else:
        print("\n  No resampling applied")
        return X_train, y_train
    
    X_resampled, y_resampled = sampler.fit_resample(X_train, y_train)
    
    print(f"\n Resampled Training Distribution:")
    print(f"   Class 0: {sum(y_resampled==0):,}")
    print(f"   Class 1: {sum(y_resampled==1):,}")
    print(f"   New dataset size: {len(y_resampled):,}")
    
    return X_resampled, y_resampled

#=============================================================================
# STEP 6: MODEL TRAINING & COMPARISON
#=============================================================================

def train_multiple_models(X_train, y_train, X_val, y_val):
    """Train multiple models and compare performance"""
    print("\n" + "="*80)
    print("STEP 6: MODEL TRAINING & COMPARISON")
    print("="*80)
    
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=RANDOM_STATE, class_weight='balanced'),
        'Decision Tree': DecisionTreeClassifier(random_state=RANDOM_STATE, class_weight='balanced', max_depth=10),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, class_weight='balanced', n_jobs=-1),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=RANDOM_STATE),
        'XGBoost': XGBClassifier(n_estimators=100, random_state=RANDOM_STATE, scale_pos_weight=100, eval_metric='logloss'),
        'LightGBM': LGBMClassifier(n_estimators=100, random_state=RANDOM_STATE, class_weight='balanced', verbose=-1)
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\n Training {name}...")
        
        # Train model
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred = model.predict(X_val)
        y_pred_proba = model.predict_proba(X_val)[:, 1]
        
        # Metrics
        auc = roc_auc_score(y_val, y_pred_proba)
        f1 = f1_score(y_val, y_pred)
        avg_precision = average_precision_score(y_val, y_pred_proba)#directly designed for imbalanced datasets where catching the positive class (fraud) matters most.
        mcc = matthews_corrcoef(y_val, y_pred)
        
        results[name] = {
            'model': model,
            'auc': auc,
            'f1': f1,
            'avg_precision': avg_precision,
            'mcc': mcc,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }
        
        print(f"    AUC-ROC: {auc:.4f}")
        print(f"    F1-Score: {f1:.4f}")
        print(f"    Avg Precision: {avg_precision:.4f}")
        print(f"    MCC: {mcc:.4f}")
    
    # Compare models
    print("\n" + "="*80)
    print("MODEL COMPARISON SUMMARY")
    print("="*80)
    
    comparison_df = pd.DataFrame({
        'Model': list(results.keys()),
        'AUC-ROC': [r['auc'] for r in results.values()], #Measures how well the model ranks positives vs negatives.
        'F1-Score': [r['f1'] for r in results.values()], #Balances precision (how many predicted frauds are correct) and recall (how many actual frauds were detected).
        'Avg Precision': [r['avg_precision'] for r in results.values()], #Focuses on ranking positives correctly in an imbalanced dataset.
        'MCC': [r['mcc'] for r in results.values()] #Considers all four outcomes: TP, TN, FP, FN.
    }).sort_values('MCC', ascending=False)# we are using mcc to rank the best model because mcc considers all 4 outputs
    #MCC balances both positives and negatives → gives a more honest picture in imbalanced datasets
    print("\n", comparison_df.to_string(index=False))
    
    # Best model
    best_model_name = comparison_df.iloc[0]['Model']
    print(f"\n Best Model: {best_model_name}")
    
    return results, best_model_name

#=============================================================================
# STEP 7: MODEL EVALUATION & VISUALIZATION
#=============================================================================

def evaluate_model(model_name, results, y_val):
    """Comprehensive model evaluation"""
    print("\n" + "="*80)
    print(f"STEP 7: DETAILED EVALUATION - {model_name}")
    print("="*80)
    
    result = results[model_name]
    y_pred = result['y_pred']
    y_pred_proba = result['y_pred_proba']
    
    # 1. Confusion Matrix
    cm = confusion_matrix(y_val, y_pred)
    
    # 2. Classification Report
    print("\n Classification Report:")
    print(classification_report(y_val, y_pred, target_names=['Legitimate', 'Fraud']))
    
    # 3. Visualizations
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Confusion Matrix
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0])
    axes[0, 0].set_title(f'Confusion Matrix - {model_name}')
    axes[0, 0].set_ylabel('True Label')
    axes[0, 0].set_xlabel('Predicted Label')
    
    # ROC Curve measures how well the model separates fraud vs legitimate transactions.
    fpr, tpr, _ = roc_curve(y_val, y_pred_proba)
    axes[0, 1].plot(fpr, tpr, label=f'AUC = {result["auc"]:.4f}')
    axes[0, 1].plot([0, 1], [0, 1], 'k--', label='Random')
    axes[0, 1].set_xlabel('False Positive Rate')
    axes[0, 1].set_ylabel('True Positive Rate')
    axes[0, 1].set_title('ROC Curve')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Precision-Recall Curve
    #curve stays high for most values, meaning:
    #When the model predicts fraud, it is usually correct (high precision).
    #It also captures most fraud cases (high recall).
    precision, recall, _ = precision_recall_curve(y_val, y_pred_proba)
    axes[1, 0].plot(recall, precision, label=f'AP = {result["avg_precision"]:.4f}')
    axes[1, 0].set_xlabel('Recall')
    axes[1, 0].set_ylabel('Precision')
    axes[1, 0].set_title('Precision-Recall Curve')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Prediction Distribution shows the predicted probabilities for each class
    axes[1, 1].hist(y_pred_proba[y_val==0], bins=50, alpha=0.5, label='Legitimate', color='green')
    axes[1, 1].hist(y_pred_proba[y_val==1], bins=50, alpha=0.5, label='Fraud', color='red')
    axes[1, 1].set_xlabel('Predicted Probability')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Prediction Probability Distribution')
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig(f'{model_name.replace(" ", "_")}_evaluation.png', dpi=300, bbox_inches='tight')
    print(f"\n Evaluation plots saved as '{model_name.replace(' ', '_')}_evaluation.png'")
    
    return cm

#=============================================================================
# STEP 8: FEATURE IMPORTANCE (for tree-based models)
#=============================================================================
#Which features (columns) are the most useful for detecting fraud?”
def plot_feature_importance(model, feature_names, model_name, top_n=20):
    """Plot feature importance for tree-based models"""
    print("\n" + "="*80)
    print(f"FEATURE IMPORTANCE - {model_name}")
    print("="*80)
    
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_ # we get the feautures
        indices = np.argsort(importances)[::-1][:top_n] # we sort them from most important to less important
        
        plt.figure(figsize=(12, 8))
        plt.title(f'Top {top_n} Feature Importances - {model_name}')
        plt.barh(range(top_n), importances[indices])
        plt.yticks(range(top_n), [feature_names[i] for i in indices])
        plt.xlabel('Importance')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig(f'{model_name.replace(" ", "_")}_feature_importance.png', dpi=300, bbox_inches='tight')
        print(f"\n Feature importance plot saved")
        
        print(f"\nTop 10 Most Important Features:")
        for i, idx in enumerate(indices[:10], 1):
            print(f"   {i}. {feature_names[idx]}: {importances[idx]:.4f}")
    else:
        print(f"\n  {model_name} doesn't support feature importance")

#=============================================================================
# STEP 9: MODEL SAVING
#=============================================================================

def save_model(model, scaler, model_name):
    """Save trained model and scaler"""
    import joblib #We import joblib, a library used to save and load machine learning objects.
    
    print("\n" + "="*80)
    print("STEP 9: MODEL SAVING")
    print("="*80)
    
    model_filename = f'{model_name.replace(" ", "_")}_model.pkl'
    scaler_filename = 'scaler.pkl'
    
    joblib.dump(model, model_filename) #save our progress
    joblib.dump(scaler, scaler_filename)
    
    print(f"\n Model saved as '{model_filename}'")
    print(f" Scaler saved as '{scaler_filename}'")
    
    return model_filename, scaler_filename

#=============================================================================
# MAIN PIPELINE
#=============================================================================

def main():
    """Execute complete ML pipeline"""
    print("\n" + "="*80)
    print("CREDIT CARD FRAUD DETECTION - COMPLETE ML PIPELINE")
    print("="*80)
    
    # Step 1: Load Data
    df = load_and_explore_data('creditcard.csv')
    
    # Step 2: EDA
    perform_eda(df)
    
    # Step 3: Feature Engineering
    df_engineered = feature_engineering(df)
    
    # Step 4: Preprocessing & Splitting
    X_train, X_val, X_test, y_train, y_val, y_test, scaler = preprocess_and_split(df_engineered)
    
    # Step 5: Handle Imbalance (try different methods)
    X_train_balanced, y_train_balanced = handle_imbalance(X_train, y_train, method='smote')
    
    # Step 6: Train Multiple Models
    results, best_model_name = train_multiple_models(X_train_balanced, y_train_balanced, X_val, y_val)
    
    # Step 7: Evaluate Best Model
    best_model = results[best_model_name]['model']
    cm = evaluate_model(best_model_name, results, y_val)
    
    # Step 8: Feature Importance
    plot_feature_importance(best_model, X_train.columns.tolist(), best_model_name)
    
    # Step 9: Test Set Evaluation
    print("\n" + "="*80)
    print("FINAL TEST SET EVALUATION")
    print("="*80)
    
    y_test_pred = best_model.predict(X_test)
    y_test_pred_proba = best_model.predict_proba(X_test)[:, 1]
    
    test_auc = roc_auc_score(y_test, y_test_pred_proba)
    test_f1 = f1_score(y_test, y_test_pred)
    test_avg_precision = average_precision_score(y_test, y_test_pred_proba)
    
    print(f"\n Final Test Metrics:")
    print(f"   AUC-ROC: {test_auc:.4f}")
    print(f"   F1-Score: {test_f1:.4f}")
    print(f"   Avg Precision: {test_avg_precision:.4f}")
    
    print("\n Test Set Classification Report:")
    print(classification_report(y_test, y_test_pred, target_names=['Legitimate', 'Fraud']))
    
    # Step 10: Save Model
    model_file, scaler_file = save_model(best_model, scaler, best_model_name)
    
    print("\n" + "="*80)
    print(" PIPELINE COMPLETE!")
    print("="*80)
    print("\n Generated Files:")
    print(f"   - {best_model_name.replace(' ', '_')}_evaluation.png")
    print(f"   - {best_model_name.replace(' ', '_')}_feature_importance.png")
    print(f"   - {model_file}")
    print(f"   - {scaler_file}")
    
    return best_model, scaler, results

if __name__ == "__main__":
    # Run the complete pipeline
    best_model, scaler, results = main()
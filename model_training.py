# DATA SCIENCE FINAL LAB EXAM – VARIANT 4
# Titanic Dataset - Binary Classification with PCA

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# TASK A: DATA LOADING, CLEANING & EXPLORATION (2 Marks)
# ============================================================================

print("="*80)
print("TASK A: DATA LOADING, CLEANING & EXPLORATION")
print("="*80)

# 1. Load the Titanic dataset
df = sns.load_dataset('titanic')
print("\n1. Dataset loaded successfully!")
print(f"   Total rows: {len(df)}")

# 2. Display the shape BEFORE splitting into X and y
print(f"\n2. Original Dataset Shape: {df.shape}")

# Separate features and target
y = df['survived']
X = df.drop('survived', axis=1)

print(f"   Features (X): {X.shape}")
print(f"   Target (y): {y.shape}")

# 3. Display the first 5 rows
print("\n3. First 5 rows of the dataset:")
print(df.head())

# 4. Handle missing values
print("\n4. Missing values BEFORE handling:")
missing_before = df.isnull().sum()
print(missing_before[missing_before > 0])

# Fill missing values
df['age'].fillna(df['age'].median(), inplace=True)
df['embarked'].fillna(df['embarked'].mode()[0], inplace=True)
df['fare'].fillna(df['fare'].median(), inplace=True)

# Drop columns with too many missing values or not useful
df = df.drop(['deck', 'embark_town', 'alive', 'class', 'who', 'adult_male', 'alone'], axis=1, errors='ignore')

print("\n   Missing values AFTER handling:")
missing_after = df.isnull().sum()
print(missing_after[missing_after > 0])
if missing_after.sum() == 0:
    print("   ✓ All missing values handled!")

# 5. Display class distribution
print("\n5. Class Distribution (survived):")
class_dist = df['survived'].value_counts().sort_index()
print(class_dist)
print(f"\n   Percentage Distribution:")
class_dist_pct = df['survived'].value_counts(normalize=True).sort_index() * 100
print(class_dist_pct)

# ============================================================================
# TASK B: PREPROCESSING, SCALING & STRATIFIED SPLIT (2 Marks)
# ============================================================================

print("\n" + "="*80)
print("TASK B: PREPROCESSING, SCALING & STRATIFIED SPLIT")
print("="*80)

# 1. Select features for modeling
print("\n1. Converting categorical features to numeric (One-Hot Encoding)...")

# Select only the required features
features_to_use = ['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked']
df_model = df[features_to_use + ['survived']].copy()

# Check for any remaining missing values
if df_model.isnull().sum().sum() > 0:
    print("   Warning: Found missing values, filling them...")
    df_model = df_model.fillna(df_model.median(numeric_only=True))
    df_model['embarked'].fillna(df_model['embarked'].mode()[0], inplace=True)

# Convert categorical features using one-hot encoding
df_encoded = pd.get_dummies(df_model, columns=['sex', 'embarked'], drop_first=True)

# Convert pclass to dummy variables
df_encoded = pd.get_dummies(df_encoded, columns=['pclass'], drop_first=True)

print(f"   Shape after encoding: {df_encoded.shape}")
print(f"   Columns: {list(df_encoded.columns)}")

# Separate features and target
X = df_encoded.drop('survived', axis=1)
y = df_encoded['survived']

print(f"\n   Final feature set shape: {X.shape}")
print(f"   Number of features: {X.shape[1]}")

# 2. Standardize all numeric features
print("\n2. Standardizing features using StandardScaler...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

print(f"   Features scaled successfully!")
print(f"   Sample statistics after scaling:")
print(f"   Mean (should be ~0): {X_scaled.mean().mean():.6f}")
print(f"   Std (should be ~1): {X_scaled.std().mean():.6f}")

# 3. Split dataset using stratified sampling (80-20 split)
print("\n3. Splitting dataset (80% train, 20% test) with stratified sampling...")
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

print(f"   Training set: {X_train.shape}")
print(f"   Testing set: {X_test.shape}")
print(f"\n   Training class distribution:")
print(y_train.value_counts().sort_index())
print(f"\n   Testing class distribution:")
print(y_test.value_counts().sort_index())

# ============================================================================
# TASK C: PCA ANALYSIS (3 Marks)
# ============================================================================

print("\n" + "="*80)
print("TASK C: PCA ANALYSIS")
print("="*80)

# 1. Apply PCA to determine variance thresholds
print("\n1. Determining number of components for variance thresholds...")

# Fit PCA with all components
pca_full = PCA()
pca_full.fit(X_train)

# Calculate cumulative variance
cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)

# Find components for 95% and 99% variance
n_components_95 = np.argmax(cumulative_variance >= 0.95) + 1
n_components_99 = np.argmax(cumulative_variance >= 0.99) + 1

print(f"\n   Components for 95% variance: {n_components_95}")
print(f"   Actual variance explained: {cumulative_variance[n_components_95-1]*100:.2f}%")
print(f"\n   Components for 99% variance: {n_components_99}")
print(f"   Actual variance explained: {cumulative_variance[n_components_99-1]*100:.2f}%")

# 2. Transform data using 95% variance PCA
print(f"\n2. Transforming data using PCA with {n_components_95} components (95% variance)...")
pca_95 = PCA(n_components=n_components_95)
X_train_pca = pca_95.fit_transform(X_train)
X_test_pca = pca_95.transform(X_test)

print(f"   Training set shape after PCA: {X_train_pca.shape}")
print(f"   Testing set shape after PCA: {X_test_pca.shape}")

# 3. Display explained variance values
print("\n3. Explained Variance Values:")
print(f"\n   Individual explained variance ratio:")
for i, var in enumerate(pca_95.explained_variance_ratio_, 1):
    cum_var = cumulative_variance[i-1]
    print(f"   PC{i}: {var:.4f} ({var*100:.2f}%) | Cumulative: {cum_var:.4f} ({cum_var*100:.2f}%)")

print(f"\n   Total variance explained: {sum(pca_95.explained_variance_ratio_):.4f} ({sum(pca_95.explained_variance_ratio_)*100:.2f}%)")

# Visualize explained variance
try:
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.bar(range(1, len(pca_95.explained_variance_ratio_) + 1), 
            pca_95.explained_variance_ratio_)
    plt.xlabel('Principal Component')
    plt.ylabel('Explained Variance Ratio')
    plt.title('Explained Variance by Component')
    plt.xticks(range(1, len(pca_95.explained_variance_ratio_) + 1))
    plt.grid(axis='y', alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, 'bo-')
    plt.axhline(y=0.95, color='r', linestyle='--', label='95% Variance', linewidth=2)
    plt.axhline(y=0.99, color='g', linestyle='--', label='99% Variance', linewidth=2)
    plt.axvline(x=n_components_95, color='r', linestyle=':', alpha=0.5)
    plt.axvline(x=n_components_99, color='g', linestyle=':', alpha=0.5)
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('Cumulative Explained Variance')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('pca_analysis.png', dpi=100, bbox_inches='tight')
    print("\n   ✓ PCA visualization saved as 'pca_analysis.png'")
    plt.close()
except Exception as e:
    print(f"\n   Warning: Could not save plot: {e}")

# ============================================================================
# TASK D: MODEL TRAINING & EVALUATION (3 Marks)
# ============================================================================

print("\n" + "="*80)
print("TASK D: MODEL TRAINING & EVALUATION")
print("="*80)

# Dictionary to store results
model_results = {}

# 1. Logistic Regression
print("\n" + "="*80)
print("1. LOGISTIC REGRESSION")
print("="*80)
lr_model = LogisticRegression(random_state=42, max_iter=1000)
lr_model.fit(X_train_pca, y_train)
y_pred_lr = lr_model.predict(X_test_pca)

lr_accuracy = accuracy_score(y_test, y_pred_lr)
lr_cm = confusion_matrix(y_test, y_pred_lr)

print(f"\n   Test Accuracy: {lr_accuracy:.4f} ({lr_accuracy*100:.2f}%)")
print(f"\n   Confusion Matrix:")
print(lr_cm)
print(f"\n   Classification Report:")
print(classification_report(y_test, y_pred_lr, target_names=['Not Survived', 'Survived']))

model_results['Logistic Regression'] = {
    'model': lr_model,
    'accuracy': lr_accuracy,
    'confusion_matrix': lr_cm,
    'predictions': y_pred_lr
}

# 2. Random Forest Classifier
print("\n" + "="*80)
print("2. RANDOM FOREST CLASSIFIER")
print("="*80)
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_pca, y_train)
y_pred_rf = rf_model.predict(X_test_pca)

rf_accuracy = accuracy_score(y_test, y_pred_rf)
rf_cm = confusion_matrix(y_test, y_pred_rf)

print(f"\n   Test Accuracy: {rf_accuracy:.4f} ({rf_accuracy*100:.2f}%)")
print(f"\n   Confusion Matrix:")
print(rf_cm)
print(f"\n   Classification Report:")
print(classification_report(y_test, y_pred_rf, target_names=['Not Survived', 'Survived']))

model_results['Random Forest'] = {
    'model': rf_model,
    'accuracy': rf_accuracy,
    'confusion_matrix': rf_cm,
    'predictions': y_pred_rf
}

# 3. Support Vector Machine (SVM)
print("\n" + "="*80)
print("3. SUPPORT VECTOR MACHINE (SVM)")
print("="*80)
svm_model = SVC(kernel='rbf', random_state=42, probability=True)
svm_model.fit(X_train_pca, y_train)
y_pred_svm = svm_model.predict(X_test_pca)

svm_accuracy = accuracy_score(y_test, y_pred_svm)
svm_cm = confusion_matrix(y_test, y_pred_svm)

print(f"\n   Test Accuracy: {svm_accuracy:.4f} ({svm_accuracy*100:.2f}%)")
print(f"\n   Confusion Matrix:")
print(svm_cm)
print(f"\n   Classification Report:")
print(classification_report(y_test, y_pred_svm, target_names=['Not Survived', 'Survived']))

model_results['SVM'] = {
    'model': svm_model,
    'accuracy': svm_accuracy,
    'confusion_matrix': svm_cm,
    'predictions': y_pred_svm
}

# Identify best model
print("\n" + "="*80)
print("MODEL COMPARISON & BEST MODEL SELECTION")
print("="*80)

print("\nModel Accuracies:")
print("-" * 50)
for model_name, results in model_results.items():
    print(f"   {model_name:25s}: {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")

best_model_name = max(model_results, key=lambda x: model_results[x]['accuracy'])
best_model = model_results[best_model_name]['model']
best_accuracy = model_results[best_model_name]['accuracy']

print(f"\n{'='*80}")
print(f"✓ BEST MODEL: {best_model_name}")
print(f"✓ ACCURACY: {best_accuracy:.4f} ({best_accuracy*100:.2f}%)")
print(f"✓ JUSTIFICATION: {best_model_name} achieved the highest test accuracy")
print(f"  with balanced performance across both classes.")
print(f"{'='*80}")

# Visualize model comparison
try:
    plt.figure(figsize=(10, 6))
    models = list(model_results.keys())
    accuracies = [model_results[m]['accuracy'] for m in models]
    colors = ['#3b82f6', '#10b981', '#f59e0b']

    bars = plt.bar(models, accuracies, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    plt.ylabel('Accuracy', fontsize=12, fontweight='bold')
    plt.xlabel('Model', fontsize=12, fontweight='bold')
    plt.title('Model Comparison - Test Accuracy', fontsize=14, fontweight='bold')
    plt.ylim([0.7, 0.9])
    plt.grid(axis='y', alpha=0.3, linestyle='--')

    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                 f'{acc:.4f}\n({acc*100:.2f}%)',
                 ha='center', va='bottom', fontweight='bold', fontsize=10)

    # Highlight best model
    best_idx = models.index(best_model_name)
    bars[best_idx].set_edgecolor('red')
    bars[best_idx].set_linewidth(3)

    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=100, bbox_inches='tight')
    print("\n✓ Model comparison visualization saved as 'model_comparison.png'")
    plt.close()
except Exception as e:
    print(f"\nWarning: Could not save plot: {e}")

# Save the best model and preprocessing objects
print("\n" + "="*80)
print("SAVING MODEL AND PREPROCESSING OBJECTS")
print("="*80)

try:
    joblib.dump(best_model, 'best_model.pkl')
    print("✓ Saved: best_model.pkl")
    
    joblib.dump(scaler, 'scaler.pkl')
    print("✓ Saved: scaler.pkl")
    
    joblib.dump(pca_95, 'pca.pkl')
    print("✓ Saved: pca.pkl")
    
    joblib.dump(X.columns.tolist(), 'feature_names.pkl')
    print("✓ Saved: feature_names.pkl")
    
    # Also save the best model name for reference
    joblib.dump(best_model_name, 'best_model_name.pkl')
    print("✓ Saved: best_model_name.pkl")
    
    print("\n✓ All files saved successfully!")
    
except Exception as e:
    print(f"\nError saving files: {e}")

print("\n" + "="*80)
print("LAB TASKS A-D COMPLETED SUCCESSFULLY!")
print("="*80)
print("\n📊 Summary:")
print(f"   - Dataset: {len(df)} passengers, {X.shape[1]} features")
print(f"   - PCA Components: {n_components_95} (95% variance)")
print(f"   - Best Model: {best_model_name} ({best_accuracy*100:.2f}%)")
print(f"   - Files Generated: 5 PKL files, 2 PNG plots")
print("\n🚀 Next Step: Run the Streamlit app for Task E")
print("   Command: streamlit run app.py")
print("="*80)
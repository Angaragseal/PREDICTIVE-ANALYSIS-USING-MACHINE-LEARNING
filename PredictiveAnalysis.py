import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score, roc_curve, roc_auc_score, ConfusionMatrixDisplay

# Load dataset from sklearn
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

print("Feature columns:", list(X.columns))
print("Target classes:", np.unique(y))

# Feature selection - pick top 10 features
selector = SelectKBest(score_func=f_classif, k=10)
X_new = selector.fit_transform(X, y)
selected_features = X.columns[selector.get_support()]
print("\nSelected Features:", selected_features)

X_selected = X[selected_features]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(),
    "K-Nearest Neighbors": KNeighborsClassifier()
}

results = {}
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n{name} Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred))
    results[name] = {
        'model': model,
        'accuracy': acc,
        'y_pred': y_pred
    }

# Select best model by accuracy
best_model_name, best_result = max(results.items(), key=lambda x: x[1]['accuracy'])
best_model = best_result['model']

# Predict probabilities for ROC
y_probs = best_model.predict_proba(X_test_scaled)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_probs)
roc_auc = roc_auc_score(y_test, y_probs)

# Plot Confusion Matrix and ROC Curve side by side
fig, ax = plt.subplots(1, 2, figsize=(12, 5))

# Confusion Matrix
ConfusionMatrixDisplay.from_estimator(best_model, X_test_scaled, y_test, ax=ax[0], cmap='Blues', colorbar=False)
ax[0].set_title(f"{best_model_name} - Confusion Matrix")

# ROC Curve
ax[1].plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}", color='darkorange')
ax[1].plot([0, 1], [0, 1], 'k--')
ax[1].set_title(f"{best_model_name} - ROC Curve")
ax[1].set_xlabel("False Positive Rate")
ax[1].set_ylabel("True Positive Rate")
ax[1].legend(loc="lower right")

plt.tight_layout()
plt.show()


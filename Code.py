import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

df = pd.read_csv("diabetic_data.csv")
ids_mapping = pd.read_csv("IDS_mapping.csv")

# ================================================
# 0. Preprocessing
# ================================================
# Predict whether or not hospital patient will be readmitted within 30 days

# Replace "?" with NaN for simplicity
df = df.replace("?", np.nan)

# Target variable includes those recently readmitted (within 30 days)
df["target"] = df["readmitted"].apply(lambda x: 1 if x == "<30" else 0)
df = df.drop(["readmitted"], axis=1)

# Drop columns not useful for prediction (all "id" columns)
df = df.drop(columns=[col for col in ids_mapping.columns if col in df.columns])

# Convert categorical to numeric
df = pd.get_dummies(df, drop_first=True)

# Split features and target
X = df.drop("target", axis=1)
y = df["target"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ================================================
# 1. k-Nearest Neighbors (kNN)
# ================================================
print("\n-- k-Nearest Neighbors --")
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)
y_pred_knn = knn.predict(X_test_scaled)

print("Accuracy:", accuracy_score(y_test, y_pred_knn))
print("Classification Report:\n", classification_report(y_test, y_pred_knn))

# ================================================
# 2. Logistic Regression
# ================================================
print("\n-- Logistic Regression --")
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train_scaled, y_train)
y_pred_lr = lr.predict(X_test_scaled)

print("Accuracy:", accuracy_score(y_test, y_pred_lr))
print("Classification Report:\n", classification_report(y_test, y_pred_lr))

# ================================================
# 3. Feedforward Neural Network (scikit-learn MLP)
# ================================================
print("\n-- Feedforward Neural Network (MLP) --")
mlp = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=1000, random_state=42)
mlp.fit(X_train_scaled, y_train)
y_pred_mlp = mlp.predict(X_test_scaled)

print("Accuracy:", accuracy_score(y_test, y_pred_mlp))
print("Classification Report:\n", classification_report(y_test, y_pred_mlp))

# ================================================
# 4. Confusion Matrices
# ================================================
def plot_confusion(title, y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues")
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

plot_confusion("kNN Confusion Matrix", y_test, y_pred_knn)
plot_confusion("Logistic Regression Confusion Matrix", y_test, y_pred_lr)
plot_confusion("Neural Network Confusion Matrix", y_test, y_pred_mlp)

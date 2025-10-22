import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

OUT_DIR = r"C:\Users\HP\Downloads\cancer_pca_project"
PLOT_PATH = os.path.join(OUT_DIR, "pca_2components.png")
REPORT_PATH = os.path.join(OUT_DIR, "logreg_report.txt")

os.makedirs(OUT_DIR, exist_ok=True)

# Load dataset
data = load_breast_cancer()
X = data.data
y = data.target
feature_names = data.feature_names

# Standardize
scaler = StandardScaler()
Xs = scaler.fit_transform(X)

# PCA to 2 components
pca = PCA(n_components=2)
X_pca = pca.fit_transform(Xs)

# Save PCA dataframe
df_pca = pd.DataFrame(X_pca, columns=["PC1", "PC2"])
df_pca['target'] = y

# Plot PCA scatter
plt.figure(figsize=(8,6))
sns.scatterplot(data=df_pca, x='PC1', y='PC2', hue='target', palette='coolwarm', alpha=0.7)
plt.title('Breast Cancer dataset PCA (2 components)')
plt.tight_layout()
plt.savefig(PLOT_PATH)
plt.close()
print('Saved PCA plot to', PLOT_PATH)

# Optional: Logistic regression on PCA components
X_train, X_test, y_train, y_test = train_test_split(df_pca[['PC1','PC2']], y, test_size=0.2, random_state=42)
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)

with open(REPORT_PATH, 'w') as f:
    f.write(f'Logistic Regression on 2 PCA components\nAccuracy: {acc:.4f}\n\n')
    f.write(classification_report(y_test, y_pred))

print('Saved logistic regression report to', REPORT_PATH)
print('Done')

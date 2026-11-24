"""Task 1: Iris Flower Classification"""
import pandas as pd, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

SEED = 42
df = pd.read_csv("Iris.csv").drop(columns="Id")
df["Species"] = df["Species"].str.replace("Iris-", "", regex=False)
features = ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]

print("Shape:", df.shape)
print("Missing values:\n", df.isna().sum().sum(), "| Duplicates:", df.duplicated().sum())
print(df.groupby("Species")[features].mean().round(2))

# ---- EDA ----
sns.pairplot(df, hue="Species", corner=True)
plt.savefig("images/pairplot.png", dpi=120, bbox_inches="tight"); plt.close()

plt.figure(figsize=(6, 5))
sns.heatmap(df[features].corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Feature correlations"); plt.tight_layout()
plt.savefig("images/correlation.png", dpi=120); plt.close()

# ---- Split ----
X, y = df[features], df["Species"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=SEED)

# ---- Compare models with 5-fold CV on the training set ----
models = {
    "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=200)),
    "KNN (k=5)": make_pipeline(StandardScaler(), KNeighborsClassifier(5)),
    "SVM (RBF)": make_pipeline(StandardScaler(), SVC()),
    "Decision Tree": DecisionTreeClassifier(random_state=SEED),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=SEED),
}
cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
rows = []
for name, m in models.items():
    s = cross_val_score(m, X_train, y_train, cv=cv)
    m.fit(X_train, y_train)
    rows.append((name, s.mean(), s.std(), accuracy_score(y_test, m.predict(X_test))))
res = pd.DataFrame(rows, columns=["Model", "CV Acc (mean)", "CV Acc (std)", "Test Acc"]).round(4)
print("\n", res.to_string(index=False))
res.to_csv("model_comparison.csv", index=False)

# ---- Final model evaluation ----
best_name = res.sort_values(["CV Acc (mean)", "Test Acc"], ascending=False).iloc[0]["Model"]
best = models[best_name]
pred = best.predict(X_test)
print(f"\nSelected model: {best_name}  |  Test accuracy: {accuracy_score(y_test, pred):.4f}")
print(classification_report(y_test, pred))

fig, ax = plt.subplots(figsize=(5, 4))
ConfusionMatrixDisplay(confusion_matrix(y_test, pred, labels=best.classes_),
                       display_labels=best.classes_).plot(ax=ax, cmap="Blues", colorbar=False)
ax.set_title(f"Confusion matrix - {best_name}"); plt.tight_layout()
plt.savefig("images/confusion_matrix.png", dpi=120); plt.close()

# Feature importance from random forest
rf = models["Random Forest"]
imp = pd.Series(rf.feature_importances_, index=features).sort_values()
imp.plot.barh(figsize=(6, 3.5), title="Random Forest feature importance")
plt.tight_layout(); plt.savefig("images/feature_importance.png", dpi=120); plt.close()
print("\nFeature importance:\n", imp.sort_values(ascending=False).round(3))

# ---- Predict a new sample ----
sample = pd.DataFrame([[5.9, 3.0, 5.1, 1.8]], columns=features)
print("\nSample", sample.values.tolist(), "->", best.predict(sample)[0])

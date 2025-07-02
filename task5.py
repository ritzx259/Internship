import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("/Users/ritzmk/PycharmProjects/pythonProject/internship/heart.csv")

X = df.drop('target', axis=1)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

dt = DecisionTreeClassifier(max_depth=4, random_state=42)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)
acc_dt = accuracy_score(y_test, y_pred_dt)

plt.figure(figsize=(15, 8))
plot_tree(dt, feature_names=df.columns[:-1], class_names=["No", "Yes"], filled=True)
plt.show()

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
acc_rf = accuracy_score(y_test, y_pred_rf)

importances = rf.feature_importances_
features = df.columns[:-1]
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10, 6))
sns.barplot(x=importances[indices], y=features[indices])
plt.title("Feature Importances")
plt.show()

cv_dt = cross_val_score(dt, X, y, cv=5).mean()
cv_rf = cross_val_score(rf, X, y, cv=5).mean()

print(f"Decision Tree Accuracy: {acc_dt}")
print(f"Random Forest Accuracy: {acc_rf}")
print(f"Decision Tree CV Score: {cv_dt}")
print(f"Random Forest CV Score: {cv_rf}")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

# Step 1: Load the dataset from CSV
df = pd.read_csv("/Users/ritzmk/PycharmProjects/pythonProject/internship/Iris.csv")

# Drop the 'Id' column if present
if 'Id' in df.columns:
    df = df.drop('Id', axis=1)

# Encode target labels
le = LabelEncoder()
df['Species'] = le.fit_transform(df['Species'])

# Split features and target
X = df.drop('Species', axis=1)
y = df['Species']

# Normalize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Step 2: Try different values of K
k_values = range(1, 11)
accuracies = []

for k in k_values:
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    accuracies.append(acc)

# Step 3: Plot accuracy vs K
plt.plot(k_values, accuracies, marker='o')
plt.title('KNN Accuracy vs K')
plt.xlabel('Number of Neighbors (K)')
plt.ylabel('Accuracy')
plt.grid(True)
plt.show()

# Step 4: Train final model with best K
best_k = k_values[np.argmax(accuracies)]
print(f"Best K: {best_k} with accuracy {max(accuracies):.2f}")

final_model = KNeighborsClassifier(n_neighbors=best_k)
final_model.fit(X_train, y_train)
y_pred = final_model.predict(X_test)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
disp.plot()
plt.title("Confusion Matrix")
plt.show()

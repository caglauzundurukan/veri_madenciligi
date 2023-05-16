import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt


# Veri setini yukle
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/ionosphere/ionosphere.data'
df = pd.read_csv(url, header=None)

# Sinif etiketlerini kodla
df[34] = pd.Categorical(df[34])
df[34] = df[34].cat.codes

# Ozellikleri ve sinif etiketlerini ayir
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# Ozellikleri Olceklendir
sc = StandardScaler()
X = sc.fit_transform(X)

# Veri setini egitim ve test kumelerine ayir
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=40)

# Modeli egit
knn = KNeighborsClassifier(n_neighbors=5, weights='distance')
knn.fit(X_train, y_train)

# Test veri setiyle modeli dogrula ve performansi hesapla
y_pred = knn.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
sensitivity = recall_score(y_test, y_pred, pos_label=1)
specificity = recall_score(y_test, y_pred, pos_label=0)

results = {"Accuracy": accuracy, "Precision": precision, "Recall": recall, "F1 Score": f1, "Sensitivity": sensitivity, "Specificity": specificity }

fig, ax = plt.subplots()
ax.bar(results.keys(), results.values())
ax.set_ylabel('Score')
ax.set_title('Model Performance')

# Performansi ekrana yazdir
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("Sensitivity:", sensitivity)
print("Specificity:", specificity)
plt.show()

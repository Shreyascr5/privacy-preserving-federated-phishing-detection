import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)
import joblib


df=pd.read_csv("dataset/processed/phishing_clean.csv")

print(df.shape)


X=df.drop("Result",axis=1)
y=df["Result"]


X_train, X_test, y_train, y_test= train_test_split(X,y,test_size=0.2,random_state=42)
scaler=StandardScaler()

X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)



model=LogisticRegression(
    max_iter=1000
)


model.fit(X_train, y_train)
y_pred=model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))

#confusion matrix
cm=confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(6,4))

sns.heatmap(
    cm,
    annot=True,
    fmt="d"
)

plt.title(
    "Confusion Matrix"
)

plt.show()



joblib.dump(
    model,
    "models/baseline_model.pkl"
)

joblib.dump(
    scaler,
    "models/scaler.pkl"
)




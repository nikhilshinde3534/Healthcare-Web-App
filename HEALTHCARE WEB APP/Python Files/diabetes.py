import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

data = pd.read_csv('dataset/diabetes.csv')

x = data.drop(['Outcome'], 1)
y = data[['Outcome']]

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.3, random_state=42)
len(x_train), len(x_test), len((y_train)), len(y_test)

model = LogisticRegression()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)

acc = accuracy_score(y_test, y_pred)
print("Accuracy is: ", acc)

filename = 'trained_models/diabetes_model.joblib'
joblib.dump(model, filename)

import pandas as pd
from sklearn.model_selection import train_test_split
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

data = pd.read_csv('dataset/cancer.csv')
cols_to_drop = ['id', 'Unnamed: 32']
data = data.drop(cols_to_drop, axis='columns')

# Convert Character into a number
data.diagnosis = data.diagnosis.map({"M": 0, "B": 1})

x = data.drop(['diagnosis'], axis='columns')
y = data[['diagnosis']]


x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.25, random_state=42)
len(x_train), len(x_test), len((y_train)), len(y_test)

# create model
model = LogisticRegression()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)


acc = accuracy_score(y_test, y_pred)
print("Accuracy is: ", acc)

# save model
filename = 'trained_models/cancer_model.joblib'
joblib.dump(model, filename)

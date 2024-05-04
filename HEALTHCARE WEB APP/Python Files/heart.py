import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib
import warnings
warnings.filterwarnings('ignore')

# read the dataset
data = pd.read_csv('dataset/heart.csv')

cols_to_drop = ['age', 'sex', 'ca', 'oldpeak', 'thal', 'slope']

data = data.drop(cols_to_drop, axis='columns')


x = data.drop('target', axis='columns')
y = data[['target']]


scaler = StandardScaler()
x = scaler.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.25, random_state=42)
len(x_train), len(x_test), len((y_train)), len(y_test)

lr = LogisticRegression()
lr.fit(x_train, y_train)
y_pred = lr.predict(x_test)

acc = accuracy_score(y_test, y_pred)
print("Accuracy is: ", acc)

# save the trained model
filename = 'trained_models/heart_model.joblib'
joblib.dump(lr, filename)

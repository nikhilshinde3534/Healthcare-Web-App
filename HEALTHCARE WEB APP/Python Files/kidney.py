import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
import joblib

# read the datset
df = pd.read_csv("dataset/kidney_disease.csv")

# shape of the dataset
print("Shape of the data:", df.shape)
cols_to_use = ['bp', 'sg', 'al', 'su', 'rbc', 'pc', 'pcc', 'classification']
df = df[cols_to_use]

# replace the values
df[['rbc', 'pc']] = df[['rbc', 'pc']].replace(
    to_replace={'abnormal': 1, 'normal': 0})
df[['pcc']] = df[['pcc']].replace(
    to_replace={'present': 1, 'notpresent': 0})

df['classification'] = df['classification'].replace(
    to_replace={'ckd': 1, 'ckd\t': 1, 'notckd': 0, 'no': 0})


# remove nan/Null value
df = df.dropna()


# Split the dataset into dependent and independent variables
X = df.drop(["classification"], axis=1)
y = df[["classification"]]


# Feature Scalling

scaler = MinMaxScaler()

columns = X.columns
X[columns] = scaler.fit_transform(X)

# split the data into training and testing set
x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Build the model
rf = RandomForestClassifier()
rf.fit(x_train, y_train)
y_pred = rf.predict(x_test)

acc = accuracy_score(y_test, y_pred)
print("Accuracy: ", acc)


joblib.dump(rf, "trained_models/kidney_model.joblib")

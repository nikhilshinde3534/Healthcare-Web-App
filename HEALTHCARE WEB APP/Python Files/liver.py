# import required libraries

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib


# read the datset
data = pd.read_csv('dataset/indian_liver_patient.csv')

# fill null values
data["Albumin_and_Globulin_Ratio"] = data["Albumin_and_Globulin_Ratio"].fillna(
    np.mean(data["Albumin_and_Globulin_Ratio"]))

# convert categorical data into numeric
data["Gender"] = data["Gender"].map({"Male": 1, "Female": 0})

# separate out dependent and independent variables
x = data.drop(['Dataset'], 1)
y = data[['Dataset']]

# split the data into trianing and testing set
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.25, random_state=42)
len(x_train), len(x_test), len((y_train)), len(y_test)

# create a model
rf = RandomForestClassifier()
rf.fit(x_train, y_train)
y_pred = rf.predict(x_test)

# accuracy of the model
accuracy_score = accuracy_score(y_test, y_pred)
print("Accuray score : ", accuracy_score)

# save the trained model into .joblib format
filename = 'trained_models/liver_model.joblib'
joblib.dump(rf, filename)

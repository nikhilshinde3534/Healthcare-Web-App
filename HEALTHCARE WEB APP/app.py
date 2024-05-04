# import required libraries
from flask import Flask
from flask import render_template
from flask import request
import numpy as np
import pandas as pd
import joblib


app = Flask(__name__, template_folder='templates')


# Home page
@app.route('/')
def home():
    return render_template('home.html')


# loading trained model
cmodel = joblib.load('trained_models/cancer_model.joblib')


@app.route("/cancer")
def cancer():
    return render_template("cancer.html")


@app.route('/cancer_predict', methods=['POST'])
def cancer_predict():
    input_features = [float(x) for x in request.form.values()]
    features_value = [np.array(input_features)]

    features_name = ['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean',
                     'smoothness_mean', 'compactness_mean', 'concavity_mean',
                     'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean',
                     'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
                     'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se',
                     'fractal_dimension_se', 'radius_worst', 'texture_worst',
                     'perimeter_worst', 'area_worst', 'smoothness_worst',
                     'compactness_worst', 'concavity_worst', 'concave points_worst',
                     'symmetry_worst', 'fractal_dimension_worst']

    df = pd.DataFrame(features_value)
    output = cmodel.predict(df)

    if output == 1:
        res_val = " Breast Cancer.."
    else:
        res_val = " No Breast Cancer"

    return render_template('result.html', prediction_text='Patient has a {}'.format(res_val))


# diabetes model
dmodel = joblib.load('trained_models/diabetes_model.joblib')


@app.route("/diabetes")
def diabetes():
    return render_template("diabetes.html")


@app.route('/diabetes_predict', methods=['POST'])
def diabetes_predict():
    input_features = [float(x) for x in request.form.values()]
    features_value = [np.array(input_features)]

    features_name = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
                     'BMI', 'DiabetesPedigreeFunction', 'Age']

    df = pd.DataFrame(features_value, columns=features_name)
    output = dmodel.predict(df)
    if output == 0:
        res_val = "Patient doesn't have a Diabetes "
    else:
        res_val = "Patient has the Diabetes"

    return render_template('result.html', prediction_text='{}'.format(res_val))


# Heart Disease Model
hmodel = joblib.load('trained_models/heart_model.joblib')


@app.route("/heart")
def heart():
    return render_template("heart.html")


@app.route('/heart_predict', methods=['POST'])
def heart_predict():

    cp = request.form.get("cp")
    trestbps = request.form.get("trestbps")
    chol = request.form.get("chol")
    fbs = request.form.get("fbs")
    restecg = request.form.get("restecg")
    thalach = request.form.get("thalach")
    exang = request.form.get("exang")

    if((cp == "#")|(fbs == "#") | (exang == "#")):
        return render_template("heart.html", error_message="Please Select All the fields!")
    
    input_features = [cp, trestbps, chol, fbs, restecg, thalach, exang]

    features_value = [np.array(input_features)]

    features_name = ['cp', 'trestbps', 'chol',
                     'fbs', 'restecg', 'thalach', 'exang']

    df = pd.DataFrame(features_value, columns=features_name)
    output = hmodel.predict(df)

    if output == 0:
        res_val = "Patient doesn't have a heart disease!"
    else:
        res_val = "Patient has the heart disease"

    return render_template('result.html', prediction_text='{}'.format(res_val))


# liver Model
lmodel = joblib.load('trained_models/liver_model.joblib')


@app.route("/liver")
def liver():
    return render_template("liver.html")


@app.route("/liver_predict", methods=['POST'])
def liver_predict():

    Age = request.form.get("Age")
    Gender = request.form.get("Gender")
    Total_Bilirubin = request.form.get("Total_Bilirubin")
    Direct_Bilirubin = request.form.get("Direct_Bilirubin")
    Alkaline_Phosphotase = request.form.get("Alkaline_Phosphotase")
    Alamine_Aminotransferase = request.form.get("Alamine_Aminotransferase")
    Aspartate_Aminotransferase = request.form.get("Aspartate_Aminotransferase")
    Total_Protiens = request.form.get("Total_Protiens")
    Albumin = request.form.get("Albumin")
    Albumin_and_Globulin_Ratio = request.form.get("Albumin_and_Globulin_Ratio")

    if(Gender == "#"):
        return render_template("liver.html", error_message="Please Select Your Gender!")

    input_features = [Age, Gender, Total_Bilirubin, Direct_Bilirubin, Alkaline_Phosphotase, Alamine_Aminotransferase, Aspartate_Aminotransferase,
                      Total_Protiens, Albumin, Albumin_and_Globulin_Ratio]

    features_value = [np.array(input_features)]

    features_name = ['Age', 'Gender', 'Total_Bilirubin', 'Direct_Bilirubin',
                     'Alkaline_Phosphotase', 'Alamine_Aminotransferase',
                     'Aspartate_Aminotransferase', 'Total_Protiens', 'Albumin',
                     'Albumin_and_Globulin_Ratio']

    df = pd.DataFrame(features_value, columns=features_name)
    output = lmodel.predict(df)

    if output == 1:
        value = "You have a Liver disease, Please Consult a doctor. Immediately!" 
    else:
        value = "No worries! You don't have a Liver disease!"

    return render_template('result.html', prediction_text='{}'.format(value))


# load trained Kindey Model
kmodel = joblib.load('trained_models/kidney_model.joblib')


@app.route("/kidney")
def predict_kidney():
    return render_template("kidney.html")


@app.route("/kidney_predict", methods=['POST'])
def kidney_predict():
    input_features = [float(x) for x in request.form.values()]
    features_value = [np.array(input_features)]
    features_name = ['bp', 'sg', 'al', 'su', 'rbc', 'pc', 'pcc']
    df = pd.DataFrame(features_value, columns=features_name)
    output = kmodel.predict(df)

    if output == 1:
        value = "You have a symptoms of the disease"
    else:
        value = "You don't have a symptoms of the disease"

    return render_template('result.html', prediction_text='{}'.format(value))


if __name__ == "__main__":
    app.run(debug=True)

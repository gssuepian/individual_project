# ***DIABETES PREDICTION WEB APPLICATION*** ⚕️
A simple web application for predicting the likelihood of diabetes using machine learning, built with Streamlit.

[![Streamlit App](https://img.shields.io/badge/Streamlit-Deployed-0A1F44?logo=streamlit)](https://individualproject-siriyakorn-suepiantham.streamlit.app/)  

Click the icon above or [here](https://individualproject-siriyakorn-suepiantham.streamlit.app/) to view the app. 🌐

# ***ABOUT*** 🔍
This web application was designed as part of a university project for a Machine Learning Class. It allows users to input medical data and predicts whether the individual is likely to have diabetes. The model is trained on the Pima Indians Diabetes Dataset and leverages a Random Forest machine learning model for predictions. This project was created using Streamlit for a user-friendly front-end experience. 

### ***DISCLAIMER!*** ⚠️
This application is for informational and educational purposes only and should not be considered medical advice. For any health-related concerns, please consult a qualified healthcare professional. 🩺 The predictions generated by this application are based on a model that was trained on only 768 rows of data and is not intended to replace professional medical assessments.

# ***MODEL INFORMATION*** 🧠

### 📊 ***DATASET*** 
The model is trained on the [*Pima Indians Diabetes Dataset*](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database). This dataset contains only 768 rows of data. This is an insufficient amount of data to be training a diabetes prediction model on. Nevertheless, as aforementioned, this project is only for educational purposes in the field of Machine Learning. For better results, a much larger dataset should be utilized.

* The dataset contains 8 feature variables and one target variable `Outcome`:

| **Feature**                    | **Description**                                                                                             |
|--------------------------------|-------------------------------------------------------------------------------------------------------------|
| ***Pregnancies***               | Number of times pregnant.                                                                                   |
| ***Glucose***                   | Plasma glucose concentration after 2 hours in an oral glucose tolerance test.                               |
| ***BloodPressure***             | Diastolic blood pressure (mm Hg).                                                                            |
| ***SkinThickness***             | Triceps skinfold thickness (mm).                                                                             |
| ***Insulin***                   | 2-Hour serum insulin (mu U/ml).                                                                              |
| ***BMI***                       | Body mass index (weight in kg/(height in m)^2).                                                              |
| ***DiabetesPedigreeFunction***  | Diabetes pedigree function (a function that scores the likelihood of diabetes based on family history).      |
| ***Age***                       | Age in years.                                                                                               |
| ***Outcome***                   | Target variable, where 1 indicates diabetes and 0 indicates no diabetes.                                     |


### 🌲 ***MODEL SELECTION***
For this project, a ***Random Decision Forest*** was used to predict the outcome. In comparison to decision trees, the Random Forest is often more accurate as it combines the predictions from many decision trees, reducing errors. This also helps reduce overfitting. The Random Forest also handles complexity well and can better capture non-linear or complex relationships between variables. Additionally, it can give insights into which features are more important for predicting diabetes, helping us understand what factors are most influential in the model's prediction.

### 🔧 ***HYPERPARAMETER TUNING*** 
In order to get the best model, we can tune the hyperparameters and do cross-validation. The best model will consequently be selected. In this case, a Grid Search methodology will be implemented.

The following parameters will be tuned:
1. `n_estimators`: the number of trees in the forest.
2. `max_depth`: the maximum depth of each tree.
3. `min_samples_split`: the minimum number of samples that the model requires to split an internal node.
4. `class_weight`: the `balanced` option automatically adjusts the weights inversely proportional to the target class frequencies.

### 🏅 ***SCORING METRIC*** 
The ***F1*** metric was chosen to balance precision and recall. In diabetes detection, both misclassifying a healthy person (false positives) and missing the detection of diabetes in a diabetic person (false negatives) have serious consequences. This metric ensures that the model performs well in both aspects. ⚖️

# ⚙️ ***TECHNOLOGIES USED*** 

![Python](https://img.shields.io/badge/Python-FFD43B?logo=python) 
![Streamlit](https://img.shields.io/badge/Streamlit-0A1F44?logo=streamlit) 
![Pandas](https://img.shields.io/badge/Pandas-150458?logo=pandas) 
![NumPy](https://img.shields.io/badge/NumPy-013243?logo=numpy) 
![Plotly Express](https://img.shields.io/badge/Plotly_Express-3F4D75?logo=plotly) 
![Seaborn](https://img.shields.io/badge/Seaborn-3498DB?logo=seaborn) 
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-66B0FF?logo=scikit-learn)
![Lottie](https://img.shields.io/badge/Streamlit--Lottie-58C4B8?logo=lottie) 
![Requests](https://img.shields.io/badge/Requests-FF6F61?logo=requests)
![Matplotlib](https://img.shields.io/badge/Matplotlib-0077B5?logo=matplotlib)

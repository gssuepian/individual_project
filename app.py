import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import seaborn as sns
import requests
from streamlit_lottie import st_lottie
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay

# ======================================================================================= #
#                                                                                         #
#   Additional resoureces used for this application:                                      #
#   ------------------------------------------------                                      #
#                                                                                         #
#   For this project, in order to enhance the web application, I did some additional      #
#   research.                                                                             #
#                                                                                         #
#   - Git Hub Repo: https://github.com/Sven-Bo/personal-website-streamlit/tree/master     #
#       - Importing lottie files                                                          #
#       - set page configuration used to format tab subheader                             #
#       - Using local CSS to hide streamlit branding                                                                                 #
#                                                                                         #
# ======================================================================================= #

# ===========================================================
# DESIGN AND FORMATTING ELEMENTS
# ===========================================================

# Setting display of application tab
st.set_page_config(page_title="Diabetes Prediction Application", page_icon=":medical_symbol:")

# Use local CSS
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
local_css("style/style.css")

# Formatting buttons with CSS
st.markdown("""
    <style>
    .stButton > button {
        height: auto;
        padding-top: 10px;
        padding-bottom: 10px;
        font-size: 18px;
        font-weight: bold;
        color: white;
        background-color: #A67258;
        border: none;
        border-radius: 8px;
        width: 100%;
        cursor: pointer;
    }
    .stButton > button:hover {
        border: 2px solid #A67258;
        background-color: white;
        color: #3E2C1C;
    }
    </style>
    """, unsafe_allow_html=True)

# Loading Animatations (Lottie Files)

# Defining functionts to load lottie urls
def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:        
        return None
    return r.json()

# Obtaining lottie animatsions
lottie_medapp = load_lottieurl("https://lottie.host/73faebea-ebb0-4f9e-9ec7-9e2ce0089261/lq5XZ9uNLh.json")
lottie_hearbeat = load_lottieurl("https://lottie.host/fa3603c0-658e-4c54-bef3-f183a49c9718/5dkcgeqgjI.json")
lottie_celebrate = load_lottieurl("https://lottie.host/ad07c8be-baeb-456e-8008-73184e0c85f5/n2N3oDon07.json")
lottie_sad = load_lottieurl("https://lottie.host/bfade23d-4365-4d32-a12d-58182aab22f1/osbW8Nsqy1.json")
lottie_ml = load_lottieurl("https://lottie.host/853402aa-1281-4013-893b-c768e291647a/r67uHtE1EM.json")


# LOADING DATASET AND MODELS
# ---------------------------

# Loading Diabetes Dataset
diabetes = pd.read_csv('diabetes.csv')

# Load the pickled model and scaler
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('sc.pkl', 'rb') as f:
    scaler = pickle.load(f)


# CONFIGURING THE NAVIATION SIDE BAR
# ----------------------------------

# Initialize session state for page navigation
if "page" not in st.session_state:
    st.session_state.page = "Home"

# Sidebar buttons for navigation
st.sidebar.title("Diabetes Prediction Application")
st.sidebar.markdown('---')
st.sidebar.header(":mag: Navigation")
st.sidebar.write("Navigate through the different sections to learn more about the dataset, model, and predictions.")

# Making buttons for al the pages
if st.sidebar.button("Home"):
    st.session_state.page = "Home"
if st.sidebar.button("Get Your Predictions"):
    st.session_state.page = "Get Your Predictions"

st.sidebar.markdown('---')
st.sidebar.write("More technical details about the dataset and the model can be seen in the following pages:")

if st.sidebar.button("About the Dataset"):
    st.session_state.page = "About the Dataset"
if st.sidebar.button("About the Model"):
    st.session_state.page = "About the Model"

# Defining pages
page = st.session_state.page


# ===========================================================
# HOME PAGE
# ===========================================================

if page == "Home":

    # Header / title of page
    st.image('images/imagetitle.png')

    st.markdown('---')

    # About the Application Section
    col1, col2 = st.columns([6,4])
    with col1:
        st.header("About the App")
        st.write("This application is designed as part of a school project for the \
            Machine Learning II class at IE University in the Masters in Big Data and Business Analytics \
            degree.")
        st.write("It is designed to predict whether or not an individual may or may not have diabetes \
            based on 8 feature variables. More information and insights on this dataset can be found \
            in the 'About the Dataset' page of the application.")    
    with col2:
        st_lottie(lottie_medapp, height=275)
    st.markdown('---')

    st.markdown("""
    <div style="padding: 15px; background-color: #ffe6e6; border-radius: 5px;">
        <h3 style="color: #fc796f; margin-top: 0; text-align: center;">DISCLAIMER</h3>
        <p style="color: black; font-weight: normal; margin: 0;">
            This diabetes prediction model is provided for informational and educational purposes 
            and is not intended for medical use. It is not a substitute for professional medical advice, 
            diagnosis, or treatment. The predictions made by this model may be inaccurate or incomplete 
            and should not be used as a basis for making health decisions. Always consult a qualified 
            healthcare provider for any questions regarding a medical condition or health concerns. 
            This model is experimental and should not be relied upon for any medical or health-related decisions.
        </p>
    </div>
    """, 
    unsafe_allow_html=True
)

    st.markdown('---')

    st.image('images/diabetesribbon.jpg')

# ===========================================================
# GET YOUR PREDICTIONS PAGE
# ===========================================================

elif page == "Get Your Predictions":

    # Header / title of page
    st.image("images/imagepredictions.png")

    st.markdown('---')

    col1, col2 = st.columns([1,18])
    with col1:
        st.subheader(":information_source:")
    with col2:
        st.subheader("Enter Your Information Here")

    # Creating inuput fields for each feature
    col1, col2 = st.columns(2)
    with col1:
        pregnancies = st.number_input("Pregnancies", min_value=0, max_value=17, value=3)
        glucose = st.number_input("Glucose", min_value=0, max_value=199, value=120)
        blood_pressure = st.number_input("Blood Pressure", min_value=0, max_value=122, value=69)
        skin_thickness = st.number_input("Skin Thickness", min_value=0, max_value=99, value=20)
    with col2:
        insulin = st.number_input("Insulin", min_value=0, max_value=846, value=79)
        bmi = st.number_input("BMI", min_value=0.0, max_value=67.1, value=32.0, step=0.1)
        diabetes_pedigree_function = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=2.42, value=0.5, step=0.01)
        age = st.number_input("Age", min_value=21, max_value=81, value=33)

    st.markdown('---')

    col1, col2 = st.columns([1,18])
    with col1:
        st.subheader(":heavy_check_mark:")
    with col2:
        st.subheader("A Summary of Your Inputs")

    # Summarize user inputs
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
            - **Pregnancies:** {pregnancies}
            - **Glucose:** {glucose}
            - **Blood Pressure:** {blood_pressure}
            - **Skin Thickness:** {skin_thickness}""")
    with col2:
        st.markdown(f"""
            - **Insulin:** {insulin}
            - **BMI:** {bmi}
            - **Diabetes Pedigree Function:** {diabetes_pedigree_function}
            - **Age:** {age}""")

    # Prepare data for prediction based off of user inputs
    data = {
        "Pregnancies": pregnancies,
        "Glucose": glucose, 
        "BloodPressure": blood_pressure,
        "SkinThickness": skin_thickness,
        "Insulin": insulin,
        "BMI": bmi,
        "DiabetesPedigreeFunction": diabetes_pedigree_function,
        "Age": age
    }

    # Convert data into dataframe
    df = pd.DataFrame.from_dict([data])
    columns = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]
    df = df.reindex(columns=columns, fill_value=0)

    # Scale the input data
    df_scaled = scaler.transform(df)

    # Getting the prediction
    prediction = model.predict(df_scaled)

    st.markdown('---')

    st.image("images/getpredictions.png")

    # Predict button
    if st.button("Predict", key="predict_button"):
        # if the prediction is 0, the person does not have diabetes
        if prediction[0] == 0:
            st.balloons()
            st.image("images/congratulations.png")
            st_lottie(lottie_celebrate, height=150)
        # otherise (prediction = 1), the person has diabetes
        else:
            st.image("images/sorry.png")
            st_lottie(lottie_sad, height=150)

    st.markdown('---')

    st.image('images/diabetesribbon.jpg')

# ===========================================================
# ABOUT THE DATASET PAGE
# ===========================================================

elif page == "About the Dataset":

    # Header / title of page
    st.image('images/imagedataset.png')
    st.markdown('---')

    # Providing some background information
    st.subheader('Background Information')
    st.write("The data was collected using the Pima Indians Diabetes Dataset "
             "from the National Institute of Diabetes and Digestive Kidney Diseases (NIDDK).")
    st.write("The dataset contains 768 samples (rows), 8 feature variables, and 1 target variable.")
    st.markdown("The list of features can be seen below:")
    st.markdown("""
    - **Pregnancies**: Number of times pregnant.
    - **Glucose**: Plasma glucose concentration after 2 hours in an oral glucose tolerance test.
    - **BloodPressure**: Diastolic blood pressure (mm Hg).
    - **SkinThickness**: Triceps skinfold thickness (mm).
    - **Insulin**: 2-Hour serum insulin (mu U/ml).
    - **BMI**: Body mass index (weight in kg/(height in m)^2).
    - **DiabetesPedigreeFunction**: Diabetes pedigree function (a function that scores the likelihood of diabetes based on family history).
    - **Age**: Age in years.
    - **Outcome**: Target variable, where 1 indicates diabetes and 0 indicates no diabetes.
        """, unsafe_allow_html=True)
    st.markdown('---')

    # Section on Exploratory Data Analysis
    st.subheader("Exploratory Data Analysis (EDA)")
    st.write("This page provides an Exploratory Data Analysis (EDA) overview of the diabetes dataset, "
             "offering insights into data distribution and patterns.")
    st.markdown("""
        * ***1 = Diabetes***
        * ***0 = No Diabetes***
        """)
    st.write("You can obtain more details by hovering over components of the visualizations.")

    # Data Visualizations
    # -------------------

    # Defining the color map for the graphs based on outcome
    color_map = {0: "#969D90", 1: "#A67258"}

    # Obtaining outcome counts
    outcome_counts = diabetes['Outcome'].value_counts()

    # Displaying outcome counts as dataframe
    st.subheader('Outcome Counts')
    st.dataframe(outcome_counts, use_container_width=True)

    # Plotting pie chart
    fig_pie = px.pie(diabetes, names="Outcome", title="Outcome Percentage", color="Outcome", color_discrete_map=color_map)
    st.plotly_chart(fig_pie, use_container_width=True)

    # Interactive histogram of outcome and other feature variables
    features = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]

    with st.container():
        st.subheader("Diabetes Feature Distribution")

        col1, col2 = st.columns(2)
        with col1:
            # Column for feature selection
            selected_feature = st.selectbox("Select a feature to display", features)
        with col2:
            # Stacking option
            stack_option = st.selectbox("Stack histogram?", ("Yes", "No"))
            stacked = True if stack_option == "Yes" else False

    # Plot histogram based on inputs from select boxes
    fig_hist = px.histogram(diabetes, x=selected_feature, color="Outcome",
                            barmode="stack" if stacked else "group", color_discrete_map=color_map,
                            title=f"{selected_feature} Distribution by Diabetes Outcome")
    st.plotly_chart(fig_hist, use_container_width=True)

    # Plotting pair plot between two selected features
    with st.container():
        st.subheader("Pairplot between Variables and Outcome")

        col1, col2 = st.columns(2)
        with col1:
            feature1 = st.selectbox("Select the first feature to display", features)
        with col2:
            feature2 = st.selectbox("Select the second feature to display", [f for f in features if f != feature1])

    fig_pair = sns.pairplot(diabetes, hue='Outcome', vars=[feature1, feature2], palette=color_map)
    fig_pair.fig.suptitle(f"Pairplot of {feature1} and {feature2}", y=1.05)

    st.pyplot(fig_pair)

    st.markdown('---')

    st.image('images/diabetesribbon.jpg')

# ===========================================================
# ABOUT THE DATASET PAGE
# ===========================================================

elif page == "About the Model":

    # Title / header of page
    st.image('images/imagemodel.png')

    st.markdown('---')

    # Background Information
    st.subheader("Random Decision Forest")
    st.markdown("""
        For this project, the ***Random Decision Forest*** was utlizied. \
        It is important to note that in order to obtain the best model possible \
        hyperparameter tuning was conducted. the following parameters were tuned:
        """)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
            ```
            param_grid = {
            'n_estimators': [50, 100, 150],
            'max_depth': [None, 5, 10, 15],
            'min_samples_split': [2, 5, 10],
            'class_weight': [None, 'balanced']
            }
            """)
        st.markdown("""
            Additionally, the ***Grid Search CV*** methodology was applied with a cross-validation of\
            5 folds, and the best model was obtained using the ***F1*** evaluation metric.
            """)
    with col2:
        st_lottie(lottie_ml, height=275)
    st.markdown("""
        Nevertheless, it is important to note that because our dataset is very small (only 768 samples), \
        the model's performance is not ideal and does not generalize well to larger, more diverse datasets, \
        and results should be interpreted with caution.
        """)
    st.markdown("""
        A much larger dataset is needed for proper training and testing.
        """)

    st.markdown('---')

    st.subheader("Why F1 Score?")
    st.markdown("""
        The ***F1 score*** is ideal for diabetes detection because it balances precision and recall, \
        which is crucial in imbalanced datasets where the number of non-diabetic cases may vastly \
        outweigh diabetic cases. In diabetes detection, both false positives (misclassifying a healthy \
        person as diabetic) and false negatives (missing a diabetic person) have serious consequences. \
        The F1 score ensures that the model performs well in identifying both diabetics and non-diabetics, \
        making it a more reliable metric than accuracy for this business case.
        """)

    st.markdown('---')

    # Model Evaluation
    # ----------------

    st.subheader("Model Evaluation")

    st.markdown("""
        The evaluation metrics of the model can be seen below. It is important to note that because the \
        dataset utlizied is very samll, the model is unable to perform well. To obtain better results for \
        this model (or any machine learning model) would require a much larger dataset.
        """)

    # Confusion Matrix
    st.markdown("""
        <h4 style="color: #3e2c1c; margin-top: 0; text-align: center;">Confusion Matrix</h3>
        """,
        unsafe_allow_html=True  
        )
    col1, col2 = st.columns(2)
    with col1:
        st.image("images/confusionmatrix_train.png")
    with col2:
        st.image("images/confusionmatrix_test.png")

    # Classification Report
    st.markdown("""
        <h4 style="color: #3e2c1c; margin-top: 0; text-align: center;">Classification Report</h3>
        """,
        unsafe_allow_html=True  
        )
    st.image("images/classificationreport.png") 

    # Comparision of Metrics Between Train and Test Data
    st.markdown("""
        <h4 style="color: #3e2c1c; margin-top: 0; text-align: center;">
        Comparision of Metrics between Train and Test Set
        </h3>
        """,
        unsafe_allow_html=True  
        )      

    metrics = {
    "Dataset": ["Training", "Test"],
    "Accuracy": [0.85, 0.75],
    "Precision": [0.74, 0.62],
    "Recall": [0.82, 0.76],
    "F1 Score": [0.80, 0.68]
    }

    metrics_df = pd.DataFrame(metrics)
    metrics_df.set_index('Dataset', inplace=True)
    st.dataframe(metrics_df, use_container_width=True)
 
    st.markdown('---')

    st.image('images/diabetesribbon.jpg')


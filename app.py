import streamlit as st
import pandas as pd
import joblib as jb

model = jb.load("./fraud_detector.joblib")

st.title(
    body = "Fraud detector"
)

st.markdown(
    """<h2>Processed Dataset Overview</h2>

<p>
    
The dataset used in this project originates from the Credit Card Transactions Fraud Detection Dataset (Kartik Shenoy, Kaggle).
After extensive data cleaning, restructuring, and preprocessing, it was prepared for training and evaluating a machine learning model designed to detect fraudulent transactions.

During preprocessing, several transformations were applied to ensure data consistency and model readiness:
</p>

<ul>
    <li>
        <strong>Data cleaning</strong>: removal of duplicates, missing values, and inconsistent records.
    </li>
    <li>
        <strong>Feature extraction</strong>: generation of new variables from trans_date_trans_time (e.g., hour, day of week, month) and geographic or demographic information.
    </li>
    <li>
        <strong>Encoding</strong>: categorical features such as category and gender were converted into numerical form using one-hot encoding.
    </li>
    <li>
        <strong>Scaling</strong>: continuous variables (like amt, city_pop, lat, long) were normalized or standardized for better model convergence.
    </li>
    <li>
        <strong>Label definition</strong>: the column is_fraud was used as the binary target variable (1 = fraud, 0 = non-fraud).
    </li>
    <li>
        <strong>Balancing</strong>: undersampling or oversampling techniques were considered to address the natural class imbalance in fraud detection tasks.
    </li>
</ul>

<p>
    The resulting dataset is now clean, structured, and ready for predictive modeling.
    It serves as the foundation for the machine learning pipeline implemented in this application, enabling real-time fraud prediction and data-driven insights into transaction behavior.
</p>

<h2>Test the model</h2>
""",
    unsafe_allow_html=True
)

amount = st.number_input(
    label = "Amount (R$)",
    min_value = 0.00,
    step = 0.01
)

col1, col2 = st.columns(
    spec = 2
)

date = col1.date_input(
    label = "Date",
    value = "today"
)

hour = col2.time_input(
    label = "Hour",
    value = "now"
)

gender = col1.selectbox(
    label = "Gender",
    options = ["Female", "Male"],
)

category = col2.selectbox(
    label = "Category",
    options = [
        'entertainment',
        'food_dining',
        'gas_transport',
        'grocery_net',
        'grocery_pos',
        'health_fitness',
        'home',
        'kids_pets',
        'misc_net',
        'misc_pos',
        'personal_care',
        'shopping_net',
        'shopping_pos',
        'travel'
    ]
)

age = col1.number_input(
    label = "Age",
    min_value = 0,
    max_value = 100,
    step = 0
)

zip_ = col2.number_input(
    label = "Zip",
    step = 0
)

button = st.button(
    label = "Check",
    type = "primary"
)

if button:

    dataset = pd.DataFrame(
        data = {
            "amount": float(amount),
            "month": int(str(date).split("-")[1]),
            "hour": int(str(hour).split(":")[0]),
            "gender": gender.replace("Female", "F").replace("Male", "M"),
            "category": category,
            "age": int(age),
            "zip": int(zip_)
        },
        index = [0]
    )

    prediction = model.predict(
        dataset
    )

    if prediction[0] == 0:
        answer = "legit"
    else: answer = "fraud"

    print(dataset)

    st.write(
        f"Transaction status: {answer}"
    )
    # print(prediction)


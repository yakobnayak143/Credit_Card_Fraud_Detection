import streamlit as st
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import plotly.express as px


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide"
)


# --------------------------------------------------
# TITLE
# --------------------------------------------------

st.title("💳 Credit Card Fraud Detection System")

st.write(
    "Machine Learning based system for detecting potentially "
    "fraudulent credit card transactions."
)

st.divider()


# --------------------------------------------------
# LOAD DATASET
# --------------------------------------------------

@st.cache_data
def load_data():
    return pd.read_csv("creditcard_data_small.csv")


df = load_data()


# --------------------------------------------------
# DATASET OVERVIEW
# --------------------------------------------------

total_transactions = len(df)

fraud_transactions = int(df["Class"].sum())

legitimate_transactions = (
    total_transactions - fraud_transactions
)

fraud_percentage = (
    fraud_transactions / total_transactions
) * 100


st.subheader("📊 Dataset Overview")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Total Transactions",
        f"{total_transactions:,}"
    )

with col2:
    st.metric(
        "Legitimate Transactions",
        f"{legitimate_transactions:,}"
    )

with col3:
    st.metric(
        "Fraud Transactions",
        f"{fraud_transactions:,}"
    )

with col4:
    st.metric(
        "Fraud Percentage",
        f"{fraud_percentage:.3f}%"
    )


st.divider()


# --------------------------------------------------
# PREPARE DATA
# --------------------------------------------------

X = df.drop("Class", axis=1)

y = df["Class"]


scaler = StandardScaler()

X["Amount"] = scaler.fit_transform(
    X[["Amount"]]
)


# --------------------------------------------------
# TRAIN MODEL
# --------------------------------------------------

@st.cache_resource
def train_model(X, y):

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    model = LogisticRegression(
        max_iter=3000
    )

    model.fit(
        X_train,
        y_train
    )

    predictions = model.predict(
        X_test
    )

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    return model, accuracy


model, accuracy = train_model(X, y)


# --------------------------------------------------
# MODEL PERFORMANCE
# --------------------------------------------------

st.subheader("🤖 Machine Learning Model")

model_col1, model_col2 = st.columns(2)

with model_col1:
    st.metric(
        "Model",
        "Logistic Regression"
    )

with model_col2:
    st.metric(
        "Accuracy",
        f"{accuracy * 100:.2f}%"
    )


st.divider()


# --------------------------------------------------
# TRANSACTION PREDICTION
# --------------------------------------------------

st.subheader("💳 Check a Transaction")

st.write(
    "Enter the transaction details below."
)

input_col1, input_col2 = st.columns(2)


# --------------------------------------------------
# TIME INPUT
# --------------------------------------------------

with input_col1:

    time_input = st.text_input(
        "Transaction Time",
        value="100",
        placeholder="Enter time value"
    )


# --------------------------------------------------
# AMOUNT INPUT
# --------------------------------------------------

with input_col2:

    amount_input = st.text_input(
        "Transaction Amount",
        value="100",
        placeholder="Enter amount"
    )


st.write("")


check_button = st.button(
    "🔍 Check Transaction",
    use_container_width=True
)


# --------------------------------------------------
# RESULT
# --------------------------------------------------

if check_button:

    try:

        time = float(time_input)
        amount = float(amount_input)

        if time < 0:
            st.error("❌ Transaction Time cannot be negative.")

        elif amount < 0:
            st.error("❌ Transaction Amount cannot be negative.")

        else:

            # Create input using average values
            # for V1-V28
            input_data = X.drop(
                columns=["Time", "Amount"]
            ).mean().to_frame().T


            # Add Time
            input_data.insert(
                0,
                "Time",
                time
            )


            # Scale Amount
            amount_df = pd.DataFrame(
                {"Amount": [amount]}
            )

            scaled_amount = scaler.transform(
                amount_df
            )[0][0]


            input_data["Amount"] = scaled_amount


            # Arrange columns correctly
            input_data = input_data[
                X.columns
            ]


            # Prediction
            prediction = model.predict(
                input_data
            )[0]


            probability = model.predict_proba(
                input_data
            )[0][1]


            st.divider()

            st.subheader("🔎 Prediction Result")


            if prediction == 1:

                st.error(
                    "🚨 FRAUDULENT TRANSACTION DETECTED"
                )

                st.warning(
                    "This transaction should be reviewed carefully."
                )

            else:

                st.success(
                    "✅ LEGITIMATE TRANSACTION"
                )

                st.info(
                    "The model does not identify this transaction as fraudulent."
                )


            st.metric(
                "Fraud Probability",
                f"{probability * 100:.2f}%"
            )


    except ValueError:

        st.error(
            "❌ Please enter valid numbers for Time and Amount."
        )


# --------------------------------------------------
# FRAUD DISTRIBUTION
# --------------------------------------------------

st.divider()

st.subheader("📈 Transaction Distribution")


chart_data = pd.DataFrame(
    {
        "Transaction Type": [
            "Legitimate",
            "Fraud"
        ],
        "Count": [
            legitimate_transactions,
            fraud_transactions
        ]
    }
)


fig = px.pie(
    chart_data,
    names="Transaction Type",
    values="Count",
    hole=0.5
)


fig.update_traces(
    textinfo="label+percent",
    textposition="inside"
)


fig.update_layout(
    height=350,
    margin=dict(
        t=30,
        b=20,
        l=20,
        r=20
    )
)


st.plotly_chart(
    fig,
    use_container_width=True
)


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "Credit Card Fraud Detection | Machine Learning Project"
)
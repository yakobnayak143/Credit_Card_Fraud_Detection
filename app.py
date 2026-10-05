import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)


# ==========================================================
# PAGE CONFIGURATION
# ==========================================================

st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide"
)


# ==========================================================
# CUSTOM CSS
# ==========================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 42px;
        font-weight: 700;
        text-align: center;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        font-size: 18px;
        color: #666;
        margin-bottom: 30px;
    }

    .metric-card {
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #ddd;
        text-align: center;
    }

    .risk-high {
        padding: 20px;
        border-radius: 12px;
        background-color: #ffe5e5;
        border: 1px solid #ff4b4b;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
    }

    .risk-low {
        padding: 20px;
        border-radius: 12px;
        background-color: #e5ffe9;
        border: 1px solid #28a745;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ==========================================================
# TITLE
# ==========================================================

st.markdown(
    '<div class="main-title">💳 Credit Card Fraud Detection System</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Machine Learning Based Fraud Detection Dashboard</div>',
    unsafe_allow_html=True
)


# ==========================================================
# LOAD DATASET
# ==========================================================

@st.cache_data
def load_data():

    return pd.read_csv(
        "creditcard_data_small.csv"
    )


df = load_data()


# ==========================================================
# DATASET OVERVIEW
# ==========================================================

st.header("📊 Dataset Overview")

total_transactions = len(df)

legitimate_transactions = int(
    (df["Class"] == 0).sum()
)

fraud_transactions = int(
    (df["Class"] == 1).sum()
)

fraud_percentage = (
    fraud_transactions / total_transactions
) * 100


col1, col2, col3, col4 = st.columns(4)


with col1:
    st.metric(
        "Total Transactions",
        f"{total_transactions:,}"
    )


with col2:
    st.metric(
        "Legitimate",
        f"{legitimate_transactions:,}"
    )


with col3:
    st.metric(
        "Fraudulent",
        f"{fraud_transactions:,}"
    )


with col4:
    st.metric(
        "Fraud Rate",
        f"{fraud_percentage:.2f}%"
    )


# ==========================================================
# PREPARE DATA
# ==========================================================

X = df.drop(
    "Class",
    axis=1
)

y = df["Class"]


# Scale Amount

scaler = StandardScaler()

X["Amount"] = scaler.fit_transform(
    X[["Amount"]]
)


# ==========================================================
# TRAIN TEST SPLIT
# ==========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ==========================================================
# TRAIN MODEL
# ==========================================================

model = LogisticRegression(
    max_iter=3000
)

model.fit(
    X_train,
    y_train
)


# ==========================================================
# MODEL PREDICTIONS
# ==========================================================

y_pred = model.predict(
    X_test
)

y_probability = model.predict_proba(
    X_test
)[:, 1]


# ==========================================================
# MODEL METRICS
# ==========================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0
)

roc_auc = roc_auc_score(
    y_test,
    y_probability
)


# ==========================================================
# MODEL PERFORMANCE
# ==========================================================

st.header("🤖 Model Performance")

m1, m2, m3, m4, m5 = st.columns(5)


with m1:
    st.metric(
        "Accuracy",
        f"{accuracy * 100:.2f}%"
    )


with m2:
    st.metric(
        "Precision",
        f"{precision * 100:.2f}%"
    )


with m3:
    st.metric(
        "Recall",
        f"{recall * 100:.2f}%"
    )


with m4:
    st.metric(
        "F1 Score",
        f"{f1 * 100:.2f}%"
    )


with m5:
    st.metric(
        "ROC-AUC",
        f"{roc_auc:.4f}"
    )


st.info(
    "The model uses Logistic Regression with standardized transaction amount values."
)


# ==========================================================
# CHARTS
# ==========================================================

st.header("📈 Transaction Analysis")


chart_col1, chart_col2 = st.columns(2)


# ----------------------------------------------------------
# Fraud Distribution
# ----------------------------------------------------------

with chart_col1:

    chart_data = pd.DataFrame(
        {
            "Transaction Type": [
                "Legitimate",
                "Fraudulent"
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
        hole=0.5,
        title="Fraud vs Legitimate Transactions"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# ----------------------------------------------------------
# Transaction Amount
# ----------------------------------------------------------

with chart_col2:

    amount_fig = px.histogram(
        df,
        x="Amount",
        color="Class",
        nbins=50,
        title="Transaction Amount Distribution",
        labels={
            "Class": "Transaction Type"
        }
    )

    st.plotly_chart(
        amount_fig,
        use_container_width=True
    )


# ==========================================================
# CONFUSION MATRIX
# ==========================================================

st.header("🔲 Confusion Matrix")

cm = confusion_matrix(
    y_test,
    y_pred
)


cm_fig = go.Figure(
    data=go.Heatmap(
        z=cm,
        x=[
            "Predicted Legitimate",
            "Predicted Fraud"
        ],
        y=[
            "Actual Legitimate",
            "Actual Fraud"
        ],
        text=cm,
        texttemplate="%{text}",
        colorscale="Blues"
    )
)


cm_fig.update_layout(
    title="Model Confusion Matrix"
)


st.plotly_chart(
    cm_fig,
    use_container_width=True
)


# ==========================================================
# TRANSACTION PREDICTION
# ==========================================================

st.header("🔍 Check a Transaction")

st.write(
    "Enter the transaction time and amount to estimate the fraud probability."
)


time_input = st.text_input(
    "Transaction Time",
    value="0",
    help="Enter transaction time as a number."
)


amount_input = st.text_input(
    "Transaction Amount (₹)",
    value="100",
    help="Enter the transaction amount."
)


check_button = st.button(
    "🔍 Check Transaction",
    type="primary"
)


if check_button:

    try:

        transaction_time = float(
            time_input
        )

        transaction_amount = float(
            amount_input
        )


        if transaction_time < 0:

            st.error(
                "Transaction time cannot be negative."
            )

        elif transaction_amount < 0:

            st.error(
                "Transaction amount cannot be negative."
            )

        else:

            # Create transaction using
            # average values for V1-V28

            transaction = pd.DataFrame(
                [[
                    transaction_time
                ] + [
                    df[column].mean()
                    for column in df.columns
                    if column.startswith("V")
                ] + [
                    transaction_amount
                ]],
                columns=[
                    "Time"
                ] + [
                    column
                    for column in df.columns
                    if column.startswith("V")
                ] + [
                    "Amount"
                ]
            )


            # Scale Amount

            transaction["Amount"] = scaler.transform(
                transaction[["Amount"]]
            )


            # Prediction

            prediction = model.predict(
                transaction
            )[0]

            probability = model.predict_proba(
                transaction
            )[0][1]


            fraud_probability = probability * 100


            # ==================================================
            # RESULT
            # ==================================================

            st.subheader(
                "Prediction Result"
            )


            if prediction == 1:

                st.error(
                    f"🚨 FRAUDULENT TRANSACTION"
                )

                st.markdown(
                    f"### Fraud Probability: {fraud_probability:.2f}%"
                )


                st.markdown(
                    '<div class="risk-high">⚠️ HIGH RISK TRANSACTION</div>',
                    unsafe_allow_html=True
                )

                st.warning(
                    "This transaction should be reviewed before approval."
                )


            else:

                st.success(
                    "✅ LEGITIMATE TRANSACTION"
                )

                st.markdown(
                    f"### Fraud Probability: {fraud_probability:.2f}%"
                )


                st.markdown(
                    '<div class="risk-low">🟢 LOW RISK TRANSACTION</div>',
                    unsafe_allow_html=True
                )

                st.info(
                    "The model considers this transaction likely legitimate."
                )


            # Probability bar

            st.progress(
                min(
                    max(
                        int(fraud_probability),
                        0
                    ),
                    100
                )
            )


    except ValueError:

        st.error(
            "Please enter valid numeric values."
        )


# ==========================================================
# CSV UPLOAD
# ==========================================================

st.header("📁 Batch Transaction Prediction")

st.write(
    "Upload a CSV file containing transaction data to predict multiple transactions."
)


uploaded_file = st.file_uploader(
    "Upload CSV",
    type=["csv"]
)


if uploaded_file is not None:

    try:

        uploaded_df = pd.read_csv(
            uploaded_file
        )


        st.write(
            "Uploaded Data:"
        )

        st.dataframe(
            uploaded_df.head()
        )


        required_columns = [
            "Time"
        ] + [
            column
            for column in df.columns
            if column.startswith("V")
        ] + [
            "Amount"
        ]


        missing_columns = [
            column
            for column in required_columns
            if column not in uploaded_df.columns
        ]


        if missing_columns:

            st.error(
                "Missing required columns: "
                + ", ".join(missing_columns)
            )

        else:

            prediction_data = uploaded_df[
                required_columns
            ].copy()


            prediction_data["Amount"] = scaler.transform(
                prediction_data[["Amount"]]
            )


            predictions = model.predict(
                prediction_data
            )

            probabilities = model.predict_proba(
                prediction_data
            )[:, 1]


            result_df = uploaded_df.copy()


            result_df["Prediction"] = np.where(
                predictions == 1,
                "Fraudulent",
                "Legitimate"
            )


            result_df["Fraud Probability (%)"] = (
                probabilities * 100
            ).round(2)


            st.success(
                "Batch prediction completed successfully!"
            )


            st.dataframe(
                result_df,
                use_container_width=True
            )


            fraud_count = int(
                (predictions == 1).sum()
            )

            legitimate_count = int(
                (predictions == 0).sum()
            )


            b1, b2 = st.columns(2)


            with b1:

                st.metric(
                    "Fraudulent Transactions",
                    fraud_count
                )


            with b2:

                st.metric(
                    "Legitimate Transactions",
                    legitimate_count
                )


            csv_data = result_df.to_csv(
                index=False
            ).encode(
                "utf-8"
            )


            st.download_button(
                label="📥 Download Prediction Results",
                data=csv_data,
                file_name="fraud_predictions.csv",
                mime="text/csv"
            )


    except Exception as e:

        st.error(
            f"Unable to process the uploaded file: {e}"
        )


# ==========================================================
# ABOUT PROJECT
# ==========================================================

st.header("ℹ️ About This Project")

st.write(
    """
    This project uses Machine Learning to identify potentially
    fraudulent credit card transactions.

    The system analyzes transaction data and predicts whether
    a transaction is likely to be legitimate or fraudulent.

    **Machine Learning Algorithm:** Logistic Regression

    **Dataset:** Credit Card Transactions

    **Main Technologies:** Python, Pandas, NumPy,
    Scikit-learn, Streamlit and Plotly
    """
)


# ==========================================================
# FOOTER
# ==========================================================

st.markdown("---")

st.markdown(
    """
    <div style="text-align:center;">
        <p>💳 Credit Card Fraud Detection System</p>
        <p>Built with Python, Machine Learning and Streamlit</p>
    </div>
    """,
    unsafe_allow_html=True
)
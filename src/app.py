import streamlit as st
import pandas as pd
import joblib
from pathlib import Path


# Page configuration


st.set_page_config(
    page_title="Loan Approval Prediction System",
    page_icon="🏦",
    layout="wide"
)


# Project paths


BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "Loan_Approval_Cleaned.csv"

DT_MODEL_PATH = (
    BASE_DIR / "models" / "decision_tree_model.pkl"
)

LR_MODEL_PATH = (
    BASE_DIR / "models" / "logistic_regression_model.pkl"
)


# Load models


@st.cache_resource
def load_models():
    decision_tree = joblib.load(DT_MODEL_PATH)
    logistic_regression = joblib.load(LR_MODEL_PATH)

    return decision_tree, logistic_regression


decision_tree_model, logistic_regression_model = load_models()


# Load dataset


@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


df = load_data()


# Sidebar navigation


st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Select Page",
    [
        "Home",
        "Loan Prediction",
        "Data Overview",
        "Model Performance"
    ]
)


# Home page


if page == "Home":

    st.title("🏦 Loan Approval Prediction System")

    st.write(
        """
        This application uses machine-learning models to predict
        whether a loan application is likely to be approved or
        rejected.
        """
    )

    st.subheader("Available Models")

    st.write(
        """
        **Decision Tree Classifier**

        A tree-based supervised machine-learning model that makes
        predictions using a sequence of decision rules.

        **Logistic Regression**

        A supervised binary classification model that estimates the
        probability of loan approval.
        """
    )

    st.info(
        "Use the navigation menu to access the prediction system."
    )

# Data overview page

elif page == "Data Overview":

    st.title("📊 Dataset Overview")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Applications",
            f"{len(df):,}"
        )

    with col2:
        st.metric(
            "Features",
            len(df.columns)
        )

    with col3:
        approval_rate = (
            df["LoanApproved"].mean() * 100
        )

        st.metric(
            "Approval Rate",
            f"{approval_rate:.2f}%"
        )

    st.subheader("Dataset Preview")

    st.dataframe(
        df.head(20),
        use_container_width=True
    )

    st.subheader("Loan Approval Distribution")

    approval_counts = (
        df["LoanApproved"]
        .value_counts()
        .rename(
            index={
                0: "Rejected",
                1: "Approved"
            }
        )
    )

    st.bar_chart(approval_counts)

# Model performance page 

elif page == "Model Performance":

    st.title("📈 Model Performance")

    performance_df = pd.DataFrame({
        "Model": [
            "Decision Tree",
            "Logistic Regression"
        ],
        "Accuracy": [
            0.9858,
            0.9955
        ],
        "Precision": [
            0.9687,
            0.9916
        ],
        "Recall": [
            0.9718,
            0.9895
        ],
        "F1 Score": [
            0.9702,
            0.9906
        ]
    })

    st.dataframe(
        performance_df,
        use_container_width=True,
        hide_index=True
    )

    st.subheader("Model Comparison")

    chart_df = performance_df.set_index(
        "Model"
    )

    st.bar_chart(chart_df)

    st.success(
        """
        Logistic Regression achieved the strongest overall
        performance on the held-out test dataset.
        """
    )

 # Leave loan page blank

elif page == "Loan Prediction":

    st.title("🔍 Loan Approval Prediction")

    st.info(
        "Applicant input form will be added next."
    )
    
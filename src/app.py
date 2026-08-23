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


# if page == "Home":

#     st.title("🏦 Loan Approval Prediction System")

#     st.write(
#         """
#         This application uses machine-learning models to predict
#         whether a loan application is likely to be approved or
#         rejected.
#         """
#     )

#     st.subheader("Available Models")

#     st.write(
#         """
#         **Decision Tree Classifier**

#         A tree-based supervised machine-learning model that makes
#         predictions using a sequence of decision rules.

#         **Logistic Regression**

#         A supervised binary classification model that estimates the
#         probability of loan approval.
#         """
#     )

#     st.info(
#         "Use the navigation menu to access the prediction system."
#     )

if page == "Home":

    st.title("🏦 Loan Approval Prediction System")

    st.write(
        """
        This application uses supervised machine-learning models to
        estimate whether a loan application is likely to be approved
        or rejected based on applicant, financial, credit, and
        loan-related information.
        """
    )

    st.info(
        """
        The application is developed for academic and demonstration
        purposes. Predictions should not be interpreted as actual
        lending decisions or financial advice.
        """
    )

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🌳 Decision Tree")

        st.write(
            """
            The Decision Tree model makes predictions using a sequence
            of decision rules learned from the training dataset.
            """
        )

        st.metric(
            "Test Accuracy",
            "98.58%"
        )

        st.metric(
            "F1 Score",
            "97.02%"
        )

    with col2:
        st.subheader("📈 Logistic Regression")

        st.write(
            """
            Logistic Regression estimates the probability of loan
            approval using a weighted combination of applicant and
            financial characteristics.
            """
        )

        st.metric(
            "Test Accuracy",
            "99.55%"
        )

        st.metric(
            "F1 Score",
            "99.06%"
        )

    st.divider()

    st.subheader("Application Features")

    st.write(
        """
        - Select between two trained machine-learning models.
        - Enter applicant and loan-related information.
        - Automatically calculate monthly income, monthly loan payment,
          and debt-to-income ratio.
        - Generate an Approved or Rejected prediction.
        - Display the model's estimated approval probability.
        - Review the dataset and model-performance information.
        """
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

    st.subheader("Performance Interpretation")

    st.write(
        """
        Both models achieved strong performance on the held-out test
        dataset. Logistic Regression produced the highest overall
        Accuracy, Precision, Recall, and F1-score.

        The Decision Tree remains useful because it provides a
        different modelling approach and captures nonlinear decision
        relationships. The application therefore allows users to
        select either model for prediction.
        """
    )

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

    st.write(
        """
        Enter the applicant and loan information below, then select
        a machine-learning model to generate a loan approval prediction.
        """
    )

    model_choice = st.selectbox(
        "Select Prediction Model",
        [
            "Decision Tree",
            "Logistic Regression"
        ]
    )

    with st.expander(
        "👤 Applicant Information",
        expanded=True
    ):

        col1, col2, col3 = st.columns(3)

        with col1:
            age = st.number_input(
                "Age",
                min_value=18,
                max_value=100,
                value=30
            )

            annual_income = st.number_input(
                "Annual Income",
                min_value=0.0,
                value=50000.0,
                step=1000.0
            )

            employment_status = st.selectbox(
                "Employment Status",
                sorted(
                    df["EmploymentStatus"]
                    .dropna()
                    .unique()
                    .tolist()
                )
            )

            education_level = st.selectbox(
                "Education Level",
                sorted(
                    df["EducationLevel"]
                    .dropna()
                    .unique()
                    .tolist()
                )
            )

            marital_status = st.selectbox(
                "Marital Status",
                sorted(
                    df["MaritalStatus"]
                    .dropna()
                    .unique()
                    .tolist()
                )
            )

            home_ownership = st.selectbox(
                "Home Ownership Status",
                sorted(
                    df["HomeOwnershipStatus"]
                    .dropna()
                    .unique()
                    .tolist()
                )
            )

        with col2:
            experience = st.number_input(
                "Experience",
                min_value=0.0,
                max_value=float(age),
                value=min(5.0, float(age))
            )

            job_tenure = st.number_input(
                "Job Tenure",
                min_value=0.0,
                value=3.0
            )

            number_dependents = st.number_input(
                "Number of Dependents",
                min_value=0,
                value=0,
                step=1
            )

            credit_score = st.number_input(
                "Credit Score",
                min_value=0.0,
                value=650.0
            )

            risk_score = st.number_input(
                "Risk Score",
                min_value=0.0,
                value=50.0
            )

            previous_defaults = st.number_input(
                "Previous Loan Defaults",
                min_value=0,
                value=0,
                step=1
            )

        with col3:
            bankruptcy_history = st.number_input(
                "Bankruptcy History",
                min_value=0,
                value=0,
                step=1
            )

            credit_history = st.number_input(
                "Length of Credit History",
                min_value=0.0,
                value=5.0
            )

            open_credit_lines = st.number_input(
                "Number of Open Credit Lines",
                min_value=0,
                value=2,
                step=1
            )

            credit_inquiries = st.number_input(
                "Number of Credit Inquiries",
                min_value=0,
                value=1,
                step=1
            )

            credit_utilisation = st.number_input(
                "Credit Card Utilization Rate",
                min_value=0.0,
                value=0.30
            )

            payment_history = st.number_input(
                "Payment History",
                min_value=0.0,
                value=90.0
            )

    with st.expander(
        "💰 Financial and Loan Information",
        expanded=True
    ):

        col4, col5, col6 = st.columns(3)

        with col4:
            loan_amount = st.number_input(
                "Loan Amount",
                min_value=0.0,
                value=20000.0,
                step=1000.0
            )

            loan_duration = st.number_input(
                "Loan Duration (Months)",
                min_value=1,
                value=48,
                step=1
            )

            interest_rate = st.number_input(
                "Interest Rate (%)",
                min_value=0.0,
                value=20.0,
                step=0.1
            )

            loan_purpose = st.selectbox(
                "Loan Purpose",
                sorted(
                    df["LoanPurpose"]
                    .dropna()
                    .unique()
                    .tolist()
                )
            )

        with col5:
            monthly_debt_payments = st.number_input(
                "Monthly Debt Payments",
                min_value=0.0,
                value=300.0
            )

            checking_balance = st.number_input(
                "Checking Account Balance",
                value=1000.0
            )

            savings_balance = st.number_input(
                "Savings Account Balance",
                value=5000.0
            )

            total_assets = st.number_input(
                "Total Assets",
                min_value=0.0,
                value=50000.0
            )

        with col6:
            total_liabilities = st.number_input(
                "Total Liabilities",
                min_value=0.0,
                value=10000.0
            )

            net_worth = st.number_input(
                "Net Worth",
                value=40000.0
            )

            monthly_income = annual_income / 12

        monthly_interest_rate = (
            interest_rate / 100 / 12
        )

        if monthly_interest_rate > 0:
            monthly_loan_payment = (
                loan_amount
                * (
                    monthly_interest_rate
                    * (1 + monthly_interest_rate)
                    ** loan_duration
                )
                / (
                    (1 + monthly_interest_rate)
                    ** loan_duration
                    - 1
                )
            )
        else:
            monthly_loan_payment = (
                loan_amount / loan_duration
            )

        if monthly_income > 0:
            total_dti = (
                (
                    monthly_loan_payment
                    + monthly_debt_payments
                )
                * 100
                / monthly_income
            )
        else:
            total_dti = 0


    st.subheader("Calculated Financial Indicators")

    m1, m2, m3 = st.columns(3)

    with m1:
        st.metric(
            "Monthly Income",
            f"{monthly_income:,.2f}"
        )

    with m2:
        st.metric(
            "Monthly Loan Payment",
            f"{monthly_loan_payment:,.2f}"
        )

    with m3:
        st.metric(
            "Debt-to-Income Ratio",
            f"{total_dti:.2f}%"
        )

        input_data = pd.DataFrame({
        "AnnualIncome": [annual_income],
        "EmploymentStatus": [employment_status],
        "LoanDuration": [loan_duration],
        "NumberOfDependents": [number_dependents],
        "MonthlyDebtPayments": [monthly_debt_payments],
        "NumberOfOpenCreditLines": [open_credit_lines],
        "BankruptcyHistory": [bankruptcy_history],
        "PreviousLoanDefaults": [previous_defaults],
        "LengthOfCreditHistory": [credit_history],
        "CheckingAccountBalance": [checking_balance],
        "TotalLiabilities": [total_liabilities],
        "NetWorth": [net_worth],
        "Age": [age],
        "CreditScore": [credit_score],
        "EducationLevel": [education_level],
        "LoanAmount": [loan_amount],
        "MaritalStatus": [marital_status],
        "HomeOwnershipStatus": [home_ownership],
        "CreditCardUtilizationRate": [credit_utilisation],
        "NumberOfCreditInquiries": [credit_inquiries],
        "LoanPurpose": [loan_purpose],
        "PaymentHistory": [payment_history],
        "SavingsAccountBalance": [savings_balance],
        "TotalAssets": [total_assets],
        "JobTenure": [job_tenure],
        "InterestRate": [interest_rate],
        "RiskScore": [risk_score],
        "Experience": [experience],
        "MonthlyLoanPayment": [monthly_loan_payment],
        "TotalDebtToIncomeRatio": [total_dti]
    })

    # if st.button(
    #     "Predict Loan Approval",
    #     type="primary"
    # ):

    #     if model_choice == "Decision Tree":
    #         selected_model = decision_tree_model
    #     else:
    #         selected_model = logistic_regression_model

    #     prediction = selected_model.predict(
    #         input_data
    #     )[0]

    #     if hasattr(
    #         selected_model,
    #         "predict_proba"
    #     ):
    #         probability = selected_model.predict_proba(
    #             input_data
    #         )[0][1]
    #     else:
    #         probability = None

    #     st.divider()

    #     if prediction == 1:
    #         st.success(
    #             "✅ Prediction: Loan Approved"
    #         )
    #     else:
    #         st.error(
    #             "❌ Prediction: Loan Rejected"
    #         )

    #     st.write(
    #         f"**Model Used:** {model_choice}"
    #     )

    #     if probability is not None:
    #         st.write(
    #             f"**Predicted Approval Probability:** "
    #             f"{probability * 100:.2f}%"
    #         )
        # Input validation
    validation_errors = []

    if experience > age:
        validation_errors.append(
            "Experience cannot be greater than age."
        )

    if annual_income <= 0:
        validation_errors.append(
            "Annual income must be greater than zero."
        )

    if loan_amount <= 0:
        validation_errors.append(
            "Loan amount must be greater than zero."
        )

    if loan_duration <= 0:
        validation_errors.append(
            "Loan duration must be greater than zero."
        )

    # Prediction button
    if st.button(
        "Predict Loan Approval",
        type="primary"
    ):

        # If there are invalid inputs, show the errors
        if validation_errors:

            for error in validation_errors:
                st.error(error)

        # Otherwise run the selected model
        else:

            if model_choice == "Decision Tree":
                selected_model = decision_tree_model
            else:
                selected_model = logistic_regression_model

            prediction = selected_model.predict(
                input_data
            )[0]

            probability = selected_model.predict_proba(
                input_data
            )[0][1]

            st.divider()

            st.subheader("Prediction Result")

            if prediction == 1:
                st.success(
                    "✅ Prediction: Loan Approved"
                )
            else:
                st.error(
                    "❌ Prediction: Loan Rejected"
                )

            result_col1, result_col2 = st.columns(2)

            with result_col1:
                st.metric(
                    "Model Used",
                    model_choice
                )

            with result_col2:
                st.metric(
                    "Estimated Approval Probability",
                    f"{probability * 100:.2f}%"
                )

            st.progress(
                float(probability)
            )

            st.caption(
                """
                The probability represents the model's estimated
                confidence for the entered applicant profile. It does
                not guarantee an actual lending outcome.
                """
            )

            # st.divider()


            # st.subheader("Prediction Result")

            # if prediction == 1:
            #     st.success(
            #         "✅ Prediction: Loan Approved"
            #     )
            # else:
            #     st.error(
            #         "❌ Prediction: Loan Rejected"
            #     )

            # st.write(
            #     f"**Model Used:** {model_choice}"
            # )

            # st.write(
            #     f"**Estimated Approval Probability:** "
            #     f"{probability * 100:.2f}%"
            # )

            # st.progress(
            #     float(probability)
            # )
st.divider()

st.caption(
    """
    CMP7005 – Programming for Data Analysis |
    Loan Approval Prediction System |
    Academic Project
    """
)
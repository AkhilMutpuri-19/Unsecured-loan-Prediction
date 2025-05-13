import streamlit as st
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load the pre-trained model and scaler
model = joblib.load("loan_model.pkl")
scaler = joblib.load("scaler.pkl")

# Define feature names for better readability
feature_names = [
    'age', 'dependents', 'employment_status', 'occupation_type', 'income', 'expenses',
    'credit_score', 'existing_loans', 'existing_loan_amount', 'outstanding_debt',
    'loan_history', 'loan_amount', 'loan_term', 'loan_purpose', 'interest_rate',
    'co_applicant', 'default_risk'
]

# Sidebar for dynamic input
st.title("Unsecured Loan Approval Prediction App")
st.sidebar.header("Enter Applicant Details")

# Sidebar inputs for dynamic feature values
age = st.sidebar.slider("Age", 18, 70, 30)
dependents = st.sidebar.slider("Number of Dependents", 0, 5, 1)
employment_status = st.sidebar.selectbox("Employment Status (0=Unemployed, 1=Employed, 2=Self-Employed)", [0, 1, 2])
occupation_type = st.sidebar.selectbox("Occupation Type (1=Business, 2=Salaried, 3=Freelancer, 4=Professional)", [1, 2, 3, 4])
income = st.sidebar.number_input("Annual Income", 10000, 5000000, 500000)
expenses = st.sidebar.number_input("Monthly Expenses", 1000, 500000, 15000)
credit_score = st.sidebar.slider("Credit Score", 300, 900, 700)
existing_loans = st.sidebar.slider("Number of Existing Loans", 0, 2, 1)
existing_loan_amount = st.sidebar.number_input("Total Existing Loan Amount", 0, 10000000, 500000)
outstanding_debt = st.sidebar.number_input("Outstanding Debt", 0, 10000000, 100000)
loan_history = st.sidebar.selectbox("Loan History (0=No, 1=Yes)", [0, 1])
loan_amount = st.sidebar.number_input("Loan Amount Requested", 10000, 100000)
loan_term = st.sidebar.selectbox("Loan Term (in Days)", list(range(12, 240)))
loan_purpose = st.sidebar.selectbox("Loan Purpose (1=Personal,2=Education)", [1, 2])
interest_rate = st.sidebar.slider("Interest Rate (%)", 3.5, 15.0, 7.0)
co_applicant = st.sidebar.selectbox("Co-Applicant (0=No, 1=Yes)", [0, 1])
default_risk = st.sidebar.slider("Default Risk Score (0-1)", 0.01, 1.0)

# Prepare input data for prediction
interest_rate = interest_rate / 100
input_data = [[
    age,
    dependents,
    employment_status,
    occupation_type,
    income,
    expenses,
    credit_score,
    existing_loans,
    existing_loan_amount,
    outstanding_debt,
    loan_history,
    loan_amount,
    loan_term,
    loan_purpose,
    interest_rate,
    co_applicant,
    default_risk
]]

# Scale the input data for prediction

scaled_input = scaler.transform(input_data)

# Predict loan approval using the trained model
prediction = model.predict(scaled_input)[0]
prediction_proba = model.predict_proba(scaled_input)[0]

# Display result on Streamlit
if prediction == 1:
    st.success("✅ Loan Approved!")
else:
    st.error("❌ Loan Denied.")

# Feature importance and dynamic visualization
if st.button("Show Feature Importance"):
    # Here we simulate feature importance by using model coefficients for Logistic Regression
    coefficients = model.coef_[0]  # For Logistic Regression
    feature_importance = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': coefficients,
        'Impact on Prediction': coefficients * scaled_input[0]  # This assumes linear relationship
    })
    
    # Sort by absolute impact
    feature_importance['Absolute Impact'] = abs(feature_importance['Impact on Prediction'])
    
    # Calculate the total absolute impact to get percentage
    total_impact = feature_importance['Absolute Impact'].sum()
    
    # Add percentage impact to the dataframe
    feature_importance['Percentage Impact'] = (feature_importance['Absolute Impact'] / total_impact) * 100
    
    # Sort by absolute impact
    feature_importance = feature_importance.sort_values(by='Absolute Impact', ascending=False)

    # Show the feature importance bar chart with percentages
    st.write("### Feature Importance (Impact on Prediction in %)")
    st.bar_chart(feature_importance.set_index('Feature')['Percentage Impact'].head(6))

    # Show the feature impact data for top 6 features
    st.write("### Top 6 Features Impacting the Prediction")
    st.write(feature_importance.head(6))

# Financial health insights based on user inputs
st.write("### Financial Health Insights")

if income > 30000 and credit_score > 750:
    st.write("You have strong financial health. Your loan approval chances are high.")

elif income > 30000 and credit_score < 750:
    st.write("While your income is good, your credit score is low. Work on improving your credit history to increase loan approval chances.")

elif income < 30000 and outstanding_debt > 10000:
    st.write("It is advisable to reduce your outstanding debt and increase your income for better loan approval chances.")

else:
    st.write("You are in an average financial position, but improving your debt-to-income ratio can increase your chances.")


# Financial Health Monitoring Feature
st.write("### Predictive Financial Health Monitoring")

# Calculate Debt-to-Income Ratio (DTI)
dti = (outstanding_debt / income) * 100 if income > 0 else 0

# Calculate Financial Health Score (simplified)
if dti < 20:
    financial_health_score = 90
elif 20 <= dti < 40:
    financial_health_score = 70
else:
    financial_health_score = 50

# Provide Financial Health Score
st.write(f"Your Financial Health Score is: **{financial_health_score}** out of 100")

# Provide advice based on Financial Health Score
if financial_health_score > 75:
    st.write("**Your financial health is strong!** Keep up the good work and maintain your current financial habits.")
elif financial_health_score > 50:
    st.write("**Your financial health is average.** Consider reviewing your expenses and debt. You may want to look at saving more or paying off some debt.")
else:
    st.write("**Your financial health needs improvement.** Work on reducing your debt and increasing your savings for better financial stability.")

# Recommend Regular Financial Check-Ins
st.write("### Financial Health Check-Ins")
st.write("We recommend updating your financial details regularly to track improvements. Consider revisiting this tool every 3-6 months to monitor progress.")



# Plots for Loan Approval Visualization
import plotly.graph_objects as go
# 1. Loan Approval Probability
def plot_prediction_probability(prediction_proba):
    labels = ['Rejected', 'Approved']
    sizes = [prediction_proba[0], prediction_proba[1]]
    colors = ['red', 'green']
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(labels, sizes, color=colors)
    ax.set_ylabel('Probability')
    ax.set_title('Loan Approval Probability')
    ax.set_ylim(0, 1)
    st.pyplot(fig)

plot_prediction_probability(prediction_proba)



# Eligibility Summary Checklist
st.subheader("Eligibility Summary Checklist")

def check_condition(label, condition, value=None, suggestion=""):
    icon = "✅" if condition else "x"
    st.markdown(f"{icon} {label}")
    if value:
        st.markdown(f"  **Current Value:** {value}")
    if not condition and suggestion:
        st.markdown(f"&nbsp;&nbsp;&nbsp; _Suggestion: {suggestion}_")

# Check for financial eligibility conditions
check_condition(
    "Credit Score ≥ 650", 
    credit_score >= 650, 
    f"{credit_score}",
    "Consider improving your credit score by reducing debt or ensuring timely payments."
)


check_condition(
    "No Outstanding Debt > ₹10,000", 
    outstanding_debt <= 10000, 
    f"₹{outstanding_debt:,}",
    "Try to reduce your outstanding debt sometimes loan may reject if it has higher value."
)



check_condition(
    "Co-Applicant Available", 
    co_applicant == 1, 
    "Yes" if co_applicant == 1 else "No",
    "Having a co-applicant is preferred and can also increase your chances of approval."
)

 
check_condition(
    "Annual Income ≥ ₹300,000", 
    income >= 300000, 
    f"₹{income:,}",
    "Consider increasing your income by seeking promotions or additional sources of income. A co-applicant with higher income can also help."
)


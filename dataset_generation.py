# ==========================================================
# AI-Powered Digital Lending Portfolio Optimization
# Dataset Generation Module
# ==========================================================

import pandas as pd
import numpy as np
import random
from faker import Faker
from datetime import datetime

fake = Faker("en_IN")

np.random.seed(42)
random.seed(42)

# ==========================================================
# CONFIGURATION
# ==========================================================

NUM_RECORDS = 50000

# ==========================================================
# LOOKUP DATA
# ==========================================================

states = [
    "Maharashtra",
    "Karnataka",
    "Delhi",
    "Gujarat",
    "Tamil Nadu",
    "West Bengal",
    "Rajasthan",
    "Uttar Pradesh"
]

employment_types = [
    "Salaried",
    "Self-Employed",
    "Business",
    "Freelancer",
    "Government"
]

loan_types = [
    "Personal Loan",
    "Home Loan",
    "Auto Loan",
    "Education Loan",
    "Business Loan"
]

acquisition_channels = [
    "Google Ads",
    "Instagram",
    "Referral",
    "Organic",
    "Partner"
]

risk_grades = ["A", "B", "C", "D"]

# ==========================================================
# HELPER FUNCTIONS
# ==========================================================

def generate_credit_score(income, employment_years):

    score = 550

    if income > 30000:
        score += 30

    if income > 50000:
        score += 30

    if income > 100000:
        score += 40

    if employment_years > 3:
        score += 20

    if employment_years > 7:
        score += 20

    score += random.randint(-50, 50)

    return max(300, min(900, score))


def assign_risk_grade(credit_score):

    if credit_score >= 750:
        return "A"

    elif credit_score >= 680:
        return "B"

    elif credit_score >= 600:
        return "C"

    else:
        return "D"


def calculate_interest_rate(risk_grade):

    rates = {
        "A": random.uniform(8, 11),
        "B": random.uniform(11, 14),
        "C": random.uniform(14, 18),
        "D": random.uniform(18, 25)
    }

    return round(rates[risk_grade], 2)


def calculate_default_probability(
        credit_score,
        income,
        missed_payments,
        balance_volatility):

    prob = 0.05

    if credit_score < 650:
        prob += 0.15

    if credit_score < 600:
        prob += 0.15

    if income < 30000:
        prob += 0.10

    if missed_payments >= 2:
        prob += 0.15

    if missed_payments >= 4:
        prob += 0.15

    if balance_volatility > 0.5:
        prob += 0.10

    return min(prob, 0.95)


def calculate_emi(principal, annual_rate, tenure_months):

    monthly_rate = annual_rate / (12 * 100)

    emi = (
        principal
        * monthly_rate
        * ((1 + monthly_rate) ** tenure_months)
        / (((1 + monthly_rate) ** tenure_months) - 1)
    )

    return round(emi, 2)


# ==========================================================
# DATA GENERATION
# ==========================================================

records = []

print("Generating synthetic lending data...")

for i in range(NUM_RECORDS):

    customer_id = f"CUST{i+1:06d}"
    loan_id = f"LOAN{i+1:06d}"

    age = random.randint(21, 65)

    gender = random.choice([
        "Male",
        "Female"
    ])

    state = random.choice(states)

    city = fake.city()

    employment_type = random.choice(
        employment_types
    )

    years_employed = random.randint(0, 25)

    income = round(
        np.random.lognormal(
            mean=11,
            sigma=0.45
        )
    )

    credit_score = generate_credit_score(
        income,
        years_employed
    )

    risk_grade = assign_risk_grade(
        credit_score
    )

    loan_type = random.choice(
        loan_types
    )

    loan_amount = round(
        income *
        random.uniform(3, 15)
    )

    tenure = random.choice([
        12,
        24,
        36,
        48,
        60,
        84,
        120
    ])

    interest_rate = calculate_interest_rate(
        risk_grade
    )

    emi = calculate_emi(
        loan_amount,
        interest_rate,
        tenure
    )

    monthly_cashflow = round(
        income *
        random.uniform(0.15, 0.60)
    )

    average_balance = round(
        income *
        random.uniform(1, 5)
    )

    balance_volatility = round(
        random.uniform(0.05, 1.0),
        2
    )

    transaction_count = random.randint(
        10,
        300
    )

    spending_shock = round(
        random.uniform(0, 1),
        2
    )

    missed_payments = random.randint(
        0,
        6
    )

    days_past_due = random.randint(
        0,
        180
    )

    partial_payments = random.randint(
        0,
        4
    )

    emis_paid = random.randint(
        0,
        tenure
    )

    acquisition_channel = random.choice(
        acquisition_channels
    )

    customer_acquisition_cost = round(
        random.uniform(500, 5000),
        2
    )

    customer_ltv = round(
        random.uniform(5000, 150000),
        2
    )

    default_probability = calculate_default_probability(
        credit_score,
        income,
        missed_payments,
        balance_volatility
    )

    default = np.random.choice(
        [0, 1],
        p=[
            1 - default_probability,
            default_probability
        ]
    )

    loan_status = (
        "Default"
        if default == 1
        else "Active"
    )

    expected_loss = (
        loan_amount * default_probability
    )

    profit = (
        loan_amount *
        (interest_rate / 100)
        -
        expected_loss
    )

    record = {

        # Customer
        "Customer_ID": customer_id,
        "Age": age,
        "Gender": gender,
        "State": state,
        "City": city,
        "Income": income,
        "Employment_Type": employment_type,
        "Years_Employed": years_employed,
        "Credit_Score": credit_score,

        # Loan
        "Loan_ID": loan_id,
        "Loan_Type": loan_type,
        "Loan_Amount": loan_amount,
        "Interest_Rate": interest_rate,
        "Tenure": tenure,
        "EMI": emi,
        "Risk_Grade": risk_grade,

        # Behaviour
        "Monthly_Cashflow": monthly_cashflow,
        "Average_Balance": average_balance,
        "Balance_Volatility": balance_volatility,
        "Transaction_Count": transaction_count,
        "Spending_Shock": spending_shock,

        # Repayment
        "Missed_Payments": missed_payments,
        "Days_Past_Due": days_past_due,
        "Partial_Payments": partial_payments,
        "EMIs_Paid": emis_paid,

        # Business
        "Acquisition_Channel":
            acquisition_channel,

        "Customer_Acquisition_Cost":
            customer_acquisition_cost,

        "Customer_LTV":
            customer_ltv,

        # Target
        "Default": default,
        "Default_Probability":
            round(default_probability, 4),

        "Loan_Status": loan_status,

        "Expected_Loss":
            round(expected_loss, 2),

        "Profit":
            round(profit, 2)
    }

    records.append(record)

# ==========================================================
# DATAFRAME
# ==========================================================

df = pd.DataFrame(records)

# ==========================================================
# EXPORT
# ==========================================================

filename = "lending_dataset.csv"

df.to_csv(
    filename,
    index=False
)

print("=" * 50)
print("Dataset Generated Successfully")
print("=" * 50)

print(f"Rows: {df.shape[0]}")
print(f"Columns: {df.shape[1]}")

print("\nDefault Rate:")
print(round(df["Default"].mean() * 100, 2), "%")

print(f"\nSaved as: {filename}")

print("\nSample Data:")
print(df.head())
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from sklearn.ensemble import IsolationForest

st.set_page_config(page_title="AML Risk Analytics", layout="wide")

st.title("🏦 Advanced AML Risk Analytics Dashboard")

# --------------------------
# Load Dataset
# --------------------------

@st.cache_data
def load_data():
    return pd.read_csv("aml_large_dataset_300k.csv")

data = load_data()

# --------------------------
# Machine Learning Model
# --------------------------

ml_features = data[["Amount", "RiskScore"]]

model = IsolationForest(contamination=0.02, random_state=42)

data["Anomaly"] = model.fit_predict(ml_features)

data["Anomaly"] = data["Anomaly"].map({1: "Normal", -1: "Suspicious"})

# --------------------------
# Sidebar Filters
# --------------------------

st.sidebar.header("Transaction Search")

bank = st.sidebar.selectbox(
    "Select Bank",
    sorted(data["Bank"].unique())
)

amount = st.sidebar.number_input(
    "Enter Transaction Amount",
    min_value=0,
    step=1000
)

st.sidebar.write("Dataset size:", len(data))

# --------------------------
# Exact Transactions
# --------------------------

exact_transactions = data[
    (data["Bank"] == bank) &
    (data["Amount"] == amount)
]

st.subheader("Customers with Exact Transaction")

if len(exact_transactions) > 0:
    st.dataframe(exact_transactions.head(50))
else:
    st.info("No exact transaction found")

# --------------------------
# Nearby Transactions
# --------------------------

st.subheader("Nearby Transactions (±5000)")

nearby = data[
    (data["Amount"] >= amount - 5000) &
    (data["Amount"] <= amount + 5000)
]

st.dataframe(nearby.head(50))

# --------------------------
# Fraud Risk Gauge
# --------------------------

st.subheader("Fraud Risk Score")

risk_score = None

# case 1 → exact transaction exists
if len(exact_transactions) > 0:
    risk_score = int(exact_transactions.iloc[0]["RiskScore"])

# case 2 → use nearby transactions
elif len(nearby) > 0:
    risk_score = int(nearby["RiskScore"].mean())

if risk_score is not None:

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=risk_score,
        title={'text': "Fraud Risk Score"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 30], 'color': "#2ecc71"},
                {'range': [30, 50], 'color': "#f1c40f"},
                {'range': [50, 70], 'color': "#e67e22"},
                {'range': [70, 100], 'color': "#e74c3c"}
            ],
        }
    ))

    st.plotly_chart(fig, use_container_width=True)

    if risk_score >= 70:
        st.error("⚠ High Risk Transaction - Needs Review")

    elif risk_score >= 50:
        st.warning("Medium Risk - Monitor")

    else:
        st.success("Low Risk Transaction")

else:
    st.info("Risk score unavailable")

# --------------------------
# Risk Category Donut Chart
# --------------------------

st.subheader("Risk Category Distribution")

def classify_risk(score):

    if score >= 70:
        return "Critical"
    elif score >= 50:
        return "High"
    elif score >= 30:
        return "Medium"
    else:
        return "Low"

data["RiskCategory"] = data["RiskScore"].apply(classify_risk)

risk_counts = data["RiskCategory"].value_counts()

fig_risk = go.Figure(data=[go.Pie(
    labels=risk_counts.index,
    values=risk_counts.values,
    hole=0.5,
    marker=dict(colors=[
        "#2ecc71",
        "#f1c40f",
        "#e67e22",
        "#e74c3c"
    ])
)])

fig_risk.update_layout(
    title="Transaction Distribution by Risk Category",
    template="plotly_dark"
)

st.plotly_chart(fig_risk, use_container_width=True)

# --------------------------
# Transactions in Other Banks
# Exact or Nearby
# --------------------------

st.subheader("Transactions in Other Banks")

exact_other = data[
    (data["Amount"] == amount) &
    (data["Bank"] != bank)
]

if len(exact_other) > 0:

    chart_data = (
        exact_other
        .groupby("Bank")
        .size()
        .reset_index(name="Transactions")
    )

    chart_title = "Exact Transaction Amount Found in Other Banks"

else:

    nearby_other = data[
        (data["Amount"] >= amount - 5000) &
        (data["Amount"] <= amount + 5000) &
        (data["Bank"] != bank)
    ]

    chart_data = (
        nearby_other
        .groupby("Bank")
        .size()
        .reset_index(name="Transactions")
    )

    chart_title = "Nearby Transactions (±5000) in Other Banks"

fig_other = px.bar(
    chart_data,
    x="Bank",
    y="Transactions",
    color="Bank",
    title=chart_title
)

fig_other.update_layout(
    template="plotly_dark",
    xaxis_title="Bank",
    yaxis_title="Number of Transactions"
)

st.plotly_chart(fig_other, use_container_width=True)

# --------------------------
# ML Suspicious Transactions
# --------------------------

st.subheader("Machine Learning Detected Suspicious Transactions")

fig_ml = px.scatter(
    data.sample(2000),
    x="Amount",
    y="RiskScore",
    color="Anomaly",
    color_discrete_map={
        "Normal": "green",
        "Suspicious": "red"
    },
    title="ML Anomaly Detection"
)

fig_ml.update_layout(template="plotly_dark")

st.plotly_chart(fig_ml, use_container_width=True)

# --------------------------
# Suspicious Transaction Table
# --------------------------

suspicious = data[data["Anomaly"] == "Suspicious"]

st.subheader("Top Suspicious Transactions")

st.dataframe(suspicious.head(20))
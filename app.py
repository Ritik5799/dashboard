import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="Fraud Detection Dashboard",
    layout="wide"
)

# =====================================================
# LOAD DATA
# =====================================================


@st.cache_data
def load_data():
    return pd.read_csv("sample_data.csv")

data = load_data()


# =====================================================
# LOAD MODEL
# =====================================================

model = joblib.load("model.pkl")

# =====================================================
# SHAP EXPLAINER
# =====================================================

@st.cache_resource
def get_explainer():
    return shap.TreeExplainer(model)

explainer = get_explainer()

# =====================================================
# SIDEBAR
# =====================================================

st.sidebar.title("Fraud Dashboard")

page = st.sidebar.radio(
    "Navigation",
    [
        "Overview",
        "Transaction Explorer",
        "SHAP Explainer",
        "Feature Importance"
    ]
)

# =====================================================
# OVERVIEW PAGE
# =====================================================

if page == "Overview":

    st.title("📊 Fraud Detection Overview")

    total_tx = len(X_test)

    total_fraud = int(sum(y_test))

    fraud_rate = (
        total_fraud / total_tx
    ) * 100

    avg_amt = X_test.loc[
        y_test == 1,
        "TransactionAmt"
    ].mean()

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Transactions",
        f"{total_tx:,}"
    )

    col2.metric(
        "Frauds",
        f"{total_fraud:,}"
    )

    col3.metric(
        "Fraud Rate",
        f"{fraud_rate:.2f}%"
    )

    col4.metric(
        "Avg Fraud Amount",
        f"${avg_amt:.2f}"
    )

    st.divider()

    st.subheader("Fraud Distribution")

    fig, ax = plt.subplots(figsize=(6,4))

    sns.countplot(
        x=pd.Series(y_test).map({
            0: "Legit",
            1: "Fraud"
        }),
        ax=ax
    )

    st.pyplot(fig)

# =====================================================
# TRANSACTION EXPLORER
# =====================================================

elif page == "Transaction Explorer":

    st.title("🔍 Transaction Explorer")

    st.dataframe(X_test.head(50))

    tx_id = st.text_input(
        "Enter TransactionID"
    )

    if tx_id:

        try:

            tx_id = int(tx_id)

            row = X_test[
                X_test["TransactionID"] == tx_id
            ]

            if not row.empty:

                prob = model.predict_proba(row)[0][1]

                st.subheader(
                    "Prediction"
                )

                st.write(
                    f"Fraud Probability: {prob:.2%}"
                )

                st.progress(float(prob))

                if prob >= 0.75:
                    st.error("🚨 Critical Risk")

                elif prob >= 0.40:
                    st.warning("⚠️ Suspicious")

                else:
                    st.success("✅ Legit")

                st.dataframe(row)

            else:
                st.error(
                    "TransactionID not found"
                )

        except:
            st.error(
                "Enter valid TransactionID"
            )

# =====================================================
# SHAP EXPLAINER
# =====================================================

elif page == "SHAP Explainer":

    st.title("🧩 SHAP Explainability")

    tx_id = st.text_input(
        "Enter TransactionID for SHAP"
    )

    if tx_id:

        try:

            tx_id = int(tx_id)

            row = X_test[
                X_test["TransactionID"] == tx_id
            ]

            if not row.empty:

                shap_values = explainer.shap_values(row)

                if isinstance(shap_values, list):
                    shap_val = shap_values[1][0]
                else:
                    shap_val = shap_values[0]

                shap.plots.waterfall(
                    shap.Explanation(
                        values=shap_val,
                        base_values=explainer.expected_value,
                        data=row.iloc[0]
                    ),
                    show=False
                )

                fig = plt.gcf()

                st.pyplot(fig)

            else:
                st.error(
                    "TransactionID not found"
                )

        except:
            st.error(
                "Enter valid TransactionID"
            )

# =====================================================
# FEATURE IMPORTANCE
# =====================================================

elif page == "Feature Importance":

    st.title("📌 Feature Importance")

    importance = pd.DataFrame({

        "Feature": X_train.columns,

        "Importance":
        model.feature_importances_

    })

    importance = importance.sort_values(
        by="Importance",
        ascending=False
    ).head(20)

    fig, ax = plt.subplots(
        figsize=(10,6)
    )

    sns.barplot(
        data=importance,
        x="Importance",
        y="Feature",
        ax=ax
    )

    st.pyplot(fig)

# =====================================================
# MODEL ACCURACY
# =====================================================

y_pred = model.predict(X_test)

acc = accuracy_score(
    y_test,
    y_pred
)

st.sidebar.divider()

st.sidebar.write(
    f"Model Accuracy: {acc:.4f}"
)

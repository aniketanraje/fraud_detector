"""Modular Streamlit UI components — gauges, risk badges, distribution plots, forms."""

from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st


# Risk Colours

_RISK_COLOURS: dict[str, str] = {
    "LOW":      "#22c55e",
    "MEDIUM":   "#f59e0b",
    "HIGH":     "#ef4444",
    "CRITICAL": "#7f1d1d",
}

_RISK_ICONS: dict[str, str] = {
    "LOW":      "✅",
    "MEDIUM":   "⚠️",
    "HIGH":     "🚨",
    "CRITICAL": "💀",
}


# Health Banner

def render_health_banner(is_healthy: bool, model_version: str) -> None:
    """Render a model health status banner.

    Args:
        is_healthy: Whether the model loaded successfully.
        model_version: Loaded model version string.
    """
    if is_healthy:
        st.success(f"🟢 Model loaded — version **{model_version}**", icon="✅")
    else:
        st.error(
            "🔴 Model unavailable — run `python -m src.main --mode train` first.",
            icon="❌",
        )


# Probability Gauge

def render_probability_gauge(probability: float, risk_level: str) -> None:
    """Render a fraud probability gauge using matplotlib.

    Args:
        probability: Fraud probability in [0, 1].
        risk_level: Risk tier string — LOW / MEDIUM / HIGH / CRITICAL.
    """
    colour = _RISK_COLOURS.get(risk_level, "#6b7280")
    icon = _RISK_ICONS.get(risk_level, "")

    fig, ax = plt.subplots(figsize=(4, 2.5), facecolor="none")
    ax.set_facecolor("none")

    # Background bar
    ax.barh(0, 1.0, height=0.4, color="#e5e7eb", zorder=1)
    # Filled bar
    ax.barh(0, probability, height=0.4, color=colour, zorder=2)
    # Threshold line at 0.5
    ax.axvline(x=0.5, color="#374151", linewidth=1.5, linestyle="--", zorder=3)

    ax.set_xlim(0, 1)
    ax.set_ylim(-0.5, 0.5)
    ax.set_yticks([])
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"], fontsize=9)
    ax.set_title(
        f"{icon} Fraud Probability: {probability:.1%}  |  Risk: {risk_level}",
        fontsize=11,
        color=colour,
        fontweight="bold",
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)

    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


# Risk Badge

def render_risk_badge(is_fraud: bool, risk_level: str, probability: float) -> None:
    """Render a styled prediction result card.

    Args:
        is_fraud: Boolean fraud classification.
        risk_level: Risk tier string.
        probability: Fraud probability score.
    """
    colour = _RISK_COLOURS.get(risk_level, "#6b7280")
    icon = _RISK_ICONS.get(risk_level, "")
    label = "FRAUD DETECTED" if is_fraud else "LEGITIMATE"

    st.markdown(
        f"""
        <div style="
            background-color: {colour}22;
            border: 2px solid {colour};
            border-radius: 12px;
            padding: 1rem 1.5rem;
            text-align: center;
            margin: 0.5rem 0;
        ">
            <div style="font-size: 2rem;">{icon}</div>
            <div style="font-size: 1.4rem; font-weight: 700; color: {colour};">{label}</div>
            <div style="font-size: 1rem; color: #374151;">
                Risk Level: <strong>{risk_level}</strong> &nbsp;|&nbsp;
                Probability: <strong>{probability:.2%}</strong>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# Fraud Probability Distribution Plot
def render_probability_distribution(df: pd.DataFrame) -> None:
    """Render a histogram of fraud probabilities from batch results.

    Args:
        df: Batch results DataFrame with a 'fraud_probability' column.
    """
    if "fraud_probability" not in df.columns:
        st.warning("No 'fraud_probability' column found in results.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Histogram
    axes[0].hist(
        df["fraud_probability"],
        bins=50,
        color="steelblue",
        edgecolor="white",
        alpha=0.85,
    )
    axes[0].axvline(x=0.5, color="red", linestyle="--", linewidth=1.5, label="Threshold (0.5)")
    axes[0].set_title("Fraud Probability Distribution", fontweight="bold")
    axes[0].set_xlabel("Fraud Probability")
    axes[0].set_ylabel("Count")
    axes[0].legend()

    # Risk tier pie chart
    risk_counts = df["risk_level"].value_counts()
    pie_colours = [_RISK_COLOURS.get(r, "#6b7280") for r in risk_counts.index]
    axes[1].pie(
        risk_counts.values,
        labels=risk_counts.index,
        colors=pie_colours,
        autopct="%1.1f%%",
        startangle=90,
        wedgeprops={"edgecolor": "white", "linewidth": 1.5},
    )
    axes[1].set_title("Risk Level Breakdown", fontweight="bold")

    fig.suptitle("Batch Prediction Analytics", fontsize=13, fontweight="bold")
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


# Batch Summary Metrics

def render_batch_summary(df: pd.DataFrame) -> None:
    """Render key batch prediction summary metrics as Streamlit metric cards.

    Args:
        df: Batch results DataFrame with 'is_fraud', 'fraud_probability', 'risk_level'.
    """
    total = len(df)
    fraud_count = int(df["is_fraud"].sum()) if "is_fraud" in df.columns else 0
    fraud_rate = (fraud_count / total * 100) if total > 0 else 0.0
    avg_prob = float(df["fraud_probability"].mean()) if "fraud_probability" in df.columns else 0.0
    critical_count = int((df["risk_level"] == "CRITICAL").sum()) if "risk_level" in df.columns else 0

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Transactions", f"{total:,}")
    col2.metric("Fraud Detected", f"{fraud_count:,}", delta=f"{fraud_rate:.1f}%")
    col3.metric("Avg Fraud Probability", f"{avg_prob:.2%}")
    col4.metric("🚨 Critical Alerts", f"{critical_count:,}")


#  Transaction Input Form

_FRAUD_SAMPLE: dict = {
    "Time": 406.0, "V1": -2.312, "V2": 1.952, "V3": -1.610, "V4": 3.998,
    "V5": -0.523, "V6": -1.427, "V7": -2.537, "V8": 1.392, "V9": -2.770,
    "V10": -2.772, "V11": 3.202, "V12": -2.900, "V13": -0.595, "V14": -4.289,
    "V15": 0.390, "V16": -1.141, "V17": -2.830, "V18": -0.017, "V19": 0.416,
    "V20": 0.127, "V21": 0.517, "V22": -0.035, "V23": -0.465, "V24": 0.320,
    "V25": 0.044, "V26": -0.202, "V27": 0.472, "V28": 0.530, "Amount": 239.93,
}

_LEGIT_SAMPLE: dict = {
    "Time": 0.0, "V1": -1.360, "V2": -0.073, "V3": 2.536, "V4": 1.378,
    "V5": -0.338, "V6": 0.462, "V7": 0.240, "V8": 0.099, "V9": 0.364,
    "V10": 0.091, "V11": -0.552, "V12": -0.618, "V13": -0.991, "V14": -0.311,
    "V15": 1.468, "V16": -0.470, "V17": 0.208, "V18": 0.026, "V19": 0.404,
    "V20": 0.251, "V21": -0.018, "V22": 0.278, "V23": -0.110, "V24": 0.067,
    "V25": 0.129, "V26": -0.189, "V27": 0.134, "V28": -0.021, "Amount": 149.62,
}


def render_transaction_form() -> Optional[dict]:
    """Render the single-transaction input form with sample pre-fill buttons.

    Returns:
        Dictionary of field values if the form was submitted, else None.
    """
    col_fill1, col_fill2 = st.columns(2)
    if col_fill1.button("🔴 Pre-fill Fraud Sample"):
        st.session_state["_form_sample"] = _FRAUD_SAMPLE
    if col_fill2.button("🟢 Pre-fill Legit Sample"):
        st.session_state["_form_sample"] = _LEGIT_SAMPLE

    sample = st.session_state.get("_form_sample", _LEGIT_SAMPLE)

    with st.form("transaction_form"):
        st.markdown("#### Transaction Features")

        c1, c2 = st.columns(2)
        time_val = c1.number_input("Time", value=float(sample["Time"]), format="%.2f")
        amount_val = c2.number_input("Amount", value=float(sample["Amount"]), format="%.2f", min_value=0.0)

        st.markdown("**PCA Components (V1–V28)**")

        v_values: dict[str, float] = {}
        cols = st.columns(4)
        for i in range(1, 29):
            key = f"V{i}"
            col_idx = (i - 1) % 4
            v_values[key] = cols[col_idx].number_input(
                key, value=float(sample[key]), format="%.4f", key=f"form_{key}"
            )

        submitted = st.form_submit_button("🔍 Predict", use_container_width=True)

    if submitted:
        return {"Time": time_val, "Amount": amount_val, **v_values}
    return None

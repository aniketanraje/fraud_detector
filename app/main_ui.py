"""Streamlit multi-tab fraud detection interface — single prediction, batch, health check."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# ── Path fix for running as `streamlit run app/main_ui.py` ────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.application.predictor import Predictor
from src.domain.entities import TransactionInput
from src.domain.exceptions import ModelRegistrationError, PredictionError
from app.ui_components import (
    render_batch_summary,
    render_health_banner,
    render_probability_distribution,
    render_probability_gauge,
    render_risk_badge,
    render_transaction_form,
)

logger: logging.Logger = logging.getLogger(__name__)

# Page Config
st.set_page_config(
    page_title="Fraud Detection System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Cached Resource Loading

@st.cache_resource(show_spinner="Loading model...")
def _load_predictor() -> tuple[Predictor | None, str]:
    """Load the Predictor singleton. Cached across all sessions.

    Returns:
        Tuple of (Predictor instance or None, status message string).
    """
    try:
        predictor = Predictor.get_instance()
        if not predictor.is_healthy:
            return None, "Model loaded but health check failed."
        return predictor, f"Loaded — version: {predictor.model_version}"
    except ModelRegistrationError as e:
        return None, str(e)
    except Exception as e:
        logger.exception("Unexpected predictor load error: %s", e)
        return None, f"Unexpected error: {e}"


# Sidebar

def _render_sidebar(predictor: Predictor | None) -> None:
    """Render the sidebar with system status and navigation info.

    Args:
        predictor: Loaded Predictor singleton or None.
    """
    with st.sidebar:
        st.title("🛡️ Fraud Detector")
        st.caption("Credit Card Fraud Detection System")
        st.divider()

        st.subheader("System Status")
        if predictor and predictor.is_healthy:
            st.success("🟢 Model Online")
            st.code(f"Version: {predictor.model_version}", language=None)

            # 🔥 ADD THIS
            model_type = predictor.model_type
            if model_type == "pytorch_mlp":
                st.success("🧠 PyTorch Model Active")
            else:
                st.info("🌲 Sklearn Model Active")
        else:
            st.error("🔴 Model Offline")
            st.code("Run: python -m src.main --mode train", language="bash")

        st.divider()
        st.subheader("Risk Thresholds")
        st.markdown(
            """
            | Level | Range |
            |---|---|
            | ✅ LOW | < 30% |
            | ⚠️ MEDIUM | 30–50% |
            | 🚨 HIGH | 50–80% |
            | 💀 CRITICAL | > 80% |
            """
        )
        st.divider()
        st.markdown(
            """
            <div style="
                border: 1px solid #444;
                padding: 12px;
                border-radius: 8px;
                text-align: center;
                background-color: #111;
            ">
                <b>Aniket Bhosale — Production ML System</b><br><br>
                <a href="mailto:aniketbhosale2808@gmail.com" style="text-decoration:none;">
                    📧 aniketbhosale2808@gmail.com
                </a>
                &nbsp;&nbsp;|&nbsp;&nbsp;
                <a href="tel:+917385542808" style="text-decoration:none;">
                    📞 +91 7385542808
                </a>
                <br><br>
                <a href="https://github.com/aniketanraje/fraud_detector" target="_blank" style="text-decoration:none;">
                    🔗 GitHub Repository
                </a>
            </div>
            """,
            unsafe_allow_html=True
        )
# Tab 1: Single Transaction

def _render_single_prediction(predictor: Predictor | None) -> None:
    """Render the single transaction prediction tab.

    Args:
        predictor: Loaded Predictor singleton or None.
    """
    st.header("🔍 Single Transaction Prediction")

    if predictor is None or not predictor.is_healthy:
        st.error("Model is not available. Please train first.")
        return

    form_data = render_transaction_form()

    if form_data is not None:
        try:
            with st.spinner("Running inference..."):
                transaction = TransactionInput(**form_data)
                result = predictor.predict(transaction)

            st.divider()
            col_badge, col_gauge = st.columns([1, 2])

            with col_badge:
                render_risk_badge(result.is_fraud, result.risk_level, result.probability)
                st.caption(f"Model version: `{result.model_version}` | Type: `{predictor.model_type}`")

            with col_gauge:
                render_probability_gauge(result.probability, result.risk_level)

            if result.risk_level in ("HIGH", "CRITICAL"):
                st.error(
                    f"⚠️ HIGH RISK ALERT — Fraud probability: {result.probability:.2%}. "
                    "Flag for manual review.",
                    icon="🚨",
                )

        except PredictionError as e:
            st.error(f"Prediction failed: {e}")
        except Exception as e:
            st.error(f"Unexpected error: {e}")


# Tab 2: Batch Prediction

def _render_batch_prediction(predictor: Predictor | None) -> None:
    """Render the batch CSV upload and prediction tab.

    Args:
        predictor: Loaded Predictor singleton or None.
    """
    st.header("📦 Batch Prediction")

    if predictor is None or not predictor.is_healthy:
        st.error("Model is not available. Please train first.")
        return

    uploaded = st.file_uploader(
        "Upload a transaction CSV",
        type=["csv"],
        help="Must contain all 30 feature columns (Time, V1–V28, Amount).",
    )

    if uploaded is not None:
        try:
            df = pd.read_csv(uploaded)
            st.info(f"Loaded {len(df):,} rows × {len(df.columns)} columns.")

            with st.spinner(f"Processing {len(df):,} transactions..."):
                results = predictor.predict_batch(df)

            st.divider()
            render_batch_summary(results)
            st.divider()
            render_probability_distribution(results)
            st.divider()

            st.subheader("Results Preview")
            preview_cols = [
                "fraud_probability", "is_fraud", "risk_level", "model_version"
            ]
            available_cols = [c for c in preview_cols if c in results.columns]
            st.dataframe(
                results[available_cols].head(100),
                use_container_width=True,
            )
            st.caption(f"Model used: {predictor.model_type}")

            csv_bytes = results.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="⬇️ Download Full Results CSV",
                data=csv_bytes,
                file_name="fraud_predictions.csv",
                mime="text/csv",
                use_container_width=True,
            )

        except PredictionError as e:
            st.error(f"Batch prediction failed: {e}")
        except Exception as e:
            st.error(f"Unexpected error: {e}")


# Tab 3: Health Check

def _render_health_check(predictor: Predictor | None, status_msg: str) -> None:
    """Render the model health check and system diagnostics tab.

    Args:
        predictor: Loaded Predictor singleton or None.
        status_msg: Human-readable status message from loader.
    """
    st.header("🩺 System Health")

    is_healthy = predictor is not None and predictor.is_healthy
    model_version = predictor.model_version if is_healthy else "N/A"

    render_health_banner(is_healthy, model_version)

    st.divider()
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Model Diagnostics")
        diagnostics = {
            "Status": "✅ Healthy" if is_healthy else "❌ Unavailable",
            "Version": model_version,
            "Model Type": predictor.model_type if is_healthy else "—",
            "Features loaded": str(len(predictor._feature_order)) if is_healthy else "—",
            "Scaler fitted": "Yes" if (is_healthy and predictor._scaler is not None) else "No",

        }

        for k, v in diagnostics.items():
            st.markdown(f"**{k}:** {v}")

    with col2:
        st.subheader("Loader Message")
        st.code(status_msg, language=None)

    st.divider()
    st.subheader("Quick Actions")
    c1, c2 = st.columns(2)

    if c1.button("🔄 Reload Model"):
        Predictor.reset()
        st.cache_resource.clear()
        st.rerun()

    if c2.button("📋 Show Feature Order"):
        if is_healthy:
            st.json(predictor._feature_order)
        else:
            st.warning("Model not loaded.")


# Main App

def main() -> None:
    """Main Streamlit application entrypoint."""
    predictor, status_msg = _load_predictor()
    _render_sidebar(predictor)

    tab1, tab2, tab3 = st.tabs(
        ["🔍 Single Prediction", "📦 Batch Prediction", "🩺 Health Check"]
    )

    with tab1:
        _render_single_prediction(predictor)

    with tab2:
        _render_batch_prediction(predictor)

    with tab3:
        _render_health_check(predictor, status_msg)


if __name__ == "__main__":
    main()

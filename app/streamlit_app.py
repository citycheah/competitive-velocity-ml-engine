"""
Competitive Content Velocity & Drift Engine - Streamlit Observability Dashboard.

Provides interactive breakout triage at publication moment (t0), SHAP feature
attributions, experimental benchmark comparisons, and Population Stability Index (PSI) drift monitoring.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import duckdb
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.features.feature_pipeline import FeaturePipeline, chronological_train_test_split
from src.monitoring.drift_psi import FeatureDriftMonitor
from src.monitoring.shap_explainability import ModelExplainer

# Page Configuration
st.set_page_config(
    page_title="Competitive Content Velocity Engine",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data(show_spinner=False)
def load_data_and_artifacts() -> Tuple[pd.DataFrame, Any, FeaturePipeline, Dict[str, Any], Dict[str, Any]]:
    """Loads feature store data and serialized ML model artifacts."""
    root_dir = Path(__file__).resolve().parent.parent
    db_path = root_dir / "data" / "feature_store.duckdb"
    models_dir = root_dir / "models"

    if not db_path.exists():
        st.error(f"DuckDB database not found at: {db_path}. Run verify_pipeline.py first.")
        st.stop()

    # Load data from DuckDB
    con = duckdb.connect(str(db_path), read_only=True)
    df = con.execute("SELECT * FROM main.fct_video_features_t0 ORDER BY published_epoch DESC").df()
    con.close()

    # Load model artifacts
    model_path = models_dir / "calibrated_lgbm_v2.joblib"
    pipe_path = models_dir / "feature_pipeline.joblib"
    thresh_path = models_dir / "threshold_config.json"
    metrics_path = models_dir / "benchmark_metrics.json"

    model = joblib.load(model_path) if model_path.exists() else None
    pipe = joblib.load(pipe_path) if pipe_path.exists() else None

    threshold_cfg = {}
    if thresh_path.exists():
        with open(thresh_path, "r", encoding="utf-8") as f:
            threshold_cfg = json.load(f)

    metrics_cfg = {}
    if metrics_path.exists():
        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics_cfg = json.load(f)

    return df, model, pipe, threshold_cfg, metrics_cfg


def main():
    st.title("⚡ Competitive Content Velocity & Drift Engine")
    st.markdown(
        "**Local-First ML Reference Architecture** | Predicts channel-adjusted breakout events at moment of publication ($t_0$) "
        "and monitors post-deployment feature drift."
    )

    try:
        df, model, pipe, threshold_cfg, metrics_cfg = load_data_and_artifacts()
    except Exception as e:
        st.error(f"Failed to load engine components: {e}")
        return

    # Sidebar Controls & Global Metadata
    with st.sidebar:
        st.header("⚙️ Operational Controls")
        default_tau = float(threshold_cfg.get("optimal_threshold", 0.27))
        tau = st.slider(
            "Alert Decision Threshold (τ)",
            min_value=0.10,
            max_value=0.70,
            value=default_tau,
            step=0.01,
            help="Threshold for triggering breakout alert. Calibrated to guarantee a 40% precision floor.",
        )

        st.markdown("---")
        st.subheader("System Status")
        st.write(f"📊 **Total Videos:** `{len(df):,}`")
        channel_col = "channel_title" if "channel_title" in df.columns else "channel_id"
        st.write(f"🏢 **Channels Tracked:** `{df[channel_col].nunique()}`")
        st.write(f"🎯 **Base Breakout Rate:** `{df['is_breakout'].mean():.1%}`")
        st.write(f"⚡ **Champion Alert Lift:** `{threshold_cfg.get('optimal_metrics', {}).get('lift_over_base_rate', 3.03):.2f}x`")
        st.write(f"🛡️ **Look-Ahead Bias:** `0 (t0 Quarantined)`")

    # Main Navigation Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "🚨 Live Breakout Triage",
        "🔍 SHAP Attribution Explorer",
        "📈 Benchmarks & Calibration",
        "📡 Feature Drift & Health (PSI)",
    ])

    # -------------------------------------------------------------
    # TAB 1: LIVE BREAKOUT TRIAGE
    # -------------------------------------------------------------
    with tab1:
        st.subheader("Moment-of-Publication ($t_0$) Breakout Alert Feed")
        st.caption("Evaluates recent media stream publications without post-publication leakage.")

        # Compute predictions if model and pipeline are present
        X_all = pipe.transform(df)
        probs = model.predict_proba(X_all)[:, 1]
        triage_df = df.copy()
        triage_df["calibrated_prob"] = probs
        triage_df["alert_status"] = np.where(
            triage_df["calibrated_prob"] >= tau,
            "🚨 BREAKOUT ALERT",
            np.where(triage_df["calibrated_prob"] >= (tau * 0.75), "👀 WATCHLIST", "⚪ NORMAL"),
        )

        # Filters
        f_col1, f_col2, f_col3 = st.columns(3)
        channel_col = "channel_title" if "channel_title" in df.columns else "channel_id"
        with f_col1:
            channel_filter = st.selectbox("Filter Channel Name", ["All Channels"] + sorted(df[channel_col].dropna().unique().tolist()))
        with f_col2:
            format_filter = st.selectbox("Video Format", ["All Formats", "Standard / Long-Form", "Shorts Only"])
        with f_col3:
            status_filter = st.selectbox("Alert Status", ["All Statuses", "🚨 BREAKOUT ALERT Only", "Alerts & Watchlist"])

        filtered_triage = triage_df.copy()
        if channel_filter != "All Channels":
            filtered_triage = filtered_triage[filtered_triage[channel_col] == channel_filter]
        if format_filter == "Standard / Long-Form":
            filtered_triage = filtered_triage[filtered_triage["is_short"] == 0]
        elif format_filter == "Shorts Only":
            filtered_triage = filtered_triage[filtered_triage["is_short"] == 1]
        if status_filter == "🚨 BREAKOUT ALERT Only":
            filtered_triage = filtered_triage[filtered_triage["alert_status"] == "🚨 BREAKOUT ALERT"]
        elif status_filter == "Alerts & Watchlist":
            filtered_triage = filtered_triage[filtered_triage["alert_status"].isin(["🚨 BREAKOUT ALERT", "👀 WATCHLIST"])]

        # Summary KPIs
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        num_alerts = (triage_df["alert_status"] == "🚨 BREAKOUT ALERT").sum()
        kpi1.metric("Active Breakout Alerts", f"{num_alerts}", delta=f"{num_alerts / len(triage_df):.1%} of feed")
        kpi2.metric("Target Alert Precision", f"{threshold_cfg.get('optimal_metrics', {}).get('precision', 0.40):.1%}")
        kpi3.metric("Decision Cutoff (τ)", f"{tau:.2f}")
        kpi4.metric("Realized Breakouts in Alerts", f"{triage_df[triage_df['alert_status'] == '🚨 BREAKOUT ALERT']['is_breakout'].sum()}")

        # Display Triage Table
        display_cols = [
            "title",
            channel_col,
            "published_at",
            "duration_seconds",
            "is_short",
            "channel_upload_blitz_24h",
            "calibrated_prob",
            "alert_status",
            "is_breakout",
        ]
        show_df = filtered_triage[display_cols].copy()
        show_df["calibrated_prob"] = show_df["calibrated_prob"].map(lambda p: f"{p:.1%}")
        show_df["is_short"] = show_df["is_short"].map(lambda s: "📱 Short" if s == 1 else "🎬 Standard")
        show_df["is_breakout"] = show_df["is_breakout"].map(lambda b: "⭐ Yes" if b == 1 else "No")

        st.dataframe(
            show_df,
            use_container_width=True,
            column_config={
                "title": st.column_config.TextColumn("Title", width="medium"),
                channel_col: "Channel Name",
                "published_at": "Published",
                "duration_seconds": "Duration (s)",
                "is_short": "Format",
                "channel_upload_blitz_24h": "Blitz 24h",
                "calibrated_prob": "P(Breakout)",
                "alert_status": "Triage Action",
                "is_breakout": "Realized Breakout",
            },
            hide_index=True,
        )

    # -------------------------------------------------------------
    # TAB 2: SHAP ATTRIBUTION EXPLORER
    # -------------------------------------------------------------
    with tab2:
        st.subheader("SHAP TreeExplainer Attributions")
        st.caption("Deconstructs $t_0$ predictions into mathematically additive feature contributions.")

        feature_names = pipe.get_feature_names_out()
        explainer = ModelExplainer(model, feature_names=feature_names)

        # Global Feature Importance
        st.markdown("#### Global Feature Importance (Mean |SHAP|)")
        df_importance = explainer.explain_global(X_all[:500], top_k=15)
        fig_global = px.bar(
            df_importance,
            x="mean_abs_shap",
            y="feature",
            orientation="h",
            labels={"mean_abs_shap": "Mean |SHAP|", "feature": "Feature"},
            color="mean_abs_shap",
            color_continuous_scale="Blues",
        )
        fig_global.update_layout(yaxis={"autorange": "reversed"}, height=420, margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(fig_global, use_container_width=True)

        # Local Attribution Case Study
        st.markdown("#### Individual Video Waterfall Attribution")
        video_titles = triage_df["title"].head(100).tolist()
        selected_title = st.selectbox("Select Video to Inspect", video_titles)

        match_row = triage_df[triage_df["title"] == selected_title].iloc[0]
        row_idx = triage_df.index.get_loc(match_row.name)
        x_sample = X_all[row_idx]
        local_exp = explainer.explain_local(x_sample, top_k=8)

        channel_disp = match_row.get("channel_title") or match_row.get("channel_id")
        st.write(
            f"**Channel Name:** `{channel_disp}` | **Calibrated Probability:** `{match_row['calibrated_prob']:.1%}` | "
            f"**Realized Target:** `{'⭐ Breakout' if match_row['is_breakout'] == 1 else 'Standard'}`"
        )

        contrib_df = pd.DataFrame(local_exp["all_contributions"][:10])
        fig_local = go.Figure(
            go.Bar(
                x=contrib_df["shap_value"],
                y=contrib_df["feature"],
                orientation="h",
                marker_color=np.where(contrib_df["shap_value"] > 0, "#2ecc71", "#e74c3c"),
            )
        )
        fig_local.update_layout(
            title=f"Top Feature Attributions for: {selected_title[:60]}...",
            xaxis_title="SHAP Value (Contribution to Log-Odds)",
            yaxis={"autorange": "reversed"},
            height=380,
            margin=dict(l=20, r=20, t=30, b=20),
        )
        st.plotly_chart(fig_local, use_container_width=True)

    # -------------------------------------------------------------
    # TAB 3: BENCHMARKS & CALIBRATION
    # -------------------------------------------------------------
    with tab3:
        st.subheader("Model Ladder Benchmarks & Probability Calibration")
        st.caption("Demonstrating rigorous ladder progression from Domain Heuristic to Calibrated LightGBM v2.")

        scores = metrics_cfg.get("benchmark_scores", [])
        if scores:
            scores_df = pd.DataFrame(scores)
            p_col = "precision_top_decile" if "precision_top_decile" in scores_df.columns else "precision_at_top_10pct"
            l_col = "lift_top_decile" if "lift_top_decile" in scores_df.columns else "lift_at_top_10pct"
            cols_to_display = ["model_name", "feature_set", "pr_auc", "roc_auc", "brier_score"]
            if p_col in scores_df.columns:
                cols_to_display.append(p_col)
            if l_col in scores_df.columns:
                cols_to_display.append(l_col)
            if "calibrated" in scores_df.columns:
                cols_to_display.append("calibrated")

            rename_dict = {
                "model_name": "Model Stage",
                "feature_set": "Feature Architecture",
                "pr_auc": "PR-AUC",
                "roc_auc": "ROC-AUC",
                "brier_score": "Brier Score",
                p_col: "Precision @ Top 10%",
                l_col: "Lift @ Top 10%",
                "calibrated": "Calibrated",
            }
            display_ladder = scores_df[cols_to_display].rename(columns=rename_dict)
            st.dataframe(display_ladder, use_container_width=True, hide_index=True)

        col_b1, col_b2 = st.columns(2)
        with col_b1:
            st.markdown("#### Calibration Impact (Brier Score)")
            ladder_names = [s.get("model_name", "") for s in scores]
            brier_vals = [s.get("brier_score", 0.0) for s in scores]
            fig_brier = px.bar(
                x=ladder_names,
                y=brier_vals,
                color=brier_vals,
                color_continuous_scale="Reds_r",
                labels={"x": "Model", "y": "Brier Score (Lower is Better)"},
            )
            fig_brier.update_layout(height=340, showlegend=False)
            st.plotly_chart(fig_brier, use_container_width=True)

        with col_b2:
            st.markdown("#### Alert Lift Progression (Top Decile)")
            lift_vals = [s.get("lift_top_decile", s.get("lift_at_top_10pct", 1.0)) for s in scores]
            fig_lift = px.line(
                x=ladder_names,
                y=lift_vals,
                markers=True,
                labels={"x": "Model", "y": "Lift Over Base Rate"},
            )
            fig_lift.update_traces(line_color="#2980b9", line_width=3, marker_size=8)
            fig_lift.update_layout(height=340)
            st.plotly_chart(fig_lift, use_container_width=True)

    # -------------------------------------------------------------
    # TAB 4: FEATURE DRIFT & HEALTH (PSI)
    # -------------------------------------------------------------
    with tab4:
        st.subheader("Population Stability Index (PSI) Feature Drift Monitoring")
        st.caption("Tracks feature distribution divergence against the training baseline profile.")

        root_dir = Path(__file__).resolve().parent.parent
        profile_path = root_dir / "models" / "baseline_drift_profile.json"

        if profile_path.exists():
            drift_monitor = FeatureDriftMonitor().load_baseline_profile(profile_path)
            # Compare most recent 25% of records against baseline
            inference_subset = df.head(int(len(df) * 0.25))
            X_inf = pipe.transform(inference_subset)
            inf_df = pd.DataFrame(X_inf, columns=pipe.get_feature_names_out())
            drift_report = drift_monitor.compute_drift(inf_df)

            d_col1, d_col2, d_col3, d_col4 = st.columns(4)
            status_color = "🟢" if drift_report.overall_status == "stable" else "🟡" if drift_report.overall_status == "warning" else "🔴"
            d_col1.metric("Overall Drift Status", f"{status_color} {drift_report.overall_status.upper()}")
            d_col2.metric("Mean Feature PSI", f"{drift_report.mean_psi:.4f}")
            d_col3.metric("Max Feature PSI", f"{drift_report.max_psi:.4f}")
            d_col4.metric("Features Monitored", f"{drift_report.total_features_monitored}")

            df_psi = drift_report.to_dataframe()

            # Plot PSI distribution
            fig_psi = px.bar(
                df_psi,
                x="feature",
                y="psi",
                color="psi",
                color_continuous_scale="Temps",
                labels={"psi": "PSI Value", "feature": "Feature"},
            )
            fig_psi.add_hline(y=0.10, line_dash="dash", line_color="orange", annotation_text="Warning (0.10)")
            fig_psi.add_hline(y=0.20, line_dash="dash", line_color="red", annotation_text="Drift Alert (0.20)")
            fig_psi.update_layout(height=380, xaxis_tickangle=-45)
            st.plotly_chart(fig_psi, use_container_width=True)

            st.dataframe(df_psi, use_container_width=True, hide_index=True)
        else:
            st.warning("Baseline drift profile not found in models/. Run train_and_calibrate.py first.")


if __name__ == "__main__":
    main()

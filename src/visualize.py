import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

def plot_forecast(df: pd.DataFrame, forecast: pd.DataFrame, model_name: str):
    """Plot forecasted sales vs historical data."""
    fig = go.Figure()

    # Historical
    fig.add_trace(go.Scatter(
        x=df["ds"], y=df["y"],
        mode="lines+markers",
        name="Historical Sales",
        line=dict(color="blue")
    ))

    # Forecast
    fig.add_trace(go.Scatter(
        x=forecast["ds"], y=forecast["yhat"],
        mode="lines",
        name="Forecast",
        line=dict(color="orange")
    ))

    # Confidence interval
    fig.add_trace(go.Scatter(
        x=pd.concat([forecast["ds"], forecast["ds"][::-1]]),
        y=pd.concat([forecast["yhat_upper"], forecast["yhat_lower"][::-1]]),
        fill="toself",
        fillcolor="rgba(255, 165, 0, 0.2)",
        line=dict(color="rgba(255,165,0,0)"),
        hoverinfo="skip",
        showlegend=True,
        name="Confidence Interval"
    ))

    fig.update_layout(
        title=f"Sales Forecast ({model_name})",
        xaxis_title="Date",
        yaxis_title="Units Sold",
        template="plotly_white"
    )

    st.plotly_chart(fig, use_container_width=True)


def plot_components(prophet_model, forecast: pd.DataFrame):
    """Show Prophet forecast components if Prophet was used."""
    try:
        from prophet.plot import plot_plotly, plot_components_plotly
        st.subheader("🔍 Forecast Components")
        comp_fig = plot_components_plotly(prophet_model, forecast)
        st.plotly_chart(comp_fig, use_container_width=True)
    except Exception as e:
        st.warning(f"Could not display components: {e}")

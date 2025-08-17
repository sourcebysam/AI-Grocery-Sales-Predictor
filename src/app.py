import pandas as pd
import streamlit as st
from predictor import prepare_df, forecast
from visualize import plot_forecast, plot_components

st.set_page_config(page_title="AI Grocery Store Sales Predictor", layout="wide")

st.title("🛒 AI Grocery Store Sales Predictor")

# Sidebar
with st.sidebar:
    st.header("Upload & Settings")
    uploaded = st.file_uploader("Upload CSV with Date, Units_Sold", type=["csv"])
    horizon = st.slider("Forecast horizon (days)", 7, 120, 30, step=1)
    show_components = st.checkbox("Show Prophet components (if available)", value=True)

# Help/Docs expander
with st.expander("CSV Format & Tips", expanded=False):
    st.markdown("""
    **Required columns:** `Date`, `Units_Sold`  
    Optional: `Item`, `Price`, `Profit`.  
    Dates should be in a standard format like `YYYY-MM-DD`.
    """)

# Load data
if uploaded is not None:
    try:
        raw = pd.read_csv(uploaded)
    except Exception as e:
        st.error(f"Failed to read uploaded CSV: {e}")
        st.stop()
else:
    try:
        raw = pd.read_csv("data/sample_sales.csv")
        st.info("Using default sample_sales.csv from data/ folder.")
    except Exception as e:
        st.error("No CSV uploaded and default sample file not found.")
        st.stop()

st.subheader("📄 Data Preview")
st.dataframe(raw.head(20), use_container_width=True)

# Aggregate multiple items if present (sum units per day)
if "Date" in raw.columns and "Units_Sold" in raw.columns:
    agg = (
        raw.assign(Date=pd.to_datetime(raw["Date"], errors="coerce"))
           .dropna(subset=["Date"])
           .groupby("Date", as_index=False)["Units_Sold"].sum()
    )
else:
    st.error("CSV must contain 'Date' and 'Units_Sold' columns.")
    st.stop()

# Historical sales chart
st.subheader("📊 Historical Sales")
import plotly.express as px
fig_hist = px.line(agg, x="Date", y="Units_Sold", title="Units Sold Over Time")
st.plotly_chart(fig_hist, use_container_width=True)

# Prepare and forecast
try:
    model_df = prepare_df(agg)
    fc, model_name, model = forecast(model_df, horizon_days=horizon)  # returns forecast + model
except Exception as e:
    st.error(f"Forecasting failed: {e}")
    st.stop()

st.subheader(f"🔮 Forecast ({model_name}) for next {horizon} days")

# Forecast chart
plot_forecast(model_df, fc, model_name)

# Show forecast table
st.subheader("📑 Forecast Data (last 10 days shown)")
st.dataframe(fc[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(10), use_container_width=True)

# Optional Prophet components
if show_components and model_name == "Prophet":
    plot_components(model, fc)

# Footer
st.markdown("---")
st.caption("⚡ Built with Python, Streamlit & Facebook Prophet | Open Source AI Project")

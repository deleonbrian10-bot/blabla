import os

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from groq import Groq


def get_groq_client_from_key(api_key: str | None) -> Groq | None:
    """
    Return a Groq client if we have an API key, otherwise None.
    Checks the explicit key from the sidebar first, then GROQ_API_KEY env var.
    """
    if api_key:
        return Groq(api_key=api_key)

    # fallback to environment variable if set
    env_key = os.getenv("GROQ_API_KEY")
    if env_key:
        return Groq(api_key=env_key)

    return None

st.title("Revenue by Product Type")

st.markdown("### Groq API key (used for AI Q&A)")
st.sidebar.header("Filters")

# Groq API key input (hidden text)
groq_api_key = st.sidebar.text_input(
    "Groq API key (used for AI Q&A)",
    type="password",
    help="Your Groq API key starting with gsk_. This is only used locally in this session."
)

uploaded = st.file_uploader("Upload your CSV", type=["csv"])


data_path = "Combined_Sales_2025.csv"

if uploaded is not None:
    df = pd.read_csv(uploaded)
else:
    try:
        df = pd.read_csv(data_path)
        st.caption(f"Loaded default file: {data_path}")
    except Exception:
        st.error("Upload a CSV, or add Combined_Sales_2025-2.csv to the repo root.")
        st.stop()
        
df["Net Revenue"] = df["Price (CAD)"] - df["Discount (CAD)"]

df.rename(columns={"weight": "Weight", "width": "Width", "length": "Length"}, inplace=True)

# Minimal validation
needed = {"Product Type", "Net Revenue"}
missing = needed - set(df.columns)
if missing:
    st.error(f"Missing column(s): {', '.join(sorted(missing))}")
    st.stop()

# --- your original code (almost unchanged) ---
rev_by_type = df.groupby("Product Type")["Net Revenue"].sum().sort_values()

plt.figure(figsize=(10,6))
rev_by_type.plot(kind="barh", color="teal")
plt.title("Revenue by Product Type")
plt.xlabel("Total Net Revenue (CAD)")
plt.ylabel("Product Type")

st.pyplot(plt.gcf())
plt.clf()

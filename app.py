import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.title("Revenue by Product Type")

# Option A: upload a CSV
uploaded = st.file_uploader("Upload your CSV", type=["csv"])

# Option B: fall back to a CSV committed in the repo
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

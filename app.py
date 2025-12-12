
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt


def get_groq_client_from_key(api_key: str | None) -> Groq | None:
    """
    Return a Groq client if we have an API key, otherwise None.
    Checks the explicit key from the sidebar first, then GROQ_API_KEY env var.
    """
    if api_key:
        return Groq(api_key=api_key)

    env_key = os.getenv("GROQ_API_KEY")
    if env_key:
        return Groq(api_key=env_key)

    return None


st.title("Revenue by Product Type")

uploaded = st.file_uploader("Upload your CSV", type=["csv"])

data_path = "Combined_Sales_2025.csv"

if uploaded is not None:
    df = pd.read_csv(uploaded)
else:
    try:
        df = pd.read_csv(data_path)
        st.caption(f"Loaded default file: {data_path}")
    except Exception:
        st.error(f"Upload a CSV, or add {data_path} to the repo root.")
        st.stop()

# Net Revenue calculation
df["Net Revenue"] = df["Price (CAD)"] - df["Discount (CAD)"]

# Rename columns (if present)
df.rename(columns={"weight": "Weight", "width": "Width", "length": "Length"}, inplace=True)

# Minimal validation
needed = {"Product Type", "Net Revenue"}
missing = needed - set(df.columns)
if missing:
    st.error(f"Missing column(s): {', '.join(sorted(missing))}")
    st.stop()

# --- Chart: Revenue by Product Type ---
rev_by_type = df.groupby("Product Type")["Net Revenue"].sum().sort_values()

plt.figure(figsize=(10, 6))
rev_by_type.plot(kind="barh", color="teal")
plt.title("Revenue by Product Type")
plt.xlabel("Total Net Revenue (CAD)")
plt.ylabel("Product Type")

st.pyplot(plt.gcf())
plt.clf()

# --- Aggregated table (what you send to Groq) ---
agg_small = (
    df.groupby("Product Type")
      .agg(
          Net_Revenue_Sum=("Net Revenue", "sum"),
          Orders=("Net Revenue", "size")
      )
      .round(2)
      .sort_values("Net_Revenue_Sum", ascending=False)
      .reset_index()
)

# --- Dropdown preview: pick how many rows to preview ---
preview_n = st.selectbox("Preview rows (aggregated table)", [5, 10, 25, 50, "All"], index=1)

st.subheader("Aggregated table preview (sent to Groq)")
if preview_n == "All":
    st.dataframe(agg_small, use_container_width=True)
else:
    st.dataframe(agg_small.head(preview_n), use_container_width=True)

# --- CSV payload you send to Groq ---
payload_csv = agg_small.to_csv(index=False)


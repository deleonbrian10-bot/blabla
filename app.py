import os

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from groq import groq

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
>>getting groq client, requests for API key (paste the key here or in the sidebar


st.title("Revenue by Product Type")
with st.sidebar:
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

# --- Groq AI Q&A ---
st.markdown("### Ask AI about this chart (Groq)")

user_q1 = st.text_area(
    "Question about revenue by product type (Chart 1)",
    key="q_chart1",
    placeholder="e.g. Which product types drive most revenue, and how concentrated is it?"
)

if st.button("Ask AI about Chart 1"):
    if not user_q1.strip():
        st.info("Please enter a question before asking the AI.")
    else:
        client = get_groq_client_from_key(groq_api_key)

        if client is None:
            st.error("Please paste your Groq API key in the sidebar first (or set GROQ_API_KEY).")
        else:
            chart_description = """
Chart 1 shows:
- x-axis: Total Net Revenue
- y-axis: Product Type
"""

            prompt1 = f"""
You are a data analyst interpreting a chart in an ammolite sales dashboard.

CHART CONTEXT:
{chart_description}

DATA (CSV) USED FOR THIS CHART:
{payload_csv}

USER QUESTION:
\"\"\"{user_q1}\"\"\"

INSTRUCTIONS:
- Base your answer ONLY on the CSV data above.
- Start with 1–2 sentences that directly answer the user's question.
- You may add up to 5 short bullet points highlighting key patterns.
- Keep the total answer under about 180 words.
- Do NOT repeat the full chart description or talk about models/APIs; focus on the data and question.
"""

            try:
                resp1 = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[
                        {"role": "system", "content": "You are a careful, concise data analyst."},
                        {"role": "user", "content": prompt1},
                    ],
                    max_completion_tokens=300,
                    temperature=0.3,
                )
                answer1 = resp1.choices[0].message.content
                st.markdown("**AI Insight (Chart 1 - Groq):**")
                st.write(answer1)
            except Exception as e:
                st.error(f"Error calling Groq API for Chart 1: {e}")



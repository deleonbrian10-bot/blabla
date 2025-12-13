import os

import streamlit as st
import pandas as pd
import plotly.express as px
from groq import Groq

# We’ll ask the user for the API key via sidebar instead of env vars.
def get_groq_client_from_key(api_key: str | None):
    if not api_key:
        return None
    return Groq(api_key=api_key)


# ======================
# Load and prepare data
# ======================
@st.cache_data
def load_data():
    df = pd.read_csv("Combined_Sales_2025.csv")

    # Parse dates
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Shipped Date"] = pd.to_datetime(df.get("Shipped Date"), errors="coerce")

    # Ensure numeric price
    df["Price (CAD)"] = pd.to_numeric(df["Price (CAD)"], errors="coerce")

    # --- COA cleaning and validation ---
    # Turn COA into a clean string, strip spaces
    coa_str = df["COA #"].astype(str).str.strip()

    # Blank / missing values become actual None
    df["COA_Clean"] = coa_str.replace(
        {"": None, "nan": None, "NaN": None, "None": None}
    )

    # Valid COA = non-null AND matches COA-######
    valid_mask = df["COA_Clean"].notna() & df["COA_Clean"].str.match(r"^COA-\d{6}$")
    # Invalid COA = non-null BUT does NOT match pattern
    invalid_mask = df["COA_Clean"].notna() & ~valid_mask

    df["COA Valid"] = valid_mask

    # COA Status:
    # - "With COA"    -> valid COA-######
    # - "No COA"      -> blank / missing COA
    # - "Invalid COA" -> non-empty but wrong format
    df["COA Status"] = "No COA"
    df.loc[valid_mask, "COA Status"] = "With COA"
    df.loc[invalid_mask, "COA Status"] = "Invalid COA"

    # --- Export permit cleaning for later validation (used in chart 2) ---
    permit_str = df["Export Permit (PDF link)"].astype(str).str.strip()
    df["Export_Permit_Clean"] = permit_str.replace(
        {"": None, "nan": None, "NaN": None, "None": None}
    )

    # Combine product and grade for unique color mapping (Chart 1)
    df["ProdGrade"] = (
        df["Product Type"].fillna("Unknown Product") + " | " +
        df["Grade"].fillna("Unknown Grade")
    )

    return df


df = load_data()

st.title("Ammolite Sales Dashboard (Compliance)")
st.markdown(
    """
Analyze how **COA presence** (with validated format) and **shipping behaviour**
vary across **Product Types**, **Grades**, and **Countries**.

- Only COAs matching `COA-######` are treated as **valid COAs**
- **Blank COA values are allowed** and treated as **"No COA"**
- Any other non-blank COA values are flagged as **invalid** and excluded
- Chart 1 colors = unique **Product Type + Grade**
- Chart 2 colors = **Grade**, consistent across countries
- Symbols = **COA vs No COA**
- Please enter your Groq API key on the left sidebar first before using this dashboard
"""
)

# ======================
# Flag & show invalid COA rows (dataset-wide)
# ======================
invalid_coa_df = df[df["COA Status"] == "Invalid COA"]

if not invalid_coa_df.empty:
    st.error(
        f"{len(invalid_coa_df)} rows have COA values that do NOT match the 'COA-######' format "
        "and are **excluded** from all charts and filters."
    )
    with st.expander("Show rows with invalid COA format"):
        st.dataframe(
            invalid_coa_df[
                [
                    "Sale ID",
                    "COA #",
                    "Product Type",
                    "Grade",
                    "Price (CAD)",
                    "Country",
                    "Customer Name",
                ]
            ]
        )

# Use only valid-COA and no-COA rows for the dashboard logic
valid_df = df[df["COA Status"] != "Invalid COA"].copy()

# ======================
# Sidebar filters
# ======================
st.sidebar.header("Filters")

# Groq API key input (hidden text)
groq_api_key = st.sidebar.text_input(
    "Groq API key (used for AI Q&A)",
    type="password",
    help="Your Groq API key starting with gsk_. This is only used locally in this session."
)

# Product Type filter (single select)
product_types = sorted(valid_df["Product Type"].dropna().unique())
if product_types:
    selected_product_type = st.sidebar.selectbox(
        "Product Type",
        options=product_types,
        index=0,
    )
else:
    selected_product_type = None

# Grade filter (multi-select)
grades = sorted(valid_df["Grade"].dropna().unique())
selected_grades = st.sidebar.multiselect(
    "Grade",
    options=grades,
    default=grades,
)

# COA filter (All / With COA / Without COA)
coa_filter = st.sidebar.radio(
    "COA Filter",
    options=["All", "With COA", "Without COA"],
    index=0,
)

# Optional: Date range filter
if valid_df["Date"].notna().any():
    min_date = valid_df["Date"].min()
    max_date = valid_df["Date"].max()
    date_range = st.sidebar.date_input(
        "Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
    )
else:
    date_range = None

# ======================
# Apply filters (shared for both charts)
# ======================
filtered = valid_df.copy()

if selected_product_type is not None:
    filtered = filtered[filtered["Product Type"] == selected_product_type]

if selected_grades:
    filtered = filtered[filtered["Grade"].isin(selected_grades)]

if coa_filter == "With COA":
    filtered = filtered[filtered["COA Status"] == "With COA"]
elif coa_filter == "Without COA":
    filtered = filtered[filtered["COA Status"] == "No COA"]

if date_range and isinstance(date_range, (list, tuple)) and len(date_range) == 2:
    start_date, end_date = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
    filtered = filtered[
        (filtered["Date"] >= start_date) &
        (filtered["Date"] <= end_date)
    ]

st.subheader("Filtered Data Snapshot")
st.dataframe(filtered.head(20))

# ======================
# Chart 1: Price vs Date with trendlines per Product+Grade
# ======================
if filtered.empty:
    st.warning("No data matches the current filters.")
else:
    st.subheader("Price vs Date by Product+Grade and COA Status (Valid & Blank COAs Only)")

    fig = px.scatter(
        filtered,
        x="Date",
        y="Price (CAD)",
        color="ProdGrade",
        symbol="COA Status",   # With COA vs No COA
        trendline="ols",
        trendline_scope="trace",  # trendline per color trace
        hover_data=[
            "Sale ID",
            "Product Type",
            "Grade",
            "COA Status",
            "Customer Name",
            "Country",
            "City",
        ],
        title="Price (CAD) over Time with COA vs No COA markers",
    )

    st.plotly_chart(fig, use_container_width=True)

    # Prepare data context for AI (Chart 1)
    summary_cols = [
        "Date", "Price (CAD)", "Product Type",
        "Grade", "COA Status", "Country"
    ]
    summary_df = filtered[summary_cols].sort_values("Date").tail(200)
    context_csv = summary_df.to_csv(index=False)

    # Optional: show data context sent to AI for Chart 1
    with st.expander("Show data sent to AI (Chart 1)", expanded=False):
        st.write("Preview of the rows passed to the Groq model for this chart:")
        st.dataframe(summary_df)
        st.text_area(
            "Raw CSV context for AI (read-only)",
            context_csv,
            height=200,
            key="ctx_chart1",
        )

    # ================
    # Groq AI Q&A for Chart 1
    # ================
    st.markdown("### Ask AI about this pricing chart (Groq)")

    user_q1 = st.text_area(
        "Question about pricing vs COA / grade (Chart 1)",
        key="q_chart1",
        placeholder="e.g. Do items with COAs seem to sell for more than those without for this product?"
    )

    if st.button("Ask AI about Chart 1"):
        if not user_q1.strip():
            st.info("Please enter a question before asking the AI.")
        elif not groq_api_key:
            st.error("Please paste your Groq API key in the sidebar first.")
        else:
            client = get_groq_client_from_key(groq_api_key)

            chart_description = """
Chart 1 shows:
- x-axis: Date
- y-axis: Price (CAD)
- Color: Product Type + Grade (combined)
- Marker symbol: COA Status ("With COA" or "No COA")
"""

            prompt1 = f"""
You are a data analyst interpreting a chart in an ammolite sales dashboard.

CHART CONTEXT:
{chart_description}

DATA (CSV) USED FOR THIS CHART:
{context_csv}

USER QUESTION:
\"\"\"{user_q1}\"\"\"


INSTRUCTIONS:
- Base your answer ONLY on the CSV data above.
- Start with 1–2 sentences that directly answer the user's question.
- If the question asks for trends, comparisons, or explanation in detail,
  you may add up to 5 short bullet points highlighting key patterns.
- Keep the total answer under about 180 words (but be flexible if the question needs more nuance).
- Do NOT repeat the full chart description or talk about models/APIs; focus on the data and question.
"""

            try:
                resp1 = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a careful, concise data analyst.",
                        },
                        {
                            "role": "user",
                            "content": prompt1,
                        },
                    ],
                    max_completion_tokens=300,
                    temperature=0.3,
                )
                answer1 = resp1.choices[0].message.content
                st.markdown("**AI Insight (Chart 1 - Groq):**")
                st.write(answer1)
            except Exception as e:
                st.error(f"Error calling Groq API for Chart 1: {e}")

# ======================
# Chart 2: COA Price Premium by Grade
# ======================
st.subheader("Chart 2: COA Price Premium by Grade")

if filtered.empty:
    st.info("No data available for the current filters to compute COA price premium.")
else:
    price_df = filtered.copy()
    price_df = price_df[price_df["Grade"].notna()]

    agg_price = (
        price_df
        .groupby(["Grade", "COA Status"], dropna=False)
        .agg(
            Avg_Price_CAD=("Price (CAD)", "mean"),
            Sale_Count=("Sale ID", "count"),
        )
        .reset_index()
    )

    if agg_price.empty:
        st.info("No Grade/COA combinations found for the current filters.")
    else:
        grade_order = sorted(agg_price["Grade"].astype(str).unique().tolist())

        fig2 = px.bar(
            agg_price,
            x="Grade",
            y="Avg_Price_CAD",
            color="COA Status",
            barmode="group",
            category_orders={"Grade": grade_order},
            hover_data=["Sale_Count"],
            labels={
                "Grade": "Grade",
                "Avg_Price_CAD": "Average Price (CAD)",
                "COA Status": "COA Status",
            },
            title="Average Price by Grade and COA Status (COA Price Premium)",
        )

        st.plotly_chart(fig2, use_container_width=True)

        # Prepare data context for AI (Chart 2)
        agg_price_rounded = agg_price.copy()
        agg_price_rounded["Avg_Price_CAD"] = agg_price_rounded["Avg_Price_CAD"].round(2)

        context_csv2 = agg_price_rounded.to_csv(index=False)

        with st.expander("Show data sent to AI (Chart 2 - COA Price Premium)", expanded=False):
            st.write("Aggregated data passed to Groq for this chart:")
            st.dataframe(
                agg_price_rounded.sort_values(
                    ["Grade", "COA Status", "Avg_Price_CAD"],
                    ascending=[True, True, False],
                )
            )
            st.text_area(
                "Raw CSV context (Chart 2)",
                context_csv2,
                height=200,
                key="ctx_chart2",
            )

        # ================
        # Groq AI Q&A for Chart 2 (concise bullets)
        # ================
        st.markdown("### Ask AI about Chart 2 (COA price premium)")

        user_q2 = st.text_area(
            "Question about price differences between COA vs No COA by Grade",
            key="q_chart2",
            placeholder="e.g. Which grades gain the largest price premium from having a COA?"
        )

if st.button("Ask AI about Chart 2"):
  if not user_q2.strip():
                st.info("Please enter a question before asking the AI.")
  elif not groq_api_key:
                st.error("Please paste your Groq API key in the sidebar first.")
  else:
                client = get_groq_client_from_key(groq_api_key)

                chart_description = """
Chart 2 shows:
- x-axis: Grade
- y-axis: Average Price (CAD)
- Grouped bars: COA Status ("With COA" vs "No COA") within each grade.
This highlights the COA price premium by grade.
"""

                prompt2 = f"""
You are a data analyst interpreting how COAs affect ammolite prices.

CHART CONTEXT:
{chart_description}

AGGREGATED DATA (CSV) USED FOR THIS CHART:
{context_csv2}

Each row:
- Grade
- COA Status
- Avg_Price_CAD
- Sale_Count

USER QUESTION:
\"\"\"{user_q2}\"\"\"


INSTRUCTIONS:
- Base your answer ONLY on the aggregated CSV data above.
- Start with 1–2 sentences that directly answer the user's question.
- If the question asks about patterns, comparisons, or "which grades benefit most",
  you may add up to 5 short bullet points.
- Keep the total answer under about 180 words (flexible as needed).
- Do NOT discuss models/APIs; focus purely on what the data shows.
"""

                try:
                    resp2 = client.chat.completions.create(
                        model="llama-3.3-70b-versatile",
                        messages=[
                            {
                                "role": "system",
                                "content": "You are a careful, concise data analyst.",
                            },
                            {
                                "role": "user",
                                "content": prompt2,
                            },
                        ],
                        max_completion_tokens=300,
                        temperature=0.3,
                    )
                    answer2 = resp2.choices[0].message.content
                    st.markdown("**AI Insight (Chart 2 - Groq, COA Price Premium):**")
                    st.write(answer2)
                except Exception as e:
                    st.error(f"Error calling Groq API for Chart 2: {e}")
# ======================
# Chart 3: Avg Days from Sale to Shipment by Country & Grade
# ======================
st.subheader("Chart 3: Average Days from Sale to Shipment by Country and Grade")

ship_df = filtered.copy()
ship_df = ship_df[ship_df["Date"].notna() & ship_df["Shipped Date"].notna()]

if ship_df.empty:
    st.info("No rows with both Sale Date and Shipped Date for the current filters.")
else:
    ship_df["Days_to_Ship"] = (ship_df["Shipped Date"] - ship_df["Date"]).dt.days

    invalid_ship_df = ship_df[ship_df["Days_to_Ship"] < 0]

    if not invalid_ship_df.empty:
        st.error(
            f"{len(invalid_ship_df)} rows have **Shipped Date earlier than Sale Date** "
            "and are excluded from the shipping-time chart."
        )
        with st.expander("Show rows with Shipped Date earlier than Sale Date"):
            st.dataframe(
                invalid_ship_df[
                    [
                        "Sale ID",
                        "Date",
                        "Shipped Date",
                        "Days_to_Ship",
                        "Product Type",
                        "Grade",
                        "Country",
                        "Customer Name",
                    ]
                ]
            )

    ship_df_valid = ship_df[ship_df["Days_to_Ship"] >= 0].copy()

    if ship_df_valid.empty:
        st.info("After removing invalid shipping dates, no rows remain for the chart.")
    else:
        non_canada_mask = (
            ship_df_valid["Country"].astype(str).str.lower() != "canada"
        )
        no_permit_mask = ship_df_valid["Export_Permit_Clean"].isna()
        invalid_export_df = ship_df_valid[non_canada_mask & no_permit_mask]

        if not invalid_export_df.empty:
            st.error(
                f"{len(invalid_export_df)} rows ship to **non-Canadian countries without an export permit** "
                "and are excluded from the shipping-time chart."
            )
            with st.expander("Show rows with missing export permits (non-Canada)"):
                st.dataframe(
                    invalid_export_df[
                        [
                            "Sale ID",
                            "Date",
                            "Shipped Date",
                            "Days_to_Ship",
                            "Product Type",
                            "Grade",
                            "Country",
                            "Customer Name",
                            "Export Permit (PDF link)",
                        ]
                    ]
                )

        ship_df_chart = ship_df_valid[~(non_canada_mask & no_permit_mask)].copy()

        if ship_df_chart.empty:
            st.info(
                "After removing invalid shipping and missing-export-permit rows, "
                "no data remains for the shipping-time chart."
            )
        else:
            ship_df_chart["Country_display"] = ship_df_chart["Country"].astype(str)
            ship_df_chart.loc[
                ship_df_chart["Country_display"].str.lower() == "canada",
                "Country_display",
            ] = "Canada"

            agg_ship = (
                ship_df_chart
                .groupby(["Country_display", "Grade"], dropna=False)
                .agg(
                    Avg_Days_to_Ship=("Days_to_Ship", "mean"),
                    Shipment_Count=("Sale ID", "count"),
                )
                .reset_index()
            )

            has_canada = agg_ship["Country_display"].str.lower().eq("canada").any()
            others = sorted(
                agg_ship.loc[
                    ~agg_ship["Country_display"].str.lower().eq("canada"),
                    "Country_display",
                ].dropna().unique().tolist()
            )
            if has_canada:
                ordered_countries = ["Canada"] + others
            else:
                ordered_countries = others

            grade_order = sorted(
                agg_ship["Grade"].dropna().astype(str).unique().tolist()
            )

            grade_color_map = {
                "AAA": "purple",
                "AA": "blue",
                "A": "green",
                "A+": "darkgreen",
                "B": "red",
                "C": "brown",
                "Commercial": "gray",
                "Collectible": "orange",
                "Collectibles": "orange",
            }

            fig3 = px.bar(
                agg_ship,
                x="Country_display",
                y="Avg_Days_to_Ship",
                color="Grade",
                barmode="group",
                category_orders={
                    "Country_display": ordered_countries,
                    "Grade": grade_order,
                },
                color_discrete_map=grade_color_map,
                hover_data=["Shipment_Count"],
                title=(
                    "Average Days from Sale to Shipment by Country and Grade "
                    "(Canada first; invalid / non-permit rows excluded)"
                ),
                labels={
                    "Country_display": "Country",
                    "Avg_Days_to_Ship": "Average Days to Ship",
                },
            )

            st.plotly_chart(fig3, use_container_width=True)

            context_csv3 = agg_ship.to_csv(index=False)

            with st.expander("Show data sent to AI (Chart 3 - Shipping)", expanded=False):
                st.write("Aggregated data passed to Groq for this chart:")
                st.dataframe(
                    agg_ship.sort_values(
                        ["Country_display", "Grade", "Avg_Days_to_Ship"],
                        ascending=[True, True, False],
                    )
                )
                st.text_area(
                    "Raw CSV context (Chart 3)",
                    context_csv3,
                    height=200,
                    key="ctx_chart3",
                )

            # ================
            # Groq AI Q&A for Chart 3 (concise bullets)
            # ================
            st.markdown("### Ask AI about Chart 3 (shipping times)")

            user_q3 = st.text_area(
                "Question about shipping time patterns",
                key="q_chart3",
                placeholder="e.g. Do certain countries or grades tend to have longer shipping times?"
            )
if st.button("Ask AI about Chart 3"):
  if not user_q3.strip():
                    st.info("Please enter a question before asking the AI.")
  elif not groq_api_key:
                    st.error("Please paste your Groq API key in the sidebar first.")
  else:
                    client = get_groq_client_from_key(groq_api_key)

                    chart_description = """
Chart 3 shows:
- x-axis: Country (Canada first, then export markets)
- y-axis: Average Days from Sale to Shipment
- Grouped bars: Grades within each country.
Data excludes negative shipping intervals and exports without permits.
"""

                    prompt3 = f"""
You are a data analyst interpreting shipping delays for ammolite sales.

CHART CONTEXT:
{chart_description}

AGGREGATED DATA (CSV) USED FOR THIS CHART:
{context_csv3}

Each row:
- Country_display
- Grade
- Avg_Days_to_Ship
- Shipment_Count

USER QUESTION:
\"\"\"{user_q3}\"\"\"


INSTRUCTIONS:
- Base your answer ONLY on the aggregated CSV data above.
- Start with 1–2 sentences that directly answer the user's question.
- If the question asks for patterns, comparisons, or outliers,
  you may add up to 5 short bullet points with key insights.
- Keep the total answer under about 180 words.
- Do NOT discuss models/APIs; focus only on shipping patterns in the data.
"""

                    try:
                        resp3 = client.chat.completions.create(
                            model="llama-3.3-70b-versatile",
                            messages=[
                                {
                                    "role": "system",
                                    "content": "You are a careful, concise data analyst.",
                                },
                                {
                                    "role": "user",
                                    "content": prompt3,
                                },
                            ],
                            max_completion_tokens=300,
                            temperature=0.3,
                        )
                        answer3 = resp3.choices[0].message.content
                        st.markdown("**AI Insight (Chart 3 - Groq, Shipping):**")
                        st.write(answer3)
                    except Exception as e:
                        st.error(f"Error calling Groq API for Chart 3: {e}")
import plotly.graph_objects as go

# ======================
# Chart 4: COA Adoption by Price Bucket (combo: stacked bar + line)
# ======================
st.subheader("Chart 4: COA Adoption by Price Bucket")

if filtered.empty:
    st.info("No data available for the current filters to analyze COA adoption by price.")
else:
    price_df = filtered.copy()
    price_df = price_df[price_df["Price (CAD)"].notna()]

    if price_df.empty:
        st.info("No rows with valid prices under the current filters.")
    else:
        st.markdown("#### Price binning for Chart 4")

        num_bins = st.slider(
            "Number of dynamic price bins",
            min_value=3,
            max_value=10,
            value=5,
            key="bins_chart4",
            help="Bins are created using quantiles (qcut). If that fails, equal-width bins are used.",
        )

        # Create dynamic price bins (quantile-based, fallback to equal-width)
        try:
            price_df["PriceBin"] = pd.qcut(
                price_df["Price (CAD)"],
                q=num_bins,
                duplicates="drop"
            )
        except ValueError:
            price_df["PriceBin"] = pd.cut(
                price_df["Price (CAD)"],
                bins=num_bins
            )

        price_df["PriceBinLabel"] = price_df["PriceBin"].astype(str)

        # Counts by PriceBin & COA Status (for stacked bars)
        grp = (
            price_df
            .groupby(["PriceBinLabel", "COA Status"], dropna=False)
            .agg(Sale_Count=("Sale ID", "count"))
            .reset_index()
        )

        if grp.empty:
            st.info("No price bins could be formed with the current filters.")
        else:
            # ---------- ORDERING FIX STARTS HERE ----------
            # Get bin labels in the exact order they appear (for consistent x-axis)
            ordered_labels = list(dict.fromkeys(grp["PriceBinLabel"].astype(str)))

            # Pivot counts and reindex to that order so the line follows the bars
            pivot = grp.pivot(
                index="PriceBinLabel",
                columns="COA Status",
                values="Sale_Count"
            ).reindex(ordered_labels).fillna(0)

            # Ensure we have columns for both statuses even if absent
            with_coa = pivot["With COA"] if "With COA" in pivot.columns else 0
            no_coa = pivot["No COA"] if "No COA" in pivot.columns else 0

            pivot["With_COA_Count"] = with_coa
            pivot["No_COA_Count"] = no_coa
            pivot["Total_Count"] = pivot["With_COA_Count"] + pivot["No_COA_Count"]
            pivot["COA_Rate"] = pivot["With_COA_Count"] / pivot["Total_Count"].replace(0, pd.NA)

            pivot = pivot.reset_index().rename(columns={"PriceBinLabel": "PriceBin"})

            # Sort grp by these ordered labels so bars and line share the same x order
            grp["PriceBinLabel"] = pd.Categorical(
                grp["PriceBinLabel"],
                categories=ordered_labels,
                ordered=True,
            )
            grp = grp.sort_values("PriceBinLabel")
            # ---------- ORDERING FIX ENDS HERE ----------

            # --- Combo figure: stacked bars + line (COA adoption %) ---
            fig4 = go.Figure()

            # Stacked bars for each COA Status
            for status in sorted(grp["COA Status"].dropna().unique()):
                data_status = grp[grp["COA Status"] == status]
                fig4.add_bar(
                    x=data_status["PriceBinLabel"].astype(str),
                    y=data_status["Sale_Count"],
                    name=status,
                )

            # Line for COA adoption rate (%), using the same ordered_labels
            fig4.add_trace(
                go.Scatter(
                    x=ordered_labels,
                    y=(pivot["COA_Rate"] * 100),
                    mode="lines+markers",
                    name="COA adoption (%)",
                    yaxis="y2",
                )
            )

            fig4.update_layout(
                barmode="stack",
                xaxis=dict(title="Price Bucket (CAD)"),
                yaxis=dict(title="Number of Sales"),
                yaxis2=dict(
                    title="COA Adoption (%)",
                    overlaying="y",
                    side="right",
                    range=[0, 100],
                ),
                title="COA Adoption and Sales Volume by Price Bucket",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
            )

            # Make sure x-axis respects that order
            fig4.update_xaxes(categoryorder="array", categoryarray=ordered_labels)

            st.plotly_chart(fig4, use_container_width=True)

            # Prepare aggregated table for AI (one row per price bucket)
            agg_ai = pivot[[
                "PriceBin",
                "With_COA_Count",
                "No_COA_Count",
                "Total_Count",
                "COA_Rate",
            ]].copy()
            agg_ai["COA_Rate"] = (agg_ai["COA_Rate"] * 100).round(1)  # percent

            context_csv4 = agg_ai.to_csv(index=False)

            with st.expander("Show data sent to AI (Chart 4 - COA Adoption by Price)", expanded=False):
                st.write("Aggregated data passed to Groq for this chart:")
                st.dataframe(agg_ai)
                st.text_area(
                    "Raw CSV context (Chart 4)",
                    context_csv4,
                    height=200,
                    key="ctx_chart4",
                )

            # ================
            # Groq AI Q&A for Chart 4 (using your existing dynamic prompt pattern)
            # ================
            st.markdown("### Ask AI about Chart 4 (COA adoption vs price)")

            user_q4 = st.text_area(
                "Question about how COA usage changes with price",
                key="q_chart4",
                placeholder="e.g. At what price range do buyers almost always get COAs?"
            )

            if st.button("Ask AI about Chart 4"):
                if not user_q4.strip():
                    st.info("Please enter a question before asking the AI.")
                elif not groq_api_key:
                    st.error("Please paste your Groq API key in the sidebar first.")
                else:
                    client = get_groq_client_from_key(groq_api_key)

                    chart_description = (
                        "Chart 4 shows stacked bars for number of sales WITH vs WITHOUT COA "
                        "in each dynamic price bucket, plus a line for COA adoption rate (%)."
                    )

                    prompt4 = (
                        "You are a data analyst interpreting how COA usage varies with price.\n\n"
                        "CHART CONTEXT:\n"
                        f"{chart_description}\n\n"
                        "AGGREGATED DATA (CSV) USED FOR THIS CHART:\n"
                        f"{context_csv4}\n\n"
                        "Columns:\n"
                        "- PriceBin: price range label\n"
                        "- With_COA_Count\n"
                        "- No_COA_Count\n"
                        "- Total_Count\n"
                        "- COA_Rate: percentage of items with COA in that bin\n\n"
                        "USER QUESTION:\n"
                        f"\"\"\"{user_q4}\"\"\"\n\n"
                        "INSTRUCTIONS:\n"
                        "- Base your answer ONLY on the aggregated CSV data above.\n"
                        "- Start with 1–2 sentences that directly answer the user's question.\n"
                        "- If the question asks about trends or thresholds, you may add up to 5 short bullet points.\n"
                        "- Keep the total answer under about 180 words.\n"
                        "- Do NOT discuss models/APIs; focus on how COA adoption changes across price bins.\n"
                    )

                    try:
                        resp4 = client.chat.completions.create(
                            model="llama-3.3-70b-versatile",
                            messages=[
                                {
                                    "role": "system",
                                    "content": "You are a careful, concise data analyst.",
                                },
                                {
                                    "role": "user",
                                    "content": prompt4,
                                },
                            ],
                            max_completion_tokens=300,
                            temperature=0.3,
                        )
                        answer4 = resp4.choices[0].message.content
                        st.markdown("**AI Insight (Chart 4 - Groq, COA Adoption vs Price):**")
                        st.write(answer4)
                    except Exception as e:
                        st.error(f"Error calling Groq API for Chart 4: {e}")
# ======================
# Chart 5: Shipping Delay vs Compliance (Violin Plot)
# ======================
st.subheader("Chart 5: Shipping Delay vs Compliance (Violin Plot)")

ship_df = filtered.copy()

# Need valid dates
ship_df = ship_df[ship_df["Date"].notna() & ship_df["Shipped Date"].notna()]

if ship_df.empty:
    st.info("No rows with both Sale Date and Shipped Date for the current filters.")
else:
    # Compute days to ship
    ship_df["Days_to_Ship"] = (ship_df["Shipped Date"] - ship_df["Date"]).dt.days

    # Flag shipped-before-sale rows (invalid)
    invalid_ship_df = ship_df[ship_df["Days_to_Ship"] < 0]

    if not invalid_ship_df.empty:
        st.error(
            f"{len(invalid_ship_df)} rows have **Shipped Date earlier than Sale Date** "
            "and are excluded from the compliance violin plot."
        )
        with st.expander("Show rows with Shipped Date earlier than Sale Date (Chart 5)"):
            st.dataframe(
                invalid_ship_df[
                    [
                        "Sale ID",
                        "Date",
                        "Shipped Date",
                        "Days_to_Ship",
                        "Product Type",
                        "Grade",
                        "Country",
                        "Customer Name",
                    ]
                ]
            )

    # Keep only non-negative days for analysis
    ship_df_valid = ship_df[ship_df["Days_to_Ship"] >= 0].copy()

    if ship_df_valid.empty:
        st.info("After removing invalid shipping dates, no rows remain for this chart.")
    else:
        # Export permit validation for non-Canadian destinations
        non_canada_mask = (
            ship_df_valid["Country"].astype(str).str.lower() != "canada"
        )
        no_permit_mask = ship_df_valid["Export_Permit_Clean"].isna()
        invalid_export_df = ship_df_valid[non_canada_mask & no_permit_mask]

        if not invalid_export_df.empty:
            st.error(
                f"{len(invalid_export_df)} rows ship to **non-Canadian countries without an export permit** "
                "and are excluded from the compliance violin plot."
            )
            with st.expander("Show rows with missing export permits (non-Canada, Chart 5)"):
                st.dataframe(
                    invalid_export_df[
                        [
                            "Sale ID",
                            "Date",
                            "Shipped Date",
                            "Days_to_Ship",
                            "Product Type",
                            "Grade",
                            "Country",
                            "Customer Name",
                            "Export Permit (PDF link)",
                        ]
                    ]
                )

        # Exclude export rows with missing permits
        ship_df_chart = ship_df_valid[~(non_canada_mask & no_permit_mask)].copy()

        if ship_df_chart.empty:
            st.info(
                "After removing invalid shipping and missing-export-permit rows, "
                "no data remains for the compliance violin plot."
            )
        else:
            # Normalize country label and define compliance dimensions
            ship_df_chart["Country_display"] = ship_df_chart["Country"].astype(str)
            ship_df_chart.loc[
                ship_df_chart["Country_display"].str.lower() == "canada",
                "Country_display",
            ] = "Canada"

            is_domestic = ship_df_chart["Country_display"].str.lower() == "canada"
            has_coa = ship_df_chart["COA Status"] == "With COA"
            # For exports, has_permit = non-null export permit; for domestic we treat as True
            has_permit = ship_df_chart["Export_Permit_Clean"].notna() | is_domestic

            # Build compliance group labels
            ship_df_chart["Compliance_Group"] = "Domestic - No COA"
            ship_df_chart.loc[is_domestic & has_coa, "Compliance_Group"] = "Domestic - With COA"
            ship_df_chart.loc[~is_domestic & has_coa & has_permit, "Compliance_Group"] = "Export - With COA & Permit"
            ship_df_chart.loc[~is_domestic & ~has_coa & has_permit, "Compliance_Group"] = "Export - No COA & Permit"

            # Only keep groups that actually appear
            possible_order = [
                "Domestic - No COA",
                "Domestic - With COA",
                "Export - No COA & Permit",
                "Export - With COA & Permit",
            ]
            present_groups = [
                g for g in possible_order
                if g in ship_df_chart["Compliance_Group"].unique()
            ]

            if not present_groups:
                st.info("No compliance groups available for the current filters.")
            else:
                ship_df_chart["Compliance_Group"] = pd.Categorical(
                    ship_df_chart["Compliance_Group"],
                    categories=present_groups,
                    ordered=True,
                )

                st.markdown(
                    """
This violin plot compares the **distribution of shipping delays** (days from sale to shipment)
across different **compliance groups** (domestic vs export, COA vs no COA, permits).
"""
                )

                # Violin plot: distribution of Days_to_Ship by compliance group
                fig5 = px.violin(
                    ship_df_chart,
                    x="Compliance_Group",
                    y="Days_to_Ship",
                    color="Compliance_Group",
                    box=True,          # show inner box plot (median, quartiles)
                    points="all",      # show individual points for context
                    title="Shipping Delay Distribution by Compliance Group",
                    labels={
                        "Compliance_Group": "Compliance Group",
                        "Days_to_Ship": "Days from Sale to Shipment",
                    },
                )

                fig5.update_layout(xaxis_tickangle=-20)
                st.plotly_chart(fig5, use_container_width=True)

                # Prepare aggregated stats for AI (one row per compliance group)
                agg_ai5 = (
                    ship_df_chart
                    .groupby("Compliance_Group", dropna=False)
                    .agg(
                        Mean_Days=("Days_to_Ship", "mean"),
                        Median_Days=("Days_to_Ship", "median"),
                        Min_Days=("Days_to_Ship", "min"),
                        Max_Days=("Days_to_Ship", "max"),
                        Shipment_Count=("Sale ID", "count"),
                    )
                    .reset_index()
                )
                agg_ai5[["Mean_Days", "Median_Days"]] = agg_ai5[["Mean_Days", "Median_Days"]].round(1)

                context_csv5 = agg_ai5.to_csv(index=False)

                with st.expander("Show data sent to AI (Chart 5 - Shipping vs Compliance)", expanded=False):
                    st.write("Aggregated shipping-delay stats passed to Groq for this chart:")
                    st.dataframe(agg_ai5)
                    st.text_area(
                        "Raw CSV context (Chart 5)",
                        context_csv5,
                        height=200,
                        key="ctx_chart5",
                    )

                # ================
                # Groq AI Q&A for Chart 5 (dynamic, concise)
                # ================
                st.markdown("### Ask AI about Chart 5 (shipping delay vs compliance)")

                user_q5 = st.text_area(
                    "Question about how compliance affects shipping delays",
                    key="q_chart5",
                    placeholder="e.g. Do fully compliant exports tend to ship faster than non-compliant or domestic shipments?"
                )

                if st.button("Ask AI about Chart 5"):
                    if not user_q5.strip():
                        st.info("Please enter a question before asking the AI.")
                    elif not groq_api_key:
                        st.error("Please paste your Groq API key in the sidebar first.")
                    else:
                        client = get_groq_client_from_key(groq_api_key)

                        chart_description = """
Chart 5 shows:
- x-axis: Compliance groups (domestic vs export, COA vs no COA, permit status).
- y-axis: Days from sale to shipment.
- Violin shapes show the full distribution; inner box marks median and quartiles.
"""

                        prompt5 = f"""
You are a data analyst interpreting shipping delays for ammolite sales.

CHART CONTEXT:
{chart_description}

AGGREGATED DATA (CSV) USED FOR THIS CHART:
{context_csv5}

Each row:
- Compliance_Group
- Mean_Days
- Median_Days
- Min_Days
- Max_Days
- Shipment_Count

USER QUESTION:
\"\"\"{user_q5}\"\"\"


INSTRUCTIONS:
- Base your answer ONLY on the aggregated CSV data above.
- Start with 1–2 sentences that directly answer the user's question.
- If the question asks about patterns, comparisons, or which groups are better/worse,
  you may add up to 5 short bullet points.
- Keep the total answer under about 180 words.
- Do NOT discuss models/APIs; focus purely on how shipping delay differs by compliance group.
"""

                        try:
                            resp5 = client.chat.completions.create(
                                model="llama-3.3-70b-versatile",
                                messages=[
                                    {
                                        "role": "system",
                                        "content": "You are a careful, concise data analyst.",
                                    },
                                    {
                                        "role": "user",
                                        "content": prompt5,
                                    },
                                ],
                                max_completion_tokens=300,
                                temperature=0.3,
                            )
                            answer5 = resp5.choices[0].message.content
                            st.markdown("**AI Insight (Chart 5 - Groq, Shipping vs Compliance):**")
                            st.write(answer5)
                        except Exception as e:
                            st.error(f"Error calling Groq API for Chart 5: {e}")
import numpy as np

# ======================
# Chart 6: Compliance Score vs Average Order Value (Domestic vs Export)
# ======================
st.subheader("Chart 6: Compliance Score vs Average Order Value (Domestic vs Export)")

comp_df = filtered.copy()
comp_df = comp_df[comp_df["Price (CAD)"].notna()]

if comp_df.empty:
    st.info("No data with valid prices under the current filters to compute compliance scores.")
else:
    # Market type
    country_str = comp_df["Country"].astype(str)
    is_domestic = country_str.str.lower() == "canada"
    comp_df["Market_Type"] = np.where(is_domestic, "Domestic", "Export")

    # Compliance ingredients
    has_coa = comp_df["COA Status"] == "With COA"
    has_permit = comp_df["Export_Permit_Clean"].notna()

    # Initialize scores as 0
    comp_df["Compliance_Score"] = 0

    # Domestic:
    # - With COA -> fully compliant (2)
    comp_df.loc[is_domestic & has_coa, "Compliance_Score"] = 2
    # Domestic & no COA remain 0

    # Export:
    exp_mask = ~is_domestic

    # Export fully compliant: COA + permit
    comp_df.loc[exp_mask & has_coa & has_permit, "Compliance_Score"] = 2

    # Export partial compliance: COA only OR permit only
    partial_mask = exp_mask & (
        (has_coa & ~has_permit) | (~has_coa & has_permit)
    )
    comp_df.loc[partial_mask, "Compliance_Score"] = 1

    # Export with no COA and no permit remain 0

    # Human-readable labels
    score_label_map = {
        0: "0 - No COA / No Permit",
        1: "1 - Partial Compliance",
        2: "2 - Fully Compliant",
    }
    comp_df["Compliance_Score_Label"] = comp_df["Compliance_Score"].map(score_label_map)

    # Aggregate average price per Market_Type & Compliance_Score
    agg_comp = (
        comp_df
        .groupby(["Market_Type", "Compliance_Score", "Compliance_Score_Label"], dropna=False)
        .agg(
            Avg_Price_CAD=("Price (CAD)", "mean"),
            Order_Count=("Sale ID", "count"),
        )
        .reset_index()
    )

    if agg_comp.empty:
        st.info("No combinations of market type and compliance score for the current filters.")
    else:
        agg_comp["Avg_Price_CAD"] = agg_comp["Avg_Price_CAD"].round(2)

        score_order = [
            "0 - No COA / No Permit",
            "1 - Partial Compliance",
            "2 - Fully Compliant",
        ]
        market_order = ["Domestic", "Export"]

        fig6 = px.bar(
            agg_comp,
            x="Compliance_Score_Label",
            y="Avg_Price_CAD",
            color="Market_Type",
            category_orders={
                "Compliance_Score_Label": score_order,
                "Market_Type": market_order,
            },
            barmode="group",
            hover_data=["Order_Count"],
            labels={
                "Compliance_Score_Label": "Compliance Score",
                "Avg_Price_CAD": "Average Order Value (CAD)",
                "Market_Type": "Market",
            },
            title="Average Order Value by Compliance Score and Market Type",
        )

        st.plotly_chart(fig6, use_container_width=True)

        # Prepare data context for AI
        context_csv6 = agg_comp[
            ["Market_Type", "Compliance_Score", "Compliance_Score_Label", "Avg_Price_CAD", "Order_Count"]
        ].to_csv(index=False)

        with st.expander("Show data sent to AI (Chart 6 - Compliance vs Order Value)", expanded=False):
            st.write("Aggregated data passed to Groq for this chart:")
            st.dataframe(
                agg_comp.sort_values(
                    ["Market_Type", "Compliance_Score"],
                    ascending=[True, True],
                )
            )
            st.text_area(
                "Raw CSV context (Chart 6)",
                context_csv6,
                height=200,
                key="ctx_chart6",
            )

        # ================
        # Groq AI Q&A for Chart 6
        # ================
        st.markdown("### Ask AI about Chart 6 (compliance vs order value)")

        user_q6 = st.text_area(
            "Question about how compliance and market type affect order value",
            key="q_chart6",
            placeholder="e.g. Do fully compliant exports have higher average value than domestic non-compliant orders?"
        )

        if st.button("Ask AI about Chart 6"):
            if not user_q6.strip():
                st.info("Please enter a question before asking the AI.")
            elif not groq_api_key:
                st.error("Please paste your Groq API key in the sidebar first.")
            else:
                client = get_groq_client_from_key(groq_api_key)

                chart_description = """
Chart 6 shows:
- x-axis: Compliance score (0, 1, 2) with labels.
- y-axis: Average Order Value (CAD).
- Bars grouped by market type (Domestic vs Export).
"""

                prompt6 = f"""
You are a data analyst interpreting how compliance level affects order value.

CHART CONTEXT:
{chart_description}

AGGREGATED DATA (CSV) USED FOR THIS CHART:
{context_csv6}

Columns:
- Market_Type (Domestic / Export)
- Compliance_Score (0, 1, 2)
- Compliance_Score_Label
- Avg_Price_CAD
- Order_Count

USER QUESTION:
\"\"\"{user_q6}\"\"\"


INSTRUCTIONS:
- Base your answer ONLY on the aggregated CSV data above.
- Start with 1–2 sentences that directly answer the user's question.
- If the question asks about patterns, comparisons, or thresholds,
  you may add up to 5 short bullet points.
- Keep the total answer under about 180 words.
- Do NOT discuss models/APIs; focus on how order value differs by compliance and market.
"""

                try:
                    resp6 = client.chat.completions.create(
                        model="llama-3.3-70b-versatile",
                        messages=[
                            {
                                "role": "system",
                                "content": "You are a careful, concise data analyst.",
                            },
                            {
                                "role": "user",
                                "content": prompt6,
                            },
                        ],
                        max_completion_tokens=300,
                        temperature=0.3,
                    )
                    answer6 = resp6.choices[0].message.content
                    st.markdown("**AI Insight (Chart 6 - Groq, Compliance vs Order Value):**")
                    st.write(answer6)
                except Exception as e:
                    st.error(f"Error calling Groq API for Chart 6: {e}")
# ======================
# Chart 7: COA Coverage by Product Type (Grade + Date filters only)
# ======================
st.subheader("Chart 7: COA Coverage by Product Type")

st.markdown(
    """
This chart uses the **full dataset minus invalid COAs**, filtered by:

- **Grade** (from the sidebar)
- **Date Range** (from the sidebar)

It **ignores the Product Type and COA filters**, so you always see COA coverage
across all product types for the selected grades and dates.
"""
)

# Start from valid_df (invalid COAs already removed earlier)
coverage_df = valid_df.copy()

# Apply Grade filter
if selected_grades:
    coverage_df = coverage_df[coverage_df["Grade"].isin(selected_grades)]

# Apply Date range filter (same logic as used for `filtered`)
if date_range and isinstance(date_range, (list, tuple)) and len(date_range) == 2:
    start_date, end_date = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
    coverage_df = coverage_df[
        coverage_df["Date"].notna() &
        (coverage_df["Date"] >= start_date) &
        (coverage_df["Date"] <= end_date)
    ]

# Require Product Type
coverage_df = coverage_df[coverage_df["Product Type"].notna()]

if coverage_df.empty:
    st.info(
        "No data available for the selected grade(s) and date range "
        "to compute COA coverage by product type."
    )
else:
    # Count by Product Type & COA Status
    grp7 = (
        coverage_df
        .groupby(["Product Type", "COA Status"], dropna=False)
        .agg(Sale_Count=("Sale ID", "count"))
        .reset_index()
    )

    if grp7.empty:
        st.info("No Product Type / COA combinations found for the current filters.")
    else:
        # Pivot to get With COA / No COA counts per product type
        pivot7 = grp7.pivot(
            index="Product Type",
            columns="COA Status",
            values="Sale_Count"
        ).fillna(0)

        # Ensure we have both columns even if missing in the data
        with_coa = pivot7["With COA"] if "With COA" in pivot7.columns else 0
        no_coa = pivot7["No COA"] if "No COA" in pivot7.columns else 0

        pivot7["With_COA_Count"] = with_coa
        pivot7["No_COA_Count"] = no_coa
        pivot7["Total_Count"] = pivot7["With_COA_Count"] + pivot7["No_COA_Count"]
        pivot7["COA_Rate"] = pivot7["With_COA_Count"] / pivot7["Total_Count"].replace(0, pd.NA)

        # Reset index to get Product Type as a column
        cov_df = pivot7.reset_index()
        cov_df.rename(columns={"Product Type": "Product_Type"}, inplace=True)

        # COA coverage as %
        cov_df["COA_Rate_Percent"] = (cov_df["COA_Rate"] * 100).round(1)

        # Sort so the bars are nicely ordered (low coverage at bottom, high at top)
        cov_df = cov_df.sort_values("COA_Rate_Percent", ascending=True)

        fig7 = px.bar(
            cov_df,
            x="COA_Rate_Percent",
            y="Product_Type",
            orientation="h",
            text="COA_Rate_Percent",
            labels={
                "COA_Rate_Percent": "COA Coverage (%)",
                "Product_Type": "Product Type",
            },
            title="COA Coverage by Product Type (filtered by Grade & Date)",
        )

        fig7.update_traces(
            texttemplate="%{text:.1f}%",
            textposition="outside",
            cliponaxis=False,
        )
        fig7.update_layout(
            xaxis=dict(range=[0, 100]),
            margin=dict(l=120, r=40, t=60, b=40),
        )

        st.plotly_chart(fig7, use_container_width=True)

        # Prepare data context for AI
        agg_ai7 = cov_df[[
            "Product_Type",
            "With_COA_Count",
            "No_COA_Count",
            "Total_Count",
            "COA_Rate_Percent",
        ]].copy()

        context_csv7 = agg_ai7.to_csv(index=False)

        with st.expander("Show data sent to AI (Chart 7 - COA Coverage by Product Type)", expanded=False):
            st.write("Aggregated coverage data passed to Groq for this chart:")
            st.dataframe(agg_ai7)
            st.text_area(
                "Raw CSV context (Chart 7)",
                context_csv7,
                height=200,
                key="ctx_chart7",
            )

        # ================
        # Groq AI Q&A for Chart 7
        # ================
        st.markdown("### Ask AI about Chart 7 (COA coverage by product type)")

        user_q7 = st.text_area(
            "Question about COA coverage patterns across product types",
            key="q_chart7",
            placeholder="e.g. Which product types are least documented with COAs for the selected grade(s) and date range?"
        )

        if st.button("Ask AI about Chart 7"):
            if not user_q7.strip():
                st.info("Please enter a question before asking the AI.")
            elif not groq_api_key:
                st.error("Please paste your Groq API key in the sidebar first.")
            else:
                client = get_groq_client_from_key(groq_api_key)

                chart_description = """
Chart 7 shows:
- Horizontal bars: COA coverage percentage for each product type.
- Coverage = With COA / (With COA + No COA),
  computed on the full dataset minus invalid COAs,
  filtered only by Grade and Date range.
"""

                prompt7 = f"""
You are a data analyst interpreting COA coverage across product types.

CHART CONTEXT:
{chart_description}

AGGREGATED DATA (CSV) USED FOR THIS CHART:
{context_csv7}

Columns:
- Product_Type
- With_COA_Count
- No_COA_Count
- Total_Count
- COA_Rate_Percent

USER QUESTION:
\"\"\"{user_q7}\"\"\"


INSTRUCTIONS:
- Base your answer ONLY on the aggregated CSV data above.
- Start with 1–2 sentences that directly answer the user's question.
- If the question asks about patterns, rankings, or gaps,
  you may add up to 5 short bullet points.
- Keep the total answer under about 180 words.
- Do NOT discuss models/APIs; focus on how COA coverage varies by product type.
"""

                try:
                    resp7 = client.chat.completions.create(
                        model="llama-3.3-70b-versatile",
                        messages=[
                            {
                                "role": "system",
                                "content": "You are a careful, concise data analyst.",
                            },
                            {
                                "role": "user",
                                "content": prompt7,
                            },
                        ],
                        max_completion_tokens=300,
                        temperature=0.3,
                    )
                    answer7 = resp7.choices[0].message.content
                    st.markdown("**AI Insight (Chart 7 - Groq, COA Coverage by Product Type):**")
                    st.write(answer7)
                except Exception as e:
                    st.error(f"Error calling Groq API for Chart 7: {e}")

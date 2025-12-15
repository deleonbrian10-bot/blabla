# TAB: PRODUCT MIX ✅ (ONLY shows inside Product Mix tab)
# ======================
with tab_mix:
    st.header("🧩 Product Mix")

    # ✅ FIX: Directly use the filtered dataframe 'f' from master dashboard
    # This ensures sidebar filters work properly
    pm_df = f.copy()

    # ---- Safety: required columns + numeric cleanup ----
    for col in ["Price (CAD)", "Discount (CAD)"]:
        if col in pm_df.columns:
            pm_df[col] = pd.to_numeric(pm_df[col], errors="coerce").fillna(0)
        else:
            pm_df[col] = 0

    if "Sale ID" not in pm_df.columns:
        pm_df["Sale ID"] = np.arange(len(pm_df)) + 1

    for col in ["Product Type", "Grade", "Species"]:
        if col not in pm_df.columns:
            pm_df[col] = "Unknown"
        pm_df[col] = pm_df[col].fillna("Unknown")

    # Optional fallback CSS
    st.markdown(
        """
        <style>
        .insight-box{
            padding: 14px 16px;
            border: 1px solid rgba(0,0,0,0.08);
            border-radius: 12px;
            background: rgba(0,0,0,0.02);
            margin: 8px 0 14px 0;
        }
        .recommendation-box{
            padding: 14px 16px;
            border: 1px solid rgba(0,0,0,0.10);
            border-radius: 12px;
            background: rgba(255, 193, 7, 0.10);
            margin: 8px 0 14px 0;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Create sub-tabs for Product Mix section (NO ICONS as requested)
    pm_tabs = st.tabs([
        "Overview",
        "Interactive Treemap",
        "Revenue Analysis",
        "Pricing Analysis",
        "Efficiency Metrics",
        "Strategic Insights"
    ])

    # =====================================================
    # TAB 1: OVERVIEW (NO CHARTS - removed as requested)
    # =====================================================
    with pm_tabs[0]:
        st.subheader("Executive Summary")

        col1, col2, col3, col4 = st.columns(4)

        total_rev = pm_df["Price (CAD)"].sum()
        total_sales = len(pm_df)
        avg_txn = pm_df["Price (CAD)"].mean() if total_sales else 0
        mean_price = pm_df["Price (CAD)"].mean() if total_sales else 0
        mean_disc = pm_df["Discount (CAD)"].mean() if total_sales else 0
        avg_disc = (mean_disc / mean_price * 100) if mean_price else 0

        with col1:
            st.metric("Total Revenue", f"${total_rev:,.0f}", "CAD")
        with col2:
            st.metric("Total Sales", f"{total_sales:,}", "transactions")
        with col3:
            st.metric("Avg Transaction", f"${avg_txn:,.2f}")
        with col4:
            st.metric("Avg Discount", f"{avg_disc:.2f}%", "Strong pricing")

        st.markdown("---")

        # DYNAMIC KEY FINDINGS
        st.subheader("Key Findings")
        st.markdown('<div class="insight-box">', unsafe_allow_html=True)
        
        # Determine pricing assessment (DYNAMIC)
        if avg_disc < 1:
            pricing_assessment = "exceptionally low"
            pricing_quality = "exceptional"
        elif avg_disc < 2:
            pricing_assessment = "very low"
            pricing_quality = "strong"
        elif avg_disc < 3:
            pricing_assessment = "low"
            pricing_quality = "good"
        else:
            pricing_assessment = "moderate"
            pricing_quality = "developing"
        
        # Determine market tier (DYNAMIC)
        if avg_txn > 5000:
            market_tier = "ultra-premium"
        elif avg_txn > 2000:
            market_tier = "premium"
        elif avg_txn > 1000:
            market_tier = "mid-tier premium"
        else:
            market_tier = "accessible"
        
        st.markdown(
            f"""
            Total revenue of **${total_rev:,.0f} CAD** across **{total_sales:,} transactions**
            with an average of **${avg_txn:,.2f}** per sale. The {pricing_assessment} discount rate of 
            **{avg_disc:.2f}%** (vs. 5-15% industry average) demonstrates {pricing_quality} brand value and 
            pricing power, positioning the business in the **{market_tier} market segment**.
            """,
            unsafe_allow_html=True
        )
        st.markdown("</div>", unsafe_allow_html=True)

        # Top/Bottom product stats
        product_stats = (
            pm_df.groupby("Product Type", as_index=False)
            .agg(**{"Total Revenue": ("Price (CAD)", "sum"), "Sales Volume": ("Sale ID", "count")})
            .sort_values("Total Revenue", ascending=False)
        )

        st.markdown("---")
        colL, colR = st.columns(2)

        with colL:
            st.markdown("### Top 3 Revenue Generators")
            top3_revenue = product_stats.head(3)
            for _, row in top3_revenue.iterrows():
                pct = (row["Total Revenue"] / total_rev * 100) if total_rev else 0
                st.metric(row["Product Type"], f"${row['Total Revenue']:,.0f}", f"{pct:.1f}% of total")

            st.markdown("### Top 3 Volume Leaders")
            top3_volume = product_stats.sort_values("Sales Volume", ascending=False).head(3)
            for _, row in top3_volume.iterrows():
                pct = (row["Sales Volume"] / total_sales * 100) if total_sales else 0
                st.metric(row["Product Type"], f"{int(row['Sales Volume']):,} sales", f"{pct:.1f}% of volume")

        with colR:
            st.markdown("### Bottom 3 Revenue Generators")
            bottom3_revenue = product_stats.tail(3).sort_values("Total Revenue", ascending=True)
            for _, row in bottom3_revenue.iterrows():
                pct = (row["Total Revenue"] / total_rev * 100) if total_rev else 0
                st.metric(row["Product Type"], f"${row['Total Revenue']:,.0f}", f"{pct:.1f}% of total")

            st.markdown("### Bottom 3 Volume Leaders")
            bottom3_volume = product_stats.sort_values("Sales Volume", ascending=True).head(3)
            for _, row in bottom3_volume.iterrows():
                pct = (row["Sales Volume"] / total_sales * 100) if total_sales else 0
                st.metric(row["Product Type"], f"{int(row['Sales Volume']):,} sales", f"{pct:.1f}% of volume")

        # DYNAMIC BUSINESS TAKEAWAYS
        st.markdown("---")
        st.markdown('<div class="recommendation-box">', unsafe_allow_html=True)
        st.markdown("**Business Takeaways:**\n\n")
        
        if len(product_stats) > 0:
            top_product = product_stats.iloc[0]["Product Type"]
            top_product_pct = (product_stats.iloc[0]["Total Revenue"] / total_rev * 100) if total_rev else 0
            bottom_product = product_stats.iloc[-1]["Product Type"]
            bottom_product_pct = (product_stats.iloc[-1]["Total Revenue"] / total_rev * 100) if total_rev else 0
            
            takeaways = []
            
            # Pricing takeaway (DYNAMIC)
            if avg_disc < 2:
                takeaways.append(f"The {pricing_assessment} discount rate of {avg_disc:.2f}% combined with ${avg_txn:,.0f} average transaction validates premium positioning. Current pricing strategy is highly effective - customers perceive significant value and pay full price willingly.")
            else:
                takeaways.append(f"The discount rate of {avg_disc:.2f}% combined with ${avg_txn:,.0f} average transaction indicates opportunity to strengthen pricing discipline and move toward premium positioning.")
            
            # Concentration takeaway (DYNAMIC)
            if top_product_pct > 50:
                takeaways.append(f"Critical concentration in {top_product} ({top_product_pct:.1f}% of revenue) presents significant portfolio dependency risk requiring immediate diversification strategy.")
            elif top_product_pct > 40:
                takeaways.append(f"High concentration in {top_product} ({top_product_pct:.1f}% of revenue) presents both opportunity (proven market success) and risk (portfolio dependency) requiring strategic balance.")
            elif top_product_pct > 30:
                takeaways.append(f"Moderate concentration in {top_product} ({top_product_pct:.1f}% of revenue) shows clear market leader while maintaining reasonable portfolio diversification.")
            else:
                takeaways.append(f"Well-balanced portfolio with {top_product} leading at {top_product_pct:.1f}% of revenue, indicating healthy diversification across product categories.")
            
            # Bottom performer takeaway (DYNAMIC)
            if bottom_product_pct < 2:
                takeaways.append(f"Bottom performers like {bottom_product} ({bottom_product_pct:.1f}% of revenue) require strategic evaluation for continuation or elimination based on strategic importance and margin contribution.")
            else:
                takeaways.append(f"Lower performers including {bottom_product} ({bottom_product_pct:.1f}% of revenue) contribute meaningful revenue and may serve strategic roles in customer acquisition or portfolio completeness.")
            
            st.markdown("\n\n".join(takeaways), unsafe_allow_html=True)
        else:
            st.markdown("No product data available in current filter selection.")
        
        st.markdown("</div>", unsafe_allow_html=True)

    # =====================================================
    # TAB 2: INTERACTIVE TREEMAP (ORIGINAL COLORS RESTORED)
    # =====================================================
    with pm_tabs[1]:
        st.subheader("Interactive Product Hierarchy")

        st.markdown('<div class="insight-box">', unsafe_allow_html=True)
        st.markdown(
            """
            This interactive treemap shows revenue flow **Product Type → Grade → Species**.
            Rectangle size = revenue. Click to zoom.
            """,
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

        # ✅ RESTORED ORIGINAL COLOR SCHEME: RdYlBu_r
        fig_treemap = px.treemap(
            pm_df,
            path=["Product Type", "Grade", "Species"],
            values="Price (CAD)",
            title="Product Mix Revenue Hierarchy (Click to Zoom)",
            color="Price (CAD)",
            color_continuous_scale='RdYlBu_r',  # ✅ ORIGINAL COLOR
        )
        fig_treemap.update_layout(height=700, font=dict(size=14))
        fig_treemap.update_traces(
            textfont_size=12,
            marker=dict(line=dict(width=2, color="white"))
        )
        st.plotly_chart(fig_treemap, use_container_width=True, key=pkey("pm_tree"))

        st.markdown("---")
        if st.checkbox("View Complete Data Table", key=wkey("pm_treemap_data")):
            hierarchy_data = (
                pm_df.groupby(["Product Type", "Grade", "Species"])
                .agg(
                    **{
                        "Total Revenue": ("Price (CAD)", "sum"),
                        "Sales Count": ("Price (CAD)", "count"),
                        "Avg Price": ("Price (CAD)", "mean"),
                    }
                )
                .round(2)
                .sort_values("Total Revenue", ascending=False)
            )
            st.dataframe(hierarchy_data, use_container_width=True)

    # =====================================================
    # TAB 3: REVENUE ANALYSIS (REMOVED Individual Product Breakdowns)
    # =====================================================
    with pm_tabs[2]:
        st.subheader("Revenue Deep Dive")
        
        st.markdown("### Total Revenue by Grade")

        grade_revenue = (
            pm_df.groupby("Grade", as_index=False)
            .agg(Revenue=("Price (CAD)", "sum"), Sales=("Sale ID", "count"))
        )
        total_grade_rev = grade_revenue["Revenue"].sum()
        grade_revenue["Percentage"] = ((grade_revenue["Revenue"] / total_grade_rev) * 100).round(1) if total_grade_rev else 0
        grade_revenue = grade_revenue.sort_values("Revenue", ascending=False)

        # ✅ ORIGINAL COLORS: Viridis
        fig_grade = px.bar(
            grade_revenue,
            x="Grade",
            y="Revenue",
            title="Total Revenue by Grade",
            text="Percentage",
            color="Revenue",
            color_continuous_scale='Viridis',  # ✅ ORIGINAL COLOR
        )
        fig_grade.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig_grade.update_layout(height=500, xaxis_title="Grade", yaxis_title="Revenue (CAD)", showlegend=False)
        st.plotly_chart(fig_grade, use_container_width=True, key=pkey("pm_grade"))

        # DYNAMIC INSIGHT
        if not grade_revenue.empty:
            top = grade_revenue.iloc[0]
            st.markdown('<div class="insight-box">', unsafe_allow_html=True)
            st.markdown(
                f"""
                Grade {top['Grade']} generates **${top['Revenue']:,.0f}** ({top['Percentage']:.1f}% of total), 
                leading revenue contribution. This grade represents the optimal balance of quality, pricing, and market demand, 
                making it the strategic sweet spot for product development and inventory focus.
                """,
                unsafe_allow_html=True,
            )
            st.markdown("</div>", unsafe_allow_html=True)

    # =====================================================
    # TAB 4: PRICING ANALYSIS (REMOVED Price Distribution by Grade)
    # =====================================================
    with pm_tabs[3]:
        st.subheader("Pricing Structure Analysis")
        
        st.markdown("### Price Distribution by Product Type")

        st.markdown('<div class="insight-box">', unsafe_allow_html=True)
        st.markdown(
            """
            Violin plots show the full price distribution for each product type.
            Wider areas = more transactions at that price.
            """,
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

        fig_violin = go.Figure()
        for pt in sorted(pm_df["Product Type"].dropna().unique()):
            data = pm_df.loc[pm_df["Product Type"] == pt, "Price (CAD)"]
            if len(data) > 0:
                fig_violin.add_trace(go.Violin(
                    y=data,
                    name=str(pt),
                    box_visible=True,
                    meanline_visible=True,
                    opacity=0.6,
                ))
        fig_violin.update_layout(
            title="Price Distribution Across Product Types (Violin Plot)",
            yaxis_title="Price (CAD)",
            xaxis_title="Product Type",
            height=600,
            showlegend=False,
            violinmode="group",
        )
        st.plotly_chart(fig_violin, use_container_width=True, key=pkey("pm_violin_type"))

        st.markdown("---")
        if st.checkbox("View Detailed Price Statistics Table", key=wkey("pm_price_stats")):
            price_stats = (
                pm_df.groupby(["Product Type", "Grade"])["Price (CAD)"]
                .agg(["count", "mean", "median", "std", "min", "max"])
                .round(2)
                .rename(columns={"count": "Sales", "mean": "Mean", "median": "Median", "std": "Std Dev", "min": "Min", "max": "Max"})
                .sort_values("Mean", ascending=False)
            )
            st.dataframe(price_stats, use_container_width=True)

    # =====================================================
    # TAB 5: EFFICIENCY METRICS (ORIGINAL COLORS)
    # =====================================================
    with pm_tabs[4]:
        st.subheader("Sales Efficiency Analysis")
        
        st.markdown("### Sales Efficiency: Volume vs Revenue")

        st.markdown('<div class="insight-box">', unsafe_allow_html=True)
        st.markdown(
            """
            Efficiency = revenue per sale. Bubble chart compares volume vs revenue,
            bubble size = total revenue contribution.
            """,
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

        efficiency = (
            pm_df.groupby("Product Type")
            .agg(**{"Sales": ("Sale ID", "count"), "Total Revenue": ("Price (CAD)", "sum"), "Avg Price": ("Price (CAD)", "mean")})
            .reset_index()
        )
        efficiency["Efficiency"] = efficiency["Total Revenue"] / efficiency["Sales"].replace(0, np.nan)
        efficiency["Efficiency"] = efficiency["Efficiency"].fillna(0)
        efficiency["Revenue %"] = ((efficiency["Total Revenue"] / efficiency["Total Revenue"].sum()) * 100).round(1) if efficiency["Total Revenue"].sum() else 0

        # ✅ ORIGINAL COLORS: RdYlGn
        fig_eff = px.scatter(
            efficiency,
            x="Sales",
            y="Total Revenue",
            size="Total Revenue",
            color="Efficiency",
            hover_name="Product Type",
            title="Product Efficiency Matrix: Sales Volume vs Total Revenue",
            labels={"Sales": "Number of Sales", "Total Revenue": "Total Revenue (CAD)", "Efficiency": "Revenue per Sale"},
            color_continuous_scale='RdYlGn',  # ✅ ORIGINAL COLOR
            size_max=80,
        )
        fig_eff.update_layout(height=600, xaxis_title="Sales Volume", yaxis_title="Total Revenue (CAD)")

        if not efficiency.empty:
            med_sales = efficiency["Sales"].median()
            med_rev = efficiency["Total Revenue"].median()
            fig_eff.add_hline(y=med_rev, line_dash="dash", line_color="gray", opacity=0.5)
            fig_eff.add_vline(x=med_sales, line_dash="dash", line_color="gray", opacity=0.5)

        st.plotly_chart(fig_eff, use_container_width=True, key=pkey("pm_eff_scatter"))

        st.markdown('<div class="insight-box">', unsafe_allow_html=True)
        st.markdown(
            """
            **Quadrant Interpretation:**
            
            - **Top Right (Stars):** High volume + High revenue - Proven winners, scale aggressively
            - **Top Left (Premium):** Low volume + High revenue - Efficient, high-margin products
            - **Bottom Right (Volume Play):** High volume + Low revenue - Evaluate strategic value
            - **Bottom Left (Underperformers):** Low volume + Low revenue - Consider elimination
            """,
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("---")
        st.subheader("Efficiency Rankings")

        eff_sorted = efficiency.sort_values("Efficiency", ascending=False)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("### Top 5 Most Efficient (Revenue per Sale)")
            for _, row in eff_sorted.head(5).iterrows():
                st.metric(row["Product Type"], f"${row['Efficiency']:,.2f}/sale", f"{row['Revenue %']:.1f}% of revenue")

        with c2:
            st.markdown("### Bottom 5 Least Efficient")
            for _, row in eff_sorted.tail(5).iterrows():
                st.metric(row["Product Type"], f"${row['Efficiency']:,.2f}/sale", f"{row['Revenue %']:.1f}% of revenue")

        st.markdown("---")
        if st.checkbox("View Efficiency Data Table", key=wkey("pm_eff_table")):
            show = eff_sorted[["Product Type", "Sales", "Total Revenue", "Efficiency", "Revenue %"]].copy()
            st.dataframe(show, use_container_width=True)

    # =====================================================
    # TAB 6: STRATEGIC INSIGHTS (DYNAMIC)
    # =====================================================
    with pm_tabs[5]:
        st.subheader("Strategic Recommendations")
        
        # Calculate metrics from FILTERED data
        total_revenue = pm_df["Price (CAD)"].sum()
        total_sales = len(pm_df)
        avg_transaction = pm_df["Price (CAD)"].mean()
        avg_disc = (pm_df["Discount (CAD)"].mean() / pm_df["Price (CAD)"].mean() * 100) if pm_df["Price (CAD)"].mean() > 0 else 0
        
        # Find dominant product
        product_revenue = pm_df.groupby("Product Type")["Price (CAD)"].sum().sort_values(ascending=False)
        top_product = product_revenue.index[0] if len(product_revenue) > 0 else "N/A"
        top_product_pct = (product_revenue.iloc[0] / total_revenue * 100) if len(product_revenue) > 0 and total_revenue > 0 else 0
        top3_pct = (product_revenue.head(3).sum() / total_revenue * 100) if len(product_revenue) >= 3 and total_revenue > 0 else 0
        
        # Find dominant grade
        if "Grade" in pm_df.columns:
            grade_revenue = pm_df.groupby("Grade")["Price (CAD)"].sum().sort_values(ascending=False)
            top_grade = grade_revenue.index[0] if len(grade_revenue) > 0 else "N/A"
            top_grade_pct = (grade_revenue.iloc[0] / total_revenue * 100) if len(grade_revenue) > 0 and total_revenue > 0 else 0
        else:
            top_grade = "N/A"
            top_grade_pct = 0
        
        # Calculate efficiency
        efficiency_calc = pm_df.groupby("Product Type").agg({
            "Sale ID": "count",
            "Price (CAD)": ["sum", "mean"]
        })
        efficiency_calc.columns = ["Sales", "Total Revenue", "Avg Price"]
        efficiency_calc["Efficiency"] = efficiency_calc["Total Revenue"] / efficiency_calc["Sales"]
        efficiency_calc = efficiency_calc.sort_values("Efficiency", ascending=False)
        
        most_efficient = efficiency_calc.index[0] if len(efficiency_calc) > 0 else "N/A"
        most_efficient_value = efficiency_calc.iloc[0]["Efficiency"] if len(efficiency_calc) > 0 else 0
        least_efficient = efficiency_calc.index[-1] if len(efficiency_calc) > 0 else "N/A"
        least_efficient_value = efficiency_calc.iloc[-1]["Efficiency"] if len(efficiency_calc) > 0 else 0
        efficiency_gap = (most_efficient_value / least_efficient_value) if least_efficient_value > 0 else 0

        with st.expander("Immediate Actions (0-3 Months)", expanded=False):
            st.markdown('<div class="recommendation-box">', unsafe_allow_html=True)
            
            # Priority 1: Pricing (DYNAMIC)
            if avg_disc < 2:
                st.markdown(f"""
                **Priority 1: Protect Pricing Power**
                - Current discount rate of {avg_disc:.2f}% is exceptional - maintain this discipline
                - Document value propositions that justify premium pricing
                - Train sales team on value-based selling vs. discount negotiations
                - **Expected Impact:** Preserve margin advantage over competitors
                """)
            else:
                st.markdown(f"""
                **Priority 1: Improve Pricing Discipline**
                - Current discount rate of {avg_disc:.2f}% is higher than optimal
                - Reduce discounting by implementing value-based pricing strategies
                - Train sales team to resist discount requests
                - **Target:** Reduce to <2% within 90 days
                - **Expected Impact:** Immediate margin improvement of {avg_disc - 1.5:.1f} percentage points
                """)
            
            # Priority 2: Supply Chain (DYNAMIC)
            if top_product_pct > 40:
                st.markdown(f"""
                **Priority 2: Secure Supply Chain**
                - {top_product} represents {top_product_pct:.1f}% of revenue - high dependency risk
                - Establish backup suppliers for {top_product} immediately
                - Increase safety stock for high-efficiency products
                - Negotiate volume commitments with key suppliers
                - **Expected Impact:** Reduce stockout risk by 50-60%
                """)
            else:
                st.markdown(f"""
                **Priority 2: Optimize Supply Chain**
                - Portfolio is well-diversified (top product: {top_product_pct:.1f}%)
                - Focus on optimizing fulfillment efficiency
                - Negotiate better terms with suppliers based on volume
                - **Expected Impact:** 5-10% cost reduction in procurement
                """)
            
            # Priority 3: Quick Wins (DYNAMIC)
            st.markdown(f"""
            **Priority 3: Quick Wins**
            - Bundle high-efficiency ({most_efficient}) with lower performers to move inventory
            - Test 10-15% price increases on top products (they have pricing power)
            - Focus sales effort on {most_efficient} (${most_efficient_value:,.0f}/sale vs ${least_efficient_value:,.0f}/sale for {least_efficient})
            - **Expected Impact:** 8-12% revenue increase with same sales effort
            """)
            
            st.markdown("</div>", unsafe_allow_html=True)

        with st.expander("Short-Term Strategy (3-6 Months)", expanded=False):
            st.markdown('<div class="recommendation-box">', unsafe_allow_html=True)
            
            # Initiative 1: Portfolio Rebalancing (DYNAMIC)
            if top_product_pct > 50:
                target_pct = 40
            elif top_product_pct > 40:
                target_pct = 35
            else:
                target_pct = max(30, top_product_pct - 5)
            
            st.markdown(f"""
            **Initiative 1: Portfolio Rebalancing**
            - **Current State:** {top_product} at {top_product_pct:.1f}%, top 3 at {top3_pct:.1f}%
            - **Target:** Reduce {top_product} to <{target_pct}% within 6 months
            - **How:** Don't reduce {top_product} sales - grow other categories around it
            - Invest in 2-3 underutilized categories with growth potential
            - Develop new SKUs in proven {top_grade} grade (top revenue generator)
            - **Expected Impact:** 15-20% revenue growth from diversification
            """)
            
            # Initiative 2: Sales Force Realignment (DYNAMIC)
            st.markdown(f"""
            **Initiative 2: Sales Force Realignment**
            - **Problem:** {efficiency_gap:.0f}x efficiency gap between {most_efficient} (${most_efficient_value:,.0f}/sale) and {least_efficient} (${least_efficient_value:,.0f}/sale)
            - **Solution:** Compensate based on profit contribution, not volume
            - Create incentives for high-efficiency product sales
            - Reduce effort on low-efficiency products (automate/eliminate)
            - **Expected Impact:** 25% improvement in team productivity
            """)
            
            # Initiative 3: Customer Segmentation
            customer_count = pm_df["Customer Name"].nunique() if "Customer Name" in pm_df.columns else total_sales
            st.markdown(f"""
            **Initiative 3: Customer Segmentation**
            - Analyze {customer_count:,} customers to identify Premium vs. Value vs. Mixed buyers
            - Develop targeted strategies for each segment
            - Create VIP program for top 20% of customers
            - **Expected Impact:** 20-30% increase in customer lifetime value
            """)
            
            st.markdown("</div>", unsafe_allow_html=True)

        with st.expander("Long-Term Vision (6-12 Months)", expanded=False):
            st.markdown('<div class="recommendation-box">', unsafe_allow_html=True)
            
            # Theme 1: Diversification (DYNAMIC)
            st.markdown(f"""
            **Theme 1: Sustainable Diversification**
            - **Target Portfolio Mix (12 months):**
              * Primary category ({top_product}): 35-40% (from current {top_product_pct:.1f}%)
              * Secondary categories: 40-45% distributed across 3-4 products
              * Emerging/experimental: 15-20% new product development
            
            - **Development Roadmap:**
              * Months 1-3: Research and identify growth categories
              * Months 4-6: Pilot products in selected categories
              * Months 7-9: Scale successful pilots
              * Months 10-12: Portfolio rebalancing complete
            
            - **Investment:** 15-20% of current marketing budget redirected
            - **Expected Outcome:** More resilient revenue base, no single product >40%
            """)
            
            # Theme 2: Premium Brand Evolution (DYNAMIC)
            if top_grade != "N/A":
                st.markdown(f"""
                **Theme 2: Premium Brand Evolution**
                - **Three-tier architecture:**
                  * Flagship: Top grade products, ultra-premium positioning
                  * Core: {top_grade} grade (proven sweet spot at {top_grade_pct:.1f}% of revenue)
                  * Access: Entry-level premium to attract new customers
                
                - **Pricing Strategy:**
                  * Maintain current {avg_disc:.2f}% discount discipline
                  * Implement dynamic pricing for scarcity items
                  * Value-based pricing across all tiers
                
                - **Expected Outcome:** 40% increase in customer lifetime value
                """)
            else:
                st.markdown(f"""
                **Theme 2: Premium Brand Evolution**
                - Develop clear product tiers based on price points
                - Maintain current {avg_disc:.2f}% discount discipline
                - Implement value-based pricing strategies
                - **Expected Outcome:** 30-40% increase in customer lifetime value
                """)
            
            # Theme 3: Operational Excellence (DYNAMIC)
            tech_investment = int(total_revenue * 0.01)
            tech_investment_rounded = round(tech_investment / 10000) * 10000
            
            st.markdown(f"""
            **Theme 3: Operational Excellence**
            - **Technology Investments:**
              * CRM system to track customer preferences (~${tech_investment_rounded/3:,.0f})
              * Inventory management optimized by product efficiency (~${tech_investment_rounded/3:,.0f})
              * Business intelligence dashboard for real-time decisions (~${tech_investment_rounded/3:,.0f})
            
            - **Process Improvements:**
              * Tier-based fulfillment (white-glove for {most_efficient}, automated for {least_efficient})
              * Focus resources on high-efficiency products
              * Streamline operations for volume products
            
            - **Investment:** ~${tech_investment_rounded:,.0f} in technology
            - **Expected Outcome:** 25-30% operational efficiency improvement
            """)
            
            st.markdown("</div>", unsafe_allow_html=True)

        # DYNAMIC EXECUTIVE SUMMARY
        st.markdown("---")
        st.markdown('<div class="recommendation-box">', unsafe_allow_html=True)
        
        # Calculate projections (DYNAMIC)
        projected_growth_low = total_revenue * 1.25
        projected_growth_high = total_revenue * 1.30
        total_investment = tech_investment_rounded + (total_revenue * 0.02)
        total_investment_rounded = round(total_investment / 10000) * 10000
        
        # Dynamic concentration message
        if top_product_pct > 50:
            concentration_msg = f"CRITICAL concentration ({top_product_pct:.1f}% in {top_product}) presents significant risk"
        elif top_product_pct > 40:
            concentration_msg = f"High concentration ({top_product_pct:.1f}% in {top_product}) presents both opportunity and risk"
        else:
            concentration_msg = f"Balanced portfolio with largest category at {top_product_pct:.1f}%"
        
        st.markdown(
            f"""
            ### Executive Summary
            
            **Current State:** ${total_revenue:,.0f} revenue from {total_sales:,} transactions with {avg_disc:.2f}% avg discount. 
            {concentration_msg}.
            
            **Key Strengths:**
            1. {"Exceptional" if avg_disc < 2 else "Good"} pricing discipline ({avg_disc:.2f}% discount rate)
            2. Strong performers: {top_product} leads with {top_product_pct:.1f}% of revenue
            3. Proven efficiency: {most_efficient} at ${most_efficient_value:,.0f} per sale
            4. Average transaction of ${avg_transaction:,.2f} indicates {"premium" if avg_transaction > 1000 else "mid-tier"} market positioning
            
            **Critical Risks:**
            1. {"Portfolio concentration risk" if top_product_pct > 40 else "Minimal concentration risk"}
            2. {efficiency_gap:.0f}x efficiency gap between best and worst performers
            3. {"Pricing pressure" if avg_disc > 3 else "Potential for margin erosion if discounting increases"}
            4. {"Supply chain dependency on " + top_product if top_product_pct > 40 else "Resource allocation inefficiency"}
            
            **Strategic Priorities:**
            1. **Immediate:** {"Protect" if avg_disc < 2 else "Improve"} pricing power and secure supply chain
            2. **Short-term:** {"Rebalance portfolio and" if top_product_pct > 40 else "Optimize"} align sales incentives  
            3. **Long-term:** Sustainable diversification and operational excellence
            
            **Expected Outcomes (12 Months):**
            - Revenue growth: 25-30% (${total_revenue:,.0f} → ${projected_growth_low:,.0f}-${projected_growth_high:,.0f})
            - Margin improvement: {"Maintain current excellence" if avg_disc < 2 else f"Improve by {avg_disc - 1.5:.1f}pp"}
            - Concentration: {"Reduce to <40%" if top_product_pct > 40 else "Maintain current balance"}
            - Operational efficiency: +30%
            
            **Investment Required:** ${total_investment_rounded:,.0f} | **Expected ROI:** 300-400% over 2 years
            """,
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------

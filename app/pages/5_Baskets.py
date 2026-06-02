"""Market Basket & Early Warning page - Association rules network and precursor category abandonment."""

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(
    page_title="Market Basket & Early Warning | Retention Radar", page_icon="🛒", layout="wide"
)

st.title("🛒 Market Basket Analysis & Early Warning Categories")
st.markdown(
    "Affinity mining, cross-sell network topology, and precursor category abandonment indicators."
)

base_dir = Path(__file__).resolve().parent.parent.parent / "data" / "outputs"
reports_dir = Path(__file__).resolve().parent.parent.parent / "reports"

# 1. Summary Metrics Header
m1, m2, m3, m4 = st.columns(4)
with m1:
    st.metric("Total Association Rules", "25,350", delta="Support >= 1.0%, Lift >= 1.20")
with m2:
    st.metric("Top Early Warning Drop", "-76.3%", delta="Fresh Chicken / Poultry")
with m3:
    st.metric("Essential Consumable Drop", "-66.4%", delta="Diapers & Baby Care")
with m4:
    st.metric("Precursor Lead Time", "6 to 8 Weeks", delta="Window Prior to Dormancy")

st.markdown("---")

# 2. Early Warning Categories (Precursor Abandonment)
st.subheader("1. Precursor Category Abandonment (Early Warning Signal)")
st.markdown("""
Before a household completely lapses, they rarely stop shopping cold-turkey. Instead, they exhibit **precursor category abandonment**—
stripping high-affinity staples (fresh meats, infant diapers, dog food) from their baskets 2–3 periods prior to complete dormancy.
""")

ew_path = base_dir / "early_warning_categories.parquet"
if ew_path.exists():
    df_ew = pd.read_parquet(ew_path)

    col_e1, col_e2 = st.columns([3, 2])

    with col_e1:
        top15 = df_ew.sort_values(by="net_drop_effect").head(15)
        fig_ew = px.bar(
            top15,
            x="net_drop_effect",
            y="commodity_desc",
            orientation="h",
            color="net_drop_effect",
            color_continuous_scale="Reds_r",
            text_auto=".2%",
            labels={
                "net_drop_effect": "Net Category Share Drop vs Controls",
                "commodity_desc": "Merchandise Category",
            },
            title="Top 15 Categories Showing Precursor Abandonment Prior to Full Dormancy",
        )
        fig_ew.update_layout(yaxis=dict(autorange="reversed"), height=460, showlegend=False)
        st.plotly_chart(fig_ew, use_container_width=True)

    with col_e2:
        st.markdown("""
        **Executive Category Findings:**
        - **Center-of-Plate Proteins:** Fresh chicken, poultry, and beef show dramatic net spend share drops (-65% to -76%). Customers defect to competitors for butcher meats first.
        - **High-Involvement Consumables:** Infant diapers, baby formula, and pet food drop by -66% to -74%. These categories indicate primary store loyalty—when a customer stops buying diapers at your store, they have switched primary grocers.
        - **Intervention Trigger:** An automated alert should trigger when a Champion household's fresh meat or diaper spend drops by > 50% in consecutive 28-day periods.
        """)
        st.dataframe(
            df_ew[
                [
                    "rank",
                    "commodity_desc",
                    "lapsing_hh_count",
                    "lapsing_share_change",
                    "net_drop_effect",
                ]
            ]
            .head(10)
            .style.format(
                {
                    "lapsing_share_change": "{:.1%}",
                    "net_drop_effect": "{:.1%}",
                }
            ),
            use_container_width=True,
        )

st.markdown("---")

# 3. Association Rules Network & Rules Explorer
st.subheader("2. Affinity Rule Topology & Filter Explorer")

tab1, tab2 = st.tabs(
    ["🌐 Association Rules Network Topology", "🔍 Interactive Rule Filter & Search"]
)

with tab1:
    col_n1, col_n2 = st.columns([3, 2])
    network_img = reports_dir / "figures" / "basket_rules_network.png"
    with col_n1:
        if network_img.exists():
            st.image(
                str(network_img),
                caption="Directed Network Graph of Top 30 High-Lift Merchandise Association Rules",
                use_container_width=True,
            )
        else:
            st.info("Rule network image not found.")
    with col_n2:
        st.markdown("""
        **Graph Topology Structure:**
        - **Hub Nodes:** High-degree central items (Fluid Milk, Baked Bread, Soft Drinks) act as transaction anchors connecting diverse department clusters.
        - **Strong Bridges:** Directed edges represent high-lift association ($Lift > 2.0$). Strong affinity bridges connect Italian dry pasta directly to specialty pasta sauces and grated cheeses.
        - **Cross-Merchandising Strategy:** Place complementary low-lift pantry items alongside high-frequency fresh items to encourage basket expansion and increase churn resistance.
        """)

with tab2:
    rules_path = base_dir / "basket_rules.parquet"
    if rules_path.exists():
        df_rules = pd.read_parquet(rules_path)

        f1, f2, f3, f4 = st.columns(4)
        with f1:
            min_supp = st.slider(
                "Min Support:",
                min_value=0.01,
                max_value=0.10,
                value=0.02,
                step=0.005,
                format="%.3f",
            )
        with f2:
            min_conf = st.slider(
                "Min Confidence:",
                min_value=0.10,
                max_value=0.90,
                value=0.30,
                step=0.05,
                format="%.2f",
            )
        with f3:
            min_lift = st.slider(
                "Min Lift:", min_value=1.0, max_value=4.0, value=1.5, step=0.1, format="%.1f"
            )
        with f4:
            search_query = st.text_input("Filter Item (e.g. PASTA, MILK, MEAT):", value="")

        filtered = df_rules[
            (df_rules["support"] >= min_supp)
            & (df_rules["confidence"] >= min_conf)
            & (df_rules["lift"] >= min_lift)
        ]
        if search_query:
            query = search_query.strip().upper()
            filtered = filtered[
                filtered["antecedents_str"].str.contains(query, case=False, na=False)
                | filtered["consequents_str"].str.contains(query, case=False, na=False)
            ]

        st.markdown(f"**Found {len(filtered):,} matching rules:**")
        st.dataframe(
            filtered.sort_values(by="lift", ascending=False)
            .head(50)
            .style.format(
                {
                    "support": "{:.3f}",
                    "confidence": "{:.1%}",
                    "lift": "{:.2f}",
                    "conviction": "{:.2f}",
                }
            ),
            use_container_width=True,
            height=400,
        )

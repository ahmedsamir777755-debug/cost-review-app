import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO

st.set_page_config(page_title="Monthly Cost Review", layout="wide")

st.title("Monthly Cost Review & Anomaly Detection")

uploaded_file = st.file_uploader("Upload Excel file", type=["xlsx"])

def parse_amount(x):
    if pd.isna(x):
        return 0.0
    x = str(x).replace(",", "")
    if x.startswith("(") and x.endswith(")"):
        return -float(x.strip("()"))
    return float(x)

if uploaded_file:
    df = pd.read_excel(uploaded_file)

    df.columns = df.columns.str.strip()

    month_cols = [c for c in df.columns if "/" in c and c[:4].isdigit()]
    df = df.drop(columns=[c for c in df.columns if c.lower() == "grand total"], errors="ignore")

    for c in month_cols:
        df[c] = df[c].apply(parse_amount)

    long_df = df.melt(
        id_vars=["GL name", "Department"],
        value_vars=month_cols,
        var_name="Month",
        value_name="Amount"
    )

    long_df = long_df.sort_values(["GL name", "Department", "Month"])

    results = []

    for (gl, dept), g in long_df.groupby(["GL name", "Department"]):
        g = g.reset_index(drop=True)

        for i in range(1, len(g)):
            hist = g.loc[:i-1, "Amount"]
            current = g.loc[i, "Amount"]

            hist_avg = hist.mean()
            hist_std = hist.std(ddof=0)
            z = (current - hist_avg) / hist_std if hist_std != 0 else 0
            mom = current - g.loc[i-1, "Amount"]

            spike = current > hist_avg + 2.5 * hist_std if hist_std != 0 else False
            drop = current < hist_avg - 2.5 * hist_std if hist_std != 0 else False
            new_cost = current > 0 and hist.tail(2).sum() == 0
            negative = current < 0
            outlier = abs(z) >= 2.5

            comment = "Normal cost pattern."
            if spike:
                comment = "Significant increase compared to historical average."
            elif drop:
                comment = "Significant decrease compared to historical average."
            elif new_cost:
                comment = "New cost appeared after inactive periods."
            elif negative:
                comment = "Negative posting detected, review adjustment."
            elif outlier:
                comment = "Historical outlier detected."

            results.append({
                "GL name": gl,
                "Department": dept,
                "Month": g.loc[i, "Month"],
                "Amount": current,
                "Historical Avg": hist_avg,
                "Z-Score": z,
                "MoM Var": mom,
                "Spike": spike,
                "Drop": drop,
                "New Cost": new_cost,
                "Negative": negative,
                "Outlier": outlier,
                "Comment_EN": comment
            })

    result_df = pd.DataFrame(results)

    tab1, tab2, tab3 = st.tabs(["Normalized Data", "All Months Comparison", "Anomaly Log"])

    with tab1:
        st.dataframe(long_df)

    with tab2:
        st.dataframe(result_df)

    with tab3:
        st.dataframe(result_df[result_df[["Spike", "Drop", "New Cost", "Negative", "Outlier"]].any(axis=1)])

    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        long_df.to_excel(writer, sheet_name="Normalized", index=False)
        result_df.to_excel(writer, sheet_name="All_Months_Comparison", index=False)
        result_df[result_df[["Spike", "Drop", "New Cost", "Negative", "Outlier"]].any(axis=1)] \
            .to_excel(writer, sheet_name="Anomaly_Log", index=False)

    st.download_button(
        "Download Excel Output",
        data=output.getvalue(),
        file_name="cost_review_output.xlsx"
    )

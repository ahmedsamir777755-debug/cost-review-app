import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO

st.set_page_config(page_title="Monthly Cost Review Dashboard", layout="wide")

st.title("Monthly Cost Review & Department Variance Dashboard")

# Thresholds as per your request
MINOR_DEV = 0.20   # 20% minor deviation
MAJOR_DEV = 0.50   # 50% major deviation

uploaded_file = st.file_uploader("Upload Excel file (.xlsx)", type=["xlsx"])

def normalize_cols(cols):
    return [str(c).strip() for c in cols]

def parse_amount(x):
    if pd.isna(x):
        return 0.0
    s = str(x).strip()
    if s == "":
        return 0.0
    s = s.replace(",", "")
    # (2,426) -> -2426
    if s.startswith("(") and s.endswith(")"):
        s = "-" + s.strip("()")
    try:
        return float(s)
    except:
        return 0.0

def detect_month_cols(df):
    # Accept columns like YYYY/MM
    cols = df.columns
    month_cols = []
    for c in cols:
        s = str(c).strip()
        if len(s) == 7 and s[:4].isdigit() and s[4] == "/" and s[5:7].isdigit():
            month_cols.append(s)
    return month_cols

def find_col(df, candidates):
    # Find column that matches any candidate (case-insensitive, stripped)
    lower_map = {c.lower().strip(): c for c in df.columns}
    for cand in candidates:
        key = cand.lower().strip()
        if key in lower_map:
            return lower_map[key]
    # fallback: contains keyword
    for c in df.columns:
        cl = c.lower()
        for kw in candidates:
            if kw.lower() in cl:
                return c
    return None

if not uploaded_file:
    st.info("Upload an Excel file to start.")
    st.stop()

# Read excel
df = pd.read_excel(uploaded_file)
df.columns = normalize_cols(df.columns)

# Drop Grand Total if exists (any casing)
for c in list(df.columns):
    if str(c).strip().lower() == "grand total":
        df = df.drop(columns=[c])

# Detect required columns flexibly
gl_col = find_col(df, ["GL name", "Gl name", "GL Name", "Account", "Account Name", "G/L", "G/L Account", "GL"])
dept_col = find_col(df, ["Department", "Dept", "Cost Center", "CostCentre", "Cost center"])

if gl_col is None or dept_col is None:
    st.error("Could not detect required columns. Please ensure the file contains GL and Department columns.")
    st.write("Detected columns:", list(df.columns))
    st.stop()

month_cols = detect_month_cols(df)
if not month_cols:
    st.error("No month columns detected. Month columns must be in YYYY/MM format (e.g., 2026/01).")
    st.write("Detected columns:", list(df.columns))
    st.stop()

# Parse amounts
for c in month_cols:
    df[c] = df[c].apply(parse_amount)

# Build normalized table (long format)
long_df = df.melt(
    id_vars=[gl_col, dept_col],
    value_vars=month_cols,
    var_name="Month",
    value_name="Amount"
).rename(columns={gl_col: "GL name", dept_col: "Department"})

# Sort month properly (YYYY/MM sorts lexicographically fine)
long_df = long_df.sort_values(["GL name", "Department", "Month"]).reset_index(drop=True)

# -------- 1) Historical comparison for EACH month vs ALL previous months --------
rows = []
for (gl, dept), g in long_df.groupby(["GL name", "Department"], sort=False):
    g = g.sort_values("Month").reset_index(drop=True)
    for i in range(len(g)):
        month = g.loc[i, "Month"]
        amt = g.loc[i, "Amount"]

        # previous month values
        prev_amt = g.loc[i-1, "Amount"] if i > 0 else np.nan

        # history is all previous months
        if i == 0:
            hist_avg = np.nan
            dev_pct = np.nan
        else:
            hist = g.loc[:i-1, "Amount"]
            hist_avg = float(hist.mean())

            if hist_avg == 0:
                dev_pct = np.inf if amt > 0 else 0.0
            else:
                dev_pct = float((amt - hist_avg) / hist_avg)

        # Severity based on deviation vs all history
        if i == 0:
            severity = "N/A"
            hist_flag = "N/A"
        else:
            if np.isinf(dev_pct):
                severity = "High"
                hist_flag = "New cost vs historical zero"
            else:
                abs_dev = abs(dev_pct)
                if abs_dev >= MAJOR_DEV:
                    severity = "High"
                    hist_flag = "Major deviation vs history"
                elif abs_dev >= MINOR_DEV:
                    severity = "Medium"
                    hist_flag = "Minor deviation vs history"
                else:
                    severity = "Normal"
                    hist_flag = "Within normal range"

        rows.append({
            "GL name": gl,
            "Department": dept,
            "Month": month,
            "Amount": amt,
            "Prev Amount": prev_amt,
            "MoM Var": (amt - prev_amt) if pd.notna(prev_amt) else np.nan,
            "MoM %": ((amt - prev_amt) / prev_amt) if (pd.notna(prev_amt) and prev_amt != 0) else (np.inf if (pd.notna(prev_amt) and prev_amt == 0 and amt > 0) else np.nan),
            "Historical Avg (All Prev Months)": hist_avg,
            "Deviation % vs History": dev_pct,
            "History Flag": hist_flag,
            "Severity": severity
        })

all_months_cmp = pd.DataFrame(rows)

# -------- 2) Latest month MoM increases + Department comments --------
latest_month = max(month_cols)
prev_month = sorted(month_cols)[-2] if len(month_cols) >= 2 else None

latest_df = all_months_cmp[all_months_cmp["Month"] == latest_month].copy()
if prev_month is not None:
    prev_df = all_months_cmp[all_months_cmp["Month"] == prev_month][["GL name", "Department", "Amount"]].rename(columns={"Amount":"PrevMonthAmount"})
    latest_df = latest_df.merge(prev_df, on=["GL name","Department"], how="left")
    latest_df["MoM Var (Latest)"] = latest_df["Amount"] - latest_df["PrevMonthAmount"]
    latest_df["MoM % (Latest)"] = np.where(
        latest_df["PrevMonthAmount"].fillna(0) == 0,
        np.where(latest_df["Amount"] > 0, np.inf, np.nan),
        (latest_df["Amount"] - latest_df["PrevMonthAmount"]) / latest_df["PrevMonthAmount"]
    )
else:
    latest_df["PrevMonthAmount"] = np.nan
    latest_df["MoM Var (Latest)"] = np.nan
    latest_df["MoM % (Latest)"] = np.nan

# Keep only increases for MoM comment requirement
latest_increases = latest_df[(latest_df["MoM Var (Latest)"].fillna(0) > 0)].copy()
latest_increases["MoM % (Latest)"] = latest_increases["MoM % (Latest)"].replace([np.inf, -np.inf], np.nan)

# Build department-level comments (English)
dept_comments = []
if prev_month is not None:
    for dept, g in latest_increases.groupby("Department"):
        # take top 5 increases by absolute value
        g2 = g.sort_values("MoM Var (Latest)", ascending=False).head(5)

        parts = []
        for _, r in g2.iterrows():
            gl = r["GL name"]
            prevv = r["PrevMonthAmount"]
            curr = r["Amount"]
            mom_pct = r["MoM % (Latest)"]
            if pd.isna(prevv):
                parts.append(f"{gl} increased in {latest_month} (no prior month value).")
            else:
                if prevv == 0:
                    parts.append(f"{gl} increased in {latest_month} from 0 to {curr:,.0f}.")
                else:
                    pct_txt = f"{(mom_pct*100):.0f}%" if pd.notna(mom_pct) else "N/A"
                    parts.append(f"{gl} increased in {latest_month} from {prevv:,.0f} to {curr:,.0f} ({pct_txt}).")

        if parts:
            comment = f"{dept}: " + " ".join(parts)
        else:
            comment = f"{dept}: No increases detected in {latest_month}."
        dept_comments.append({"Department": dept, "Latest Month": latest_month, "Comment_EN": comment})

dept_comments_df = pd.DataFrame(dept_comments).sort_values("Department") if dept_comments else pd.DataFrame(columns=["Department","Latest Month","Comment_EN"])

# -------- Dashboard aggregations --------
# Department totals per month
dept_month_totals = long_df.groupby(["Department","Month"], as_index=False)["Amount"].sum()

# Latest month department totals & anomaly counts
latest_sev = latest_df[["Department","Severity"]].copy()
sev_counts = latest_sev.groupby(["Department","Severity"]).size().unstack(fill_value=0).reset_index()

latest_dept_totals = dept_month_totals[dept_month_totals["Month"] == latest_month].rename(columns={"Amount":"Total Cost (Latest)"})
dashboard_dept = latest_dept_totals.merge(sev_counts, on="Department", how="left").fillna(0)

# -------- UI --------
tab_dash, tab_norm, tab_cmp, tab_anom, tab_dept = st.tabs(["Dashboard", "Normalized Data", "All Months Comparison", "Anomaly Log", "Dept Comments (Latest MoM)"])

with tab_dash:
    st.subheader("Dashboard")

    col1, col2, col3, col4 = st.columns(4)
    total_latest = dept_month_totals[dept_month_totals["Month"]==latest_month]["Amount"].sum()
    total_prev = dept_month_totals[dept_month_totals["Month"]==prev_month]["Amount"].sum() if prev_month else np.nan
    mom_total_pct = ((total_latest-total_prev)/total_prev) if (prev_month and total_prev!=0) else np.nan

    col1.metric("Latest Month", latest_month)
    col2.metric("Total Cost (Latest)", f"{total_latest:,.0f}")
    col3.metric("Total Cost MoM %", f"{mom_total_pct*100:,.1f}%" if pd.notna(mom_total_pct) else "N/A")
    col4.metric("Departments", int(long_df["Department"].nunique()))

    st.markdown("### Departments overview (Latest month)")
    st.dataframe(dashboard_dept.sort_values("Total Cost (Latest)", ascending=False), use_container_width=True)

    st.markdown("### Top increases (Latest month vs previous month)")
    st.dataframe(
        latest_increases[["Department","GL name","PrevMonthAmount","Amount","MoM Var (Latest)","MoM % (Latest)"]]
        .sort_values("MoM Var (Latest)", ascending=False)
        .head(20),
        use_container_width=True
    )

with tab_norm:
    st.subheader("Normalized Data (GL name | Department | Month | Amount)")
    st.dataframe(long_df, use_container_width=True)

with tab_cmp:
    st.subheader("All Months Comparison (Each month vs ALL previous months)")
    st.dataframe(all_months_cmp, use_container_width=True)

with tab_anom:
    st.subheader("Anomaly Log (History-based)")
    anom = all_months_cmp[(all_months_cmp["Severity"].isin(["High","Medium"]))].copy()

    # Add a simple English comment per row (history-based)
    def row_comment(r):
        if r["Severity"] == "High":
            return f"Significant deviation for {r['GL name']} in {r['Department']} during {r['Month']} vs historical average."
        if r["Severity"] == "Medium":
            return f"Moderate deviation for {r['GL name']} in {r['Department']} during {r['Month']} vs historical average."
        return "Normal."

    anom["Comment_EN"] = anom.apply(row_comment, axis=1)

    st.dataframe(anom.sort_values(["Month","Severity"], ascending=[False, True]), use_container_width=True)

with tab_dept:
    st.subheader(f"Department comments (Latest month {latest_month} vs previous month {prev_month})")
    st.dataframe(dept_comments_df, use_container_width=True)

# -------- Export to Excel --------
output = BytesIO()
with pd.ExcelWriter(output, engine="openpyxl") as writer:
    long_df.to_excel(writer, sheet_name="Normalized", index=False)
    all_months_cmp.to_excel(writer, sheet_name="All_Months_Comparison", index=False)
    anom.to_excel(writer, sheet_name="Anomaly_Log", index=False)
    dept_comments_df.to_excel(writer, sheet_name="Dept_Comments_LatestMoM", index=False)
    dashboard_dept.to_excel(writer, sheet_name="Dashboard_Dept", index=False)
    latest_increases.to_excel(writer, sheet_name="Latest_MoM_Increases", index=False)

st.download_button(
    "Download Excel Output",
    data=output.getvalue(),
    file_name="cost_review_output.xlsx"
)

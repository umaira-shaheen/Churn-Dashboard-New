"""Customer Churn & Subscriptions — interactive Streamlit dashboard.

To run: python -m streamlit run app.py
"""

from __future__ import annotations

from pathlib import Path
import html

import pandas as pd
import plotly.graph_objects as go
import streamlit as st


# Built-in analytics — app.py also runs alone without analytics.py.
REQUIRED_COLUMNS = ("CustomerID", "SubscriptionType", "MonthlyCharges", "Churn")
PLAN_ORDER = ("Basic", "Standard", "Premium")
BAND_EDGES = (4.99, 8.0, 11.0, 14.0, 17.0, 20.0)
BAND_LABELS = ("4.99–<8", "8–<11", "11–<14", "14–<17", "17–<20")
ALL_PLANS = "All plans"


def prepare_data(raw: pd.DataFrame) -> pd.DataFrame:
    """Check the dataset, then add ordered, non-time monthly-charge bands.

    Raise a user-facing ValueError rather than silently discarding customers.
    Bounds intentionally follow the supplied data's ~4.99 to ~19.99 charges.
    """
    missing = sorted(set(REQUIRED_COLUMNS) - set(raw.columns))
    if missing:
        raise ValueError("Missing required column(s): " + ", ".join(missing))

    frame = raw.copy()
    if frame[list(REQUIRED_COLUMNS)].isna().any().any():
        raise ValueError("The required fields contain missing values; please clean the CSV.")

    if frame["CustomerID"].duplicated().any():
        raise ValueError("CustomerID must be unique so customers are not counted twice.")

    frame["MonthlyCharges"] = pd.to_numeric(frame["MonthlyCharges"], errors="coerce")
    frame["Churn"] = pd.to_numeric(frame["Churn"], errors="coerce")
    if frame[["MonthlyCharges", "Churn"]].isna().any().any():
        raise ValueError("MonthlyCharges and Churn must contain numeric values.")
    if not frame["Churn"].isin([0, 1]).all():
        raise ValueError("Churn must contain only 0 (retained) and 1 (cancelled).")

    observed_plans = set(frame["SubscriptionType"].unique())
    if not observed_plans.issubset(set(PLAN_ORDER)):
        raise ValueError("SubscriptionType must use Basic, Standard, or Premium.")

    charges = frame["MonthlyCharges"]
    if not ((charges >= BAND_EDGES[0]) & (charges < BAND_EDGES[-1])).all():
        raise ValueError("This dashboard uses charge bands from 4.99 to under 20; data outside that range was found.")

    frame["ChargeBand"] = pd.cut(
        charges,
        bins=BAND_EDGES,
        labels=BAND_LABELS,
        right=False,
        include_lowest=True,
        ordered=True,
    )
    return frame


def overall_summary(frame: pd.DataFrame) -> dict:
    customers = int(frame["CustomerID"].nunique())
    cancelled = int(frame["Churn"].sum())
    return {
        "customers": customers,
        "cancelled": cancelled,
        "cancellation_pct": cancelled / customers * 100 if customers else 0.0,
    }


def plan_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """One row per subscription type; counts and percentages share a denominator."""
    grouped = frame.groupby("SubscriptionType", observed=True).agg(
        customers=("CustomerID", "nunique"), cancelled=("Churn", "sum")
    )
    grouped = grouped.reindex(PLAN_ORDER, fill_value=0).reset_index()
    grouped["cancellation_pct"] = grouped["cancelled"].div(grouped["customers"].replace(0, float("nan"))).mul(100)
    return grouped


def charge_band_summary(frame: pd.DataFrame, selected_plan: str = ALL_PLANS) -> pd.DataFrame:
    """Calculate rate within each band and selected plan (never a raw count trend)."""
    if selected_plan != ALL_PLANS:
        if selected_plan not in PLAN_ORDER:
            raise ValueError(f"Unknown plan: {selected_plan}")
        frame = frame.loc[frame["SubscriptionType"] == selected_plan]

    grouped = frame.groupby("ChargeBand", observed=False).agg(
        customers=("CustomerID", "nunique"), cancelled=("Churn", "sum")
    )
    grouped = grouped.reindex(BAND_LABELS, fill_value=0).reset_index()
    grouped["ChargeBand"] = grouped["ChargeBand"].astype(str)
    grouped["cancellation_pct"] = grouped["cancelled"].div(grouped["customers"].replace(0, float("nan"))).mul(100)
    return grouped


# Exact counts computed from the supplied 243,787-row CSV. The app can render
# from these anonymised aggregates even when GitHub does not contain the raw CSV.
# Format: subscription plan, ordered charge-band, customers, cancelled.
BUNDLED_PLAN_BAND_COUNTS = (
    ("Basic", "4.99–<8", 16257, 2328),
    ("Basic", "8–<11", 16095, 2686),
    ("Basic", "11–<14", 16092, 3141),
    ("Basic", "14–<17", 16565, 3675),
    ("Basic", "17–<20", 16041, 4098),
    ("Standard", "4.99–<8", 16498, 2160),
    ("Standard", "8–<11", 16319, 2493),
    ("Standard", "11–<14", 16242, 3031),
    ("Standard", "14–<17", 16412, 3429),
    ("Standard", "17–<20", 16449, 3986),
    ("Premium", "4.99–<8", 16170, 1874),
    ("Premium", "8–<11", 16261, 2244),
    ("Premium", "11–<14", 16073, 2584),
    ("Premium", "14–<17", 16313, 2998),
    ("Premium", "17–<20", 16000, 3455),
)


def bundled_counts() -> pd.DataFrame:
    """Return actual, privacy-preserving aggregated counts from the supplied CSV."""
    return pd.DataFrame(
        BUNDLED_PLAN_BAND_COUNTS,
        columns=["SubscriptionType", "ChargeBand", "customers", "cancelled"],
    )


def aggregate_counts(frame: pd.DataFrame) -> pd.DataFrame:
    """Compress checked customer rows into 15 plan-by-band groups."""
    grouped = frame.groupby(["SubscriptionType", "ChargeBand"], observed=True).agg(
        customers=("CustomerID", "nunique"), cancelled=("Churn", "sum")
    ).reset_index()
    grouped["ChargeBand"] = grouped["ChargeBand"].astype(str)
    return grouped


def overall_from_counts(counts: pd.DataFrame) -> dict:
    customers = int(counts["customers"].sum())
    cancelled = int(counts["cancelled"].sum())
    return {
        "customers": customers,
        "cancelled": cancelled,
        "cancellation_pct": cancelled / customers * 100 if customers else 0.0,
    }


def plan_from_counts(counts: pd.DataFrame) -> pd.DataFrame:
    grouped = counts.groupby("SubscriptionType", as_index=True).agg(
        customers=("customers", "sum"), cancelled=("cancelled", "sum")
    )
    grouped = grouped.reindex(PLAN_ORDER, fill_value=0).reset_index()
    grouped["cancellation_pct"] = grouped["cancelled"].div(
        grouped["customers"].replace(0, float("nan"))
    ).mul(100)
    return grouped


def charge_band_from_counts(counts: pd.DataFrame, selected_plan: str = ALL_PLANS) -> pd.DataFrame:
    if selected_plan != ALL_PLANS:
        if selected_plan not in PLAN_ORDER:
            raise ValueError(f"Unknown plan: {selected_plan}")
        counts = counts.loc[counts["SubscriptionType"] == selected_plan]
    grouped = counts.groupby("ChargeBand", as_index=True).agg(
        customers=("customers", "sum"), cancelled=("cancelled", "sum")
    )
    grouped = grouped.reindex(BAND_LABELS, fill_value=0).reset_index()
    grouped["cancellation_pct"] = grouped["cancelled"].div(
        grouped["customers"].replace(0, float("nan"))
    ).mul(100)
    return grouped


st.set_page_config(
    page_title="Customer Churn | Subscriptions",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

PROJECT_DIR = Path(__file__).resolve().parent
DATA_PATH = PROJECT_DIR / "data" / "subscription_churn_full.csv"
COLORS = {
    "All plans": "#5464DE",
    "Basic": "#D76C48",
    "Standard": "#168D87",
    "Premium": "#8066C8",
}
INK = "#1B2841"
MUTED = "#63748B"
GRID = "#E9EEF5"

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');
    html, body, [data-testid="stApp"] {font-family:'DM Sans',system-ui,sans-serif;}
    [data-testid="stAppViewContainer"] {background:#F6F8FC;}
    .block-container {max-width:1250px; padding-top:2rem; padding-bottom:4rem;}
    h1, h2, h3, h4 {font-family:'Manrope',system-ui,sans-serif; letter-spacing:-.035em;}
    h1 {font-size:2.15rem !important;}
    div[data-testid="stVerticalBlockBorderWrapper"] > div {border-color:#E6ECF4 !important;}
    .hero {position:relative;overflow:hidden;isolation:isolate;
      background:linear-gradient(115deg,#1A2A47 0%,#283A6C 68%,#394C89 100%);
      border-radius:22px;padding:30px 34px 29px;color:white;
      box-shadow:0 14px 32px rgba(25,39,77,.11);}
    .hero:before {content:"";position:absolute;inset:auto -75px -145px auto;
      width:360px;height:360px;border-radius:50%;z-index:-1;
      background:radial-gradient(circle,rgba(142,167,255,.20) 0%,rgba(142,167,255,0) 69%);
      pointer-events:none;}
    .hero-eyebrow {display:flex;align-items:center;gap:9px;
      text-transform:uppercase;font-size:11px;font-weight:800;letter-spacing:.15em;
      color:#B8CCFF;margin-bottom:16px;}
    .hero-eyebrow-dot {width:8px;height:8px;flex-shrink:0;border-radius:50%;
      background:#7DE1C3;box-shadow:0 0 0 4px rgba(125,225,195,.13);}
    .hero-title-row {display:flex;align-items:center;gap:18px;margin-bottom:14px;}
    .hero-brandmark {width:64px;height:64px;flex-shrink:0;border-radius:18px;
      display:flex;align-items:center;justify-content:center;
      background:linear-gradient(145deg,rgba(173,194,255,.28),rgba(121,157,248,.10));
      border:1px solid rgba(215,226,255,.30);box-shadow:0 8px 20px rgba(8,17,48,.14);
      transition:transform .28s ease,background .28s ease,box-shadow .28s ease;}
    .hero-brandmark svg {width:40px;height:40px;display:block;}
    .hero:hover .hero-brandmark {transform:translateY(-3px) rotate(-3deg);
      box-shadow:0 12px 26px rgba(8,17,48,.22);}
    .hero h1 {color:white;font-size:36px !important;line-height:1.16;margin:0;font-weight:800;}
    .hero p {color:#D7E0F8;font-size:15px;margin:0;max-width:840px;line-height:1.7;}
    .hero-strip {display:flex;gap:9px;flex-wrap:wrap;margin-top:22px;align-items:center;}
    .hero-pill {background:rgba(255,255,255,.09);border:1px solid rgba(255,255,255,.18);
      border-radius:999px;font-size:11px;letter-spacing:.03em;padding:8px 12px;color:#ECF0FF;
      transition:background .2s ease,transform .2s ease,border-color .2s ease;}
    .hero-pill:hover {background:rgba(255,255,255,.16);border-color:rgba(255,255,255,.32);
      transform:translateY(-2px);}
    .hero-links {display:flex;gap:10px;flex-wrap:wrap;margin-top:18px;}
    .hero-link {display:inline-flex;align-items:center;gap:7px;text-decoration:none !important;
      background:rgba(169,194,255,.15);border:1px solid rgba(193,211,255,.32);
      padding:9px 13px;border-radius:10px;color:#FFF !important;font-weight:700;font-size:12px;
      transition:transform .2s ease,background .2s ease;}
    .hero-link:hover {background:rgba(169,194,255,.29);transform:translateY(-2px);}
    #plan-comparison, #charge-band-pattern {scroll-margin-top:80px;}
    @media (prefers-reduced-motion:reduce) {
      .hero-brandmark,.hero-pill,.hero-link {transition:none !important;}
      .hero:hover .hero-brandmark,.hero-pill:hover,.hero-link:hover {transform:none !important;}
    }
    .section-kicker {font-size:11px;letter-spacing:.13em;text-transform:uppercase;color:#6170B6;font-weight:800;margin-bottom:2px;}
    .section-heading {color:#20304D;font-size:23px;font-weight:800;margin:0 0 6px;}
    .section-description {font-size:14px;color:#697A8F;line-height:1.65;margin-bottom:12px;}
    .metric-card {min-height:130px;background:white;border:1px solid #E7EDF5;border-radius:18px;
      padding:20px 20px 17px;box-shadow:0 4px 15px rgba(37,58,94,.035);}
    .metric-label {color:#6D7D92;font-size:12px;font-weight:700;letter-spacing:.01em;}
    .metric-value {color:#202F4A;font-family:'Manrope',sans-serif;font-weight:800;font-size:29px;letter-spacing:-.05em;margin:9px 0 2px;}
    .metric-note {color:#728196;font-size:11px;}
    .metric-highlight {background:#EEF2FF;border-color:#DEE5FF;}
    .metric-highlight .metric-value {color:#3E4FB8;}
    .insight-card {background:#F4F6FF;border:1px solid #E5EAFD;border-radius:16px;
      padding:19px 19px 18px; margin:4px 0 16px;}
    .insight-title {font-size:11px;letter-spacing:.11em;text-transform:uppercase;color:#5464BD;font-weight:800;margin:0 0 11px;}
    .insight-main {font-family:'Manrope',sans-serif;color:#23345C;font-weight:800;font-size:17px;line-height:1.35;margin-bottom:10px;}
    .insight-body {font-size:13px;line-height:1.65;color:#5F6F88;}
    .mini-stat {border-top:1px solid #DDE4F8;padding-top:12px;margin-top:15px;color:#5F6F88;font-size:12px;}
    .mini-stat strong {font-weight:800;color:#2E4165;}
    .method-note {font-size:12px;line-height:1.7;color:#77869A;}
    [data-testid="stRadio"] p {font-size:12px;font-weight:700;color:#5B6E8C;}
    [data-testid="stHorizontalBlock"] {align-items:stretch;}
    @media (max-width:700px) {
      .hero {padding:24px 22px;border-radius:18px;}
      .hero h1 {font-size:27px !important;}
      .hero-title-row {gap:12px;}
      .hero-brandmark {width:50px;height:50px;border-radius:14px;}
      .hero-brandmark svg {width:32px;height:32px;}
      .hero-links {gap:8px;}
      .metric-card {min-height:116px;}
      .block-container {padding-top:1.1rem;}
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def metric_card(label: str, value: str, note: str, highlight: bool = False) -> None:
    highlight_class = " metric-highlight" if highlight else ""
    st.markdown(
        f'<div class="metric-card{highlight_class}">'
        f'<div class="metric-label">{html.escape(label)}</div>'
        f'<div class="metric-value">{html.escape(value)}</div>'
        f'<div class="metric-note">{html.escape(note)}</div></div>',
        unsafe_allow_html=True,
    )


def section_header(number: str, heading: str, description: str) -> None:
    section_id = number.split("/", 1)[-1].strip().lower().replace(" ", "-")
    st.markdown(
        f'<div id="{html.escape(section_id)}" class="section-kicker">{html.escape(number)}</div>'
        f'<div class="section-heading">{html.escape(heading)}</div>'
        f'<div class="section-description">{html.escape(description)}</div>',
        unsafe_allow_html=True,
    )


def insight_card(heading: str, main: str, body: str, tail: str = "") -> None:
    tail_html = f'<div class="mini-stat">{tail}</div>' if tail else ""
    st.markdown(
        f'<div class="insight-card"><div class="insight-title">{html.escape(heading)}</div>'
        f'<div class="insight-main">{html.escape(main)}</div>'
        f'<div class="insight-body">{html.escape(body)}</div>{tail_html}</div>',
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def load_data(path: str) -> tuple[pd.DataFrame, str]:
    """Use the original CSV if available; otherwise use its exact anonymised counts.

    This avoids FileNotFoundError on GitHub deployments that only include code.
    """
    preferred = Path(path)
    candidates = [
        preferred,
        PROJECT_DIR / "subscription_churn_full.csv",
        PROJECT_DIR / "subscription_churn_full (1)(1).csv",
    ]
    # Also support a CSV inside data/ under either supplied filename.
    candidates.append(PROJECT_DIR / "data" / "subscription_churn_full (1)(1).csv")
    for candidate in dict.fromkeys(candidates):
        if candidate.is_file():
            raw = prepare_data(pd.read_csv(candidate))
            return aggregate_counts(raw), "Full customer dataset (CSV)"
    return bundled_counts(), "Bundled anonymised counts (calculated from the provided CSV)"


def base_layout(height: int = 375) -> dict:
    return dict(
        template="plotly_white",
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="DM Sans, Arial, sans-serif", size=12, color=INK),
        margin=dict(l=52, r=18, t=37, b=56),
        showlegend=False,
        hoverlabel=dict(bgcolor="#20304D", font_color="white", font_size=12),
    )


def make_plan_chart(summary: pd.DataFrame) -> go.Figure:
    source = summary[summary["customers"] > 0]
    colors = [COLORS[name] for name in source["SubscriptionType"]]
    fig = go.Figure()
    fig.add_bar(
        x=source["SubscriptionType"],
        y=source["cancellation_pct"],
        marker=dict(color=colors, line=dict(color="rgba(0,0,0,0)", width=0)),
        width=0.51,
        text=[f"{x:.2f}%" for x in source["cancellation_pct"]],
        textposition="outside",
        textfont=dict(size=17, color=INK, family="Manrope, sans-serif"),
        customdata=source[["customers", "cancelled"]].to_numpy(),
        hovertemplate=(
            "<b>%{x}</b><br>Cancellation rate: <b>%{y:.2f}%</b>"
            "<br>Customers: %{customdata[0]:,}"
            "<br>Cancelled: %{customdata[1]:,}<extra></extra>"
        ),
    )
    ymax = max(25.0, float(source["cancellation_pct"].max()) * 1.23)
    fig.update_layout(**base_layout(340), bargap=0.40)
    fig.update_yaxes(
        title_text="Cancellation rate (%)",
        range=[0, ymax],
        ticksuffix="%",
        showgrid=True,
        gridcolor=GRID,
        zeroline=False,
        tickfont=dict(color=MUTED, size=11),
        title_font=dict(color=MUTED, size=12),
        dtick=5,
    )
    fig.update_xaxes(
        title_text=None,
        tickfont=dict(color=INK, size=13),
        showgrid=False,
        linecolor="#DFE6EF",
    )
    return fig


def make_band_chart(
    counts: pd.DataFrame,
    selected_plan: str,
    overlay_plans: bool,
) -> go.Figure:
    fig = go.Figure()
    overlays = list(PLAN_ORDER) if overlay_plans else [selected_plan]

    for plan in overlays:
        source = charge_band_from_counts(counts, plan)
        is_primary = (plan == selected_plan or not overlay_plans)
        name = "All customers" if plan == ALL_PLANS else plan
        color = COLORS[plan]
        point_text = (
            [f"{value:.1f}%" if pd.notna(value) else "" for value in source["cancellation_pct"]]
            if not overlay_plans else None
        )
        fig.add_trace(go.Scatter(
            x=list(BAND_LABELS),
            y=source["cancellation_pct"],
            mode="lines+markers+text" if point_text is not None else "lines+markers",
            name=name,
            line=dict(color=color, width=3.5 if is_primary else 2.5, shape="linear"),
            marker=dict(size=11 if is_primary else 9, color=color, line=dict(color="white", width=2)),
            text=point_text,
            textposition="top center",
            textfont=dict(size=12, color=color, family="Manrope, sans-serif"),
            customdata=source[["customers", "cancelled"]].to_numpy(),
            hovertemplate=(
                f"<b>{html.escape(name)}</b><br>Charge band: %{{x}}"
                "<br>Cancellation rate: <b>%{y:.2f}%</b>"
                "<br>Customers: %{customdata[0]:,}"
                "<br>Cancelled: %{customdata[1]:,}<extra></extra>"
            ),
            connectgaps=False,
        ))

    fig.update_layout(**base_layout(366))
    fig.update_layout(
        margin=dict(l=52, r=25, t=45, b=73),
        showlegend=overlay_plans,
        legend=dict(orientation="h", yanchor="bottom", y=1.10, x=0.0, xanchor="left", font=dict(size=11)),
        hovermode="x unified" if overlay_plans else "closest",
    )
    fig.update_yaxes(
        title_text="Cancellation rate (%)",
        range=[0, 34],
        ticksuffix="%",
        dtick=5,
        gridcolor=GRID,
        zeroline=False,
        tickfont=dict(color=MUTED, size=11),
        title_font=dict(color=MUTED, size=12),
    )
    # Labels are deliberately ordinal; this is NOT a chronological time-series chart.
    fig.update_xaxes(
        title_text="Monthly-charge bands (currency not provided)",
        categoryorder="array",
        categoryarray=list(BAND_LABELS),
        tickfont=dict(color=INK, size=11),
        title_font=dict(color=MUTED, size=12),
        showgrid=False,
        linecolor="#DFE6EF",
    )
    return fig


try:
    df, data_source = load_data(str(DATA_PATH))
except (OSError, ValueError, pd.errors.ParserError) as exc:
    st.error(f"Could not load the dashboard data: {exc}")
    st.stop()

stats = overall_from_counts(df)
plans = plan_from_counts(df)
nonempty_plans = plans.loc[plans["customers"] > 0]
highest = nonempty_plans.sort_values("cancellation_pct", ascending=False).iloc[0]
lowest = nonempty_plans.sort_values("cancellation_pct", ascending=True).iloc[0]
plan_gap = float(highest["cancellation_pct"] - lowest["cancellation_pct"])

st.markdown(
    '<div class="hero">'
    '<div class="hero-eyebrow"><span class="hero-eyebrow-dot" aria-hidden="true"></span>'
    'Customer retention insights</div>'
    '<div class="hero-title-row">'
    '<div class="hero-brandmark" role="img" aria-label="Analytics chart logo">'
    '<svg viewBox="0 0 48 48" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">'
    '<rect x="6" y="30" width="7" height="12" rx="2" fill="#A8CBFF"/>'
    '<rect x="20" y="24" width="7" height="18" rx="2" fill="#8BE1D1"/>'
    '<rect x="34" y="16" width="7" height="26" rx="2" fill="#D8E7FF"/>'
    '<path d="M8 20L18 12L26 17L40 6M32 6H40V14" '
    'stroke="#FFFFFF" stroke-width="2.6" stroke-linecap="round" '
    'stroke-linejoin="round" fill="none"/>'
    '</svg></div>'
    '<h1>Customer churn &amp; subscriptions</h1></div>'
    '<p>Discover how subscription plans and monthly charges relate to cancellations. '
    'Compare Basic, Standard and Premium, then explore the trends using interactive filters.</p>'
    f'<div class="hero-strip"><span class="hero-pill">{stats["customers"]:,} customer records</span>'
    '<span class="hero-pill">Subscription plan comparison</span>'
    '<span class="hero-pill">Monthly-charge insights</span>'
    '<span class="hero-pill">Interactive plan filters</span></div>'
    '<div class="hero-links">'
    '<a class="hero-link" href="#plan-comparison">View plan comparison ↗</a>'
    '<a class="hero-link" href="#charge-band-pattern">Explore charge-band trends ↗</a>'
    '</div></div>',
    unsafe_allow_html=True,
)

st.write("")
st.markdown('<div class="section-kicker">The big picture</div>', unsafe_allow_html=True)
st.caption(f"Data source: {data_source}. The rates are calculated using exact customer counts.")

k1, k2, k3, k4 = st.columns(4, gap="small")
with k1:
    metric_card("Customers analysed", f'{stats["customers"]:,}', "Unique CustomerID values")
with k2:
    metric_card("Cancelled subscriptions", f'{stats["cancelled"]:,}', "Churn = 1")
with k3:
    metric_card("Overall cancellation rate", f'{stats["cancellation_pct"]:.2f}%', "Cancelled / all customers", True)
with k4:
    metric_card("Highest-cancellation plan", str(highest["SubscriptionType"]), f'{highest["cancellation_pct"]:.2f}% cancelled')

st.write("")
st.write("")
section_header(
    "01 / PLAN COMPARISON",
    "Which plan has the highest cancellation percentage?",
    "A common zero baseline makes the three plan rates directly comparable. Customer totals appear below and on hover.",
)
plan_chart_col, plan_insight_col = st.columns([3.05, 1.15], gap="large")
with plan_chart_col:
    with st.container(border=True):
        st.plotly_chart(make_plan_chart(plans), width="stretch", theme=None, config={"displaylogo": False, "responsive": True})
        counts_str = "  ·  ".join(
            f'{r.SubscriptionType}: {int(r.customers):,} customers'
            for r in plans.itertuples()
        )
        st.caption(f"Sample size — {counts_str}")
with plan_insight_col:
    insight_card(
        "Key takeaway",
        f'{highest["SubscriptionType"]} has the highest cancellation rate.',
        f'{float(highest["cancellation_pct"]):.2f}% of {highest["SubscriptionType"]} customers cancelled, versus '
        f'{float(lowest["cancellation_pct"]):.2f}% for {lowest["SubscriptionType"]}.',
        f'<strong>{plan_gap:.2f} percentage-point gap</strong> between the highest and lowest plans.',
    )
    st.markdown(
        '<div class="method-note">Each bar shows a percentage of customers within that plan, '
        'not the number of cancelled customers. Hover for both the numerator and denominator.</div>',
        unsafe_allow_html=True,
    )

st.write("")
st.write("")
section_header(
    "02 / CHARGE-BAND PATTERN",
    "How does cancellation vary with monthly charges?",
    "Five ordered charge ranges show cancellation percentage, not elapsed time. Select a plan to test whether the pattern changes.",
)
controls_left, controls_right = st.columns([3.1, 1], gap="large")
with controls_left:
    selected = st.radio(
        "Show charge-band rates for",
        options=[ALL_PLANS, *PLAN_ORDER],
        horizontal=True,
        index=0,
        help="The line updates to show cancellation within the selected subscription plan.",
    )
with controls_right:
    compare = st.checkbox("Compare all three plans", value=False, help="Overlay separate lines for Basic, Standard and Premium.")

band_chart_col, band_insight_col = st.columns([3.05, 1.15], gap="large")
source = charge_band_from_counts(df, selected)
nonempty = source.loc[source["customers"] > 0]
first_rate = float(nonempty.iloc[0]["cancellation_pct"])
last_rate = float(nonempty.iloc[-1]["cancellation_pct"])
diff_rate = last_rate - first_rate
with band_chart_col:
    with st.container(border=True):
        st.plotly_chart(
            make_band_chart(df, selected, compare),
            width="stretch",
            theme=None,
            config={"displaylogo": False, "responsive": True},
        )
with band_insight_col:
    subject = "All customers" if selected == ALL_PLANS else f"{selected} customers"
    direction = "higher" if diff_rate > 0 else "lower" if diff_rate < 0 else "unchanged"
    insight_card(
        "What the selection shows",
        f"{subject}: {abs(diff_rate):.2f} pp {direction} in the highest band.",
        f"Lowest band: {first_rate:.2f}% cancelled. Highest band: {last_rate:.2f}% cancelled. "
        "This comparison is descriptive, not causal.",
        f'<strong>{int(nonempty["customers"].sum()):,} customers</strong> in the selected group.',
    )
    st.markdown(
        '<div class="method-note">Changing the plan recalculates each percentage '
        'using customers <em>within that plan and charge band</em> as the denominator.</div>',
        unsafe_allow_html=True,
    )

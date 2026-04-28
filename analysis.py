"""
analysis.py  —  Travel & Wellness Tourism Dashboard
=====================================================
Backend: data loading, KPIs, all EDA charts.
Univariate + Bivariate charts now accept column & chart-type arguments
so the Streamlit UI can drive them via dropdowns.
"""

import warnings
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

warnings.filterwarnings("ignore")

PAL = ["#00D4FF","#7B61FF","#FF3CAC","#FFB547","#00E5A0","#4ECDC4","#FF6B6B","#C77DFF"]

# ─────────────────────────────────────────────────────────────────
# THEME
# ─────────────────────────────────────────────────────────────────
def _theme(fig: go.Figure, height: int = 340) -> go.Figure:
    fig.update_layout(
        height=height, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="DM Sans", color="#6A7D9C", size=11),
        legend=dict(bgcolor="rgba(3,6,15,.9)", bordercolor="rgba(255,255,255,.07)",
                    borderwidth=1, font=dict(color="#8A9AB8", size=11)),
        xaxis=dict(gridcolor="rgba(255,255,255,.034)", linecolor="rgba(255,255,255,.07)",
                   tickfont=dict(color="#48586C", size=10), zeroline=False),
        yaxis=dict(gridcolor="rgba(255,255,255,.034)", linecolor="rgba(255,255,255,.07)",
                   tickfont=dict(color="#48586C", size=10), zeroline=False),
        margin=dict(l=8, r=8, t=28, b=8),
        hoverlabel=dict(bgcolor="rgba(3,6,15,.97)", bordercolor="rgba(0,212,255,.3)",
                        font=dict(color="#EEF2FF", size=12, family="DM Sans")),
    )
    return fig

# ─────────────────────────────────────────────────────────────────
# DATA
# ─────────────────────────────────────────────────────────────────
def load_raw() -> pd.DataFrame:
    return pd.read_csv("travel_dataset.csv")

def load_cleaned() -> pd.DataFrame:
    try:
        df = pd.read_csv("cleaned_travel_dataset.csv")
    except FileNotFoundError:
        df = clean_data(load_raw())
    df["AgeGroup"] = pd.cut(df["Age"],
        bins=[0,25,35,45,55,65,120],
        labels=["<25","26-35","36-45","46-55","56-65","65+"])
    return df

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    dfc = df.copy()
    dfc["Gender"] = dfc["Gender"].replace("Fe Male","Female")
    for col in dfc.select_dtypes(include=["float64","int64"]).columns:
        if dfc[col].isnull().sum() > 0:
            dfc[col].fillna(dfc[col].median(), inplace=True)
    for col in dfc.select_dtypes(include="object").columns:
        if dfc[col].isnull().sum() > 0:
            dfc[col].fillna(dfc[col].mode()[0], inplace=True)
    for col in ["Age","DurationOfPitch","MonthlyIncome","NumberOfTrips",
                "NumberOfPersonVisiting","NumberOfFollowups","NumberOfChildrenVisiting"]:
        if col in dfc.columns:
            Q1,Q3 = dfc[col].quantile([0.25,0.75])
            IQR   = Q3-Q1
            dfc[col] = dfc[col].clip(lower=Q1-1.5*IQR, upper=Q3+1.5*IQR)
    return dfc

# ─────────────────────────────────────────────────────────────────
# KPIs
# ─────────────────────────────────────────────────────────────────
def compute_kpis(df: pd.DataFrame) -> dict:
    total    = len(df)
    purchased= int(df["ProdTaken"].sum()) if "ProdTaken" in df.columns else 0
    return {
        "total":     total,
        "purchased": purchased,
        "conv_rate": purchased/total*100 if total else 0.0,
        "avg_inc":   df["MonthlyIncome"].mean() if "MonthlyIncome" in df.columns else 0.0,
        "pasp_pct":  df["Passport"].mean()*100 if "Passport" in df.columns else 0.0,
        "avg_age":   df["Age"].mean() if "Age" in df.columns else 0.0,
        "avg_trips": df["NumberOfTrips"].mean() if "NumberOfTrips" in df.columns else 0.0,
    }

# ─────────────────────────────────────────────────────────────────
# COLUMN CATALOGUE  (used to populate dropdowns)
# ─────────────────────────────────────────────────────────────────
def get_numeric_cols(df: pd.DataFrame) -> list[str]:
    exclude = {"CustomerID","ProdTaken","Passport","OwnCar","HighIncome","YoungAdult"}
    return [c for c in df.select_dtypes(include=np.number).columns if c not in exclude]

def get_categorical_cols(df: pd.DataFrame) -> list[str]:
    always_cat = ["ProdTaken","Gender","Occupation","MaritalStatus","Designation",
                  "TypeofContact","ProductPitched","AgeGroup","CityTier"]
    return [c for c in always_cat if c in df.columns]

def get_all_cols(df: pd.DataFrame) -> list[str]:
    skip = {"CustomerID"}
    return [c for c in df.columns if c not in skip]

# Chart type catalogues per variable type
UNIVARIATE_NUM_CHARTS  = ["Histogram","Box Plot","Violin","ECDF","Strip Plot"]
UNIVARIATE_CAT_CHARTS  = ["Bar Chart","Horizontal Bar","Pie Chart","Donut Chart","Treemap"]
BIVARIATE_CHART_TYPES  = [
    "Box Plot","Violin","Scatter","Bar (mean)","Bar (conversion %)",
    "Strip Plot","Histogram (grouped)","KDE Overlay","Line (trend)","Heatmap (2D bin)",
]

# ─────────────────────────────────────────────────────────────────
# OVERVIEW CHARTS  (fixed — used on Overview page)
# ─────────────────────────────────────────────────────────────────
def fig_purchase_donut(df: pd.DataFrame) -> go.Figure:
    counts = df["ProdTaken"].value_counts()
    fig = px.pie(values=counts.values, names=["Not Purchased","Purchased"],
                 color_discrete_sequence=["#0D1E38","#00D4FF"], hole=0.58)
    fig.update_traces(textposition="inside", textinfo="percent+label",
                      textfont=dict(size=12,color="white"),
                      marker=dict(line=dict(color="rgba(0,0,0,0)",width=0)))
    return _theme(fig, 290)

def fig_products_bar(df: pd.DataFrame) -> go.Figure:
    pc = df["ProductPitched"].value_counts().reset_index()
    pc.columns = ["Product","Count"]
    fig = px.bar(pc, x="Product", y="Count", color="Count",
                 color_continuous_scale=[[0,"#0D1E38"],[.45,"#7B61FF"],[1,"#00D4FF"]], text="Count")
    fig.update_traces(textposition="outside", textfont=dict(size=10,color="#6A7D9C"))
    fig.update_coloraxes(showscale=False)
    return _theme(fig, 290)

def fig_age_histogram(df: pd.DataFrame) -> go.Figure:
    fig = px.histogram(df, x="Age", nbins=25, color_discrete_sequence=["#7B61FF"])
    fig.update_traces(marker_line_width=0, opacity=0.82)
    return _theme(fig, 240)

def fig_gender_pie(df: pd.DataFrame) -> go.Figure:
    gc = df["Gender"].value_counts()
    fig = px.pie(values=gc.values, names=gc.index,
                 color_discrete_sequence=["#00D4FF","#FF3CAC","#FFB547"], hole=0.52)
    fig.update_traces(textinfo="percent", textfont=dict(size=11,color="white"))
    return _theme(fig, 240)

def fig_city_tier_bar(df: pd.DataFrame) -> go.Figure:
    ct = df["CityTier"].value_counts().reset_index()
    ct.columns = ["Tier","Count"]; ct["Tier"] = ct["Tier"].astype(str)
    fig = px.bar(ct, x="Tier", y="Count", color="Count",
                 color_continuous_scale=[[0,"#FF3CAC"],[1,"#FFB547"]], text="Count")
    fig.update_traces(textposition="outside", textfont=dict(size=10,color="#6A7D9C"))
    fig.update_coloraxes(showscale=False)
    return _theme(fig, 240)

# ─────────────────────────────────────────────────────────────────
# ═══ DYNAMIC UNIVARIATE CHART ════════════════════════════════════
# ─────────────────────────────────────────────────────────────────
def fig_univariate(df: pd.DataFrame, col: str, chart_type: str,color: str = "#7B61FF",
                   start_color: str = "#0D1E38") -> go.Figure:
    """
    Render a univariate chart for any column with any supported chart type.
    Numeric chart types : Histogram | Box Plot | Violin | ECDF | Strip Plot
    Categorical types   : Bar Chart | Horizontal Bar | Pie Chart | Donut Chart | Treemap
    """
    is_cat = col in get_categorical_cols(df)
    is_num = not is_cat and (col in get_numeric_cols(df) or pd.api.types.is_numeric_dtype(df[col]))

    # ── Numeric ──────────────────────────────────────────────────
    if is_num:
        data = df[col].dropna()
        # In fig_univariate() function, replace the Histogram section (lines ~130-165) with this updated version:

        if chart_type == "Histogram":
            import plotly.colors as pc
            hv, be = np.histogram(data, bins=28)
            bc = (be[:-1] + be[1:]) / 2
            norm = (hv - hv.min()) / max(hv.max() - hv.min(), 1)
            colorscale = [[0, "#0D1E38"], [0.5, start_color], [1, "#FFB547"]]
            colors_hex = pc.sample_colorscale(colorscale, norm.tolist())

            fig = go.Figure()
            for x0, x1, cnt, c in zip(be[:-1], be[1:], hv, colors_hex):
                fig.add_trace(go.Bar(
                    x=[(x0 + x1) / 2], y=[cnt], width=(x1 - x0) * 0.95,
                    marker_color=c, marker_line_width=0, opacity=0.88,
                    showlegend=False,
                    hovertemplate=f"Range: {x0:.1f}–{x1:.1f}<br>Count: {cnt}<extra></extra>",
                ))
            # Colorbar via dummy scatter (consistent with other chart types)
            fig.add_trace(go.Scatter(
                x=[None], y=[None], mode="markers",
                marker=dict(
                    colorscale=colorscale, cmin=float(data.min()), cmax=float(data.max()),
                    color=[0], showscale=True,
                    colorbar=dict(
                        title=dict(text=col, font=dict(color="#6A7D9C", size=11)),
                        tickfont=dict(color="#6A7D9C", size=10),
                        thickness=14, len=0.75, bgcolor="rgba(9,15,30,0.6)",
                        bordercolor="rgba(255,255,255,0.07)", borderwidth=1,
                        outlinecolor="rgba(0,0,0,0)",
                    ),
                ), showlegend=False,
            ))
            fig.update_layout(barmode="overlay")
            # KDE overlay
            if len(data) > 20:
                xs = np.linspace(bc[0], bc[-1], 200)
                sig = max((bc[-1] - bc[0]) / 12, 1e-6)
                ys = sum(bv * np.exp(-0.5 * ((xs - bx) / sig) ** 2) for bx, bv in zip(bc, hv))
                if ys.max() > 0: ys = ys / ys.max() * hv.max()
                fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines",
                                         line=dict(color="#00D4FF", width=2.2), showlegend=False))
            _add_stats_annotations(fig, data)

        elif chart_type == "Box Plot":
            # Color-map outlier points by value for the colorbar effect
            colorscale_cb = [[0, "#0D1E38"], [0.5, start_color], [1, color]]
            fig = go.Figure()
            fig.add_trace(go.Box(y=data, boxmean="sd", marker_color=color,
                line_color=color, fillcolor=color.replace("FF","44") if "#" in color else color,
                opacity=0.75, name=col, boxpoints="outliers",
                marker=dict(color="#FF3CAC", size=3)))
            # Colorbar via dummy scatter
            fig.add_trace(go.Scatter(
                x=[None], y=[None], mode="markers",
                marker=dict(
                    colorscale=colorscale_cb, cmin=float(data.min()), cmax=float(data.max()),
                    color=[0], showscale=True,
                    colorbar=dict(
                        title=dict(text=col, font=dict(color="#6A7D9C", size=11)),
                        tickfont=dict(color="#6A7D9C", size=10),
                        thickness=14, len=0.75, bgcolor="rgba(9,15,30,0.6)",
                        bordercolor="rgba(255,255,255,0.07)", borderwidth=1,
                        outlinecolor="rgba(0,0,0,0)",
                    ),
                ), showlegend=False,
            ))

        elif chart_type == "Violin":
            colorscale_cb = [[0, "#0D1E38"], [0.5, start_color], [1, color]]
            fig = go.Figure()
            fig.add_trace(go.Violin(y=data, box_visible=True, meanline_visible=True,
                fillcolor=color, opacity=0.6, line_color=color, name=col))
            fig.add_trace(go.Scatter(
                x=[None], y=[None], mode="markers",
                marker=dict(
                    colorscale=colorscale_cb, cmin=float(data.min()), cmax=float(data.max()),
                    color=[0], showscale=True,
                    colorbar=dict(
                        title=dict(text=col, font=dict(color="#6A7D9C", size=11)),
                        tickfont=dict(color="#6A7D9C", size=10),
                        thickness=14, len=0.75, bgcolor="rgba(9,15,30,0.6)",
                        bordercolor="rgba(255,255,255,0.07)", borderwidth=1,
                        outlinecolor="rgba(0,0,0,0)",
                    ),
                ), showlegend=False,
            ))

        elif chart_type == "ECDF":
            sorted_data = np.sort(data)
            ecdf = np.arange(1, len(sorted_data)+1) / len(sorted_data)
            colorscale_cb = [[0, "#0D1E38"], [0.5, start_color], [1, color]]
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=sorted_data, y=ecdf, mode="lines",
                line=dict(color=color, width=2.5), name="ECDF",
                fill="tozeroy", fillcolor=color.replace("#","rgba(").rstrip("FF")+",.08)" if len(color)==7 else "rgba(0,212,255,.08)"))
            fig.add_trace(go.Scatter(
                x=[None], y=[None], mode="markers",
                marker=dict(
                    colorscale=colorscale_cb, cmin=float(sorted_data.min()), cmax=float(sorted_data.max()),
                    color=[0], showscale=True,
                    colorbar=dict(
                        title=dict(text=col, font=dict(color="#6A7D9C", size=11)),
                        tickfont=dict(color="#6A7D9C", size=10),
                        thickness=14, len=0.75, bgcolor="rgba(9,15,30,0.6)",
                        bordercolor="rgba(255,255,255,0.07)", borderwidth=1,
                        outlinecolor="rgba(0,0,0,0)",
                    ),
                ), showlegend=False,
            ))

        elif chart_type == "Strip Plot":
            jitter = np.random.uniform(-0.2, 0.2, len(data))
            fig = go.Figure(go.Scatter(x=jitter, y=data.values, mode="markers",
                marker=dict(color=data.values, colorscale="Viridis", size=4, opacity=0.55,
                            showscale=True), name=col))
            fig.update_xaxes(visible=False)
        else:
            fig = go.Figure()

        # ── Categorical ───────────────────────────────────────────────
    # ── Categorical ───────────────────────────────────────────────
    else:
        col_data = df[col].copy()
        if col == "ProdTaken":
            col_data = col_data.map({1: "Purchased", 0: "Not Purchased"})
        vc = col_data.value_counts().reset_index()
        vc.columns = [col, "Count"]

        # In the categorical section, update bar charts:
        if chart_type == "Bar Chart":
            # Use dynamic start and end colors
            colorscale = [[0, start_color], [1, color]]
            fig = px.bar(vc, x=col, y="Count", color="Count",
                         color_continuous_scale=colorscale, text="Count")
            fig.update_traces(textposition="outside", textfont=dict(size=10, color="#6A7D9C"))
            fig.update_coloraxes(showscale=True, colorbar=dict(
                title=dict(text="Count", font=dict(color="#6A7D9C", size=11)),
                tickfont=dict(color="#6A7D9C", size=10),
                thickness=14, len=0.75, bgcolor="rgba(9,15,30,0.6)",
                bordercolor="rgba(255,255,255,0.07)", borderwidth=1,
                outlinecolor="rgba(0,0,0,0)"))

        elif chart_type == "Horizontal Bar":
            # Use dynamic start and end colors
            colorscale = [[0, start_color], [1, color]]
            fig = px.bar(vc, y=col, x="Count", orientation="h", color="Count",
                         color_continuous_scale=colorscale, text="Count")
            fig.update_traces(textposition="outside", textfont=dict(size=10, color="#6A7D9C"))
            fig.update_coloraxes(showscale=True, colorbar=dict(
                title=dict(text="Count", font=dict(color="#6A7D9C", size=11)),
                tickfont=dict(color="#6A7D9C", size=10),
                thickness=14, len=0.75, bgcolor="rgba(9,15,30,0.6)",
                bordercolor="rgba(255,255,255,0.07)", borderwidth=1,
                outlinecolor="rgba(0,0,0,0)"))

        elif chart_type == "Pie Chart":
            # Use the selected color as the primary color, with PAL as fallback
            custom_colors = [color] + [c for c in PAL if c != color][:len(vc) - 1]
            fig = px.pie(vc, names=col, values="Count",
                         color_discrete_sequence=custom_colors, hole=0)
            fig.update_traces(textinfo="percent+label", textfont=dict(size=10, color="white"))
            fig.update_layout(showlegend=True, legend=dict(
                title=dict(text=col, font=dict(color="#6A7D9C", size=10)),
                bgcolor="rgba(9,15,30,0.6)",
                bordercolor="rgba(255,255,255,0.07)", borderwidth=1
            ))

        elif chart_type == "Donut Chart":
            # Use the selected color as the primary color, with PAL as fallback
            custom_colors = [color] + [c for c in PAL if c != color][:len(vc) - 1]
            fig = px.pie(vc, names=col, values="Count",
                         color_discrete_sequence=custom_colors, hole=0.52)
            fig.update_traces(textinfo="percent+label", textfont=dict(size=10, color="white"))
            fig.update_layout(showlegend=True, legend=dict(
                title=dict(text=col, font=dict(color="#6A7D9C", size=10)),
                bgcolor="rgba(9,15,30,0.6)",
                bordercolor="rgba(255,255,255,0.07)", borderwidth=1
            ))

        elif chart_type == "Treemap":
            fig = px.treemap(vc, path=[col], values="Count",
                             color="Count", color_continuous_scale=[[0, "#0D1E38"], [1, color]])
            fig.update_coloraxes(showscale=True, colorbar=dict(
                title=dict(text="Count", font=dict(color="#6A7D9C", size=11)),
                tickfont=dict(color="#6A7D9C", size=10),
                thickness=14, len=0.75, bgcolor="rgba(9,15,30,0.6)",
                bordercolor="rgba(255,255,255,0.07)", borderwidth=1,
                outlinecolor="rgba(0,0,0,0)"))
        else:
            fig = go.Figure()

    fig.update_layout(title_text=f"{col} — {chart_type}",
                      title_font=dict(size=12, color="#EEF2FF"))
    return _theme(fig, 380)


def _add_stats_annotations(fig, data):
    """Add mean/median vertical lines to a histogram figure."""
    fig.add_vline(x=float(data.mean()), line_dash="dash", line_color="#FFB547", line_width=1.6,
                  annotation_text=f"Mean {data.mean():.1f}", annotation_font_color="#FFB547",
                  annotation_font_size=9)
    fig.add_vline(x=float(data.median()), line_dash="dot", line_color="#00E5A0", line_width=1.6,
                  annotation_text=f"Median {data.median():.1f}", annotation_font_color="#00E5A0",
                  annotation_font_size=9)

def get_numeric_summary(df: pd.DataFrame, col: str) -> dict:
    s = df[col].dropna()
    return {"mean":round(s.mean(),2),"median":round(s.median(),2),
            "std":round(s.std(),2),"skew":round(s.skew(),2),
            "min":round(s.min(),2),"max":round(s.max(),2)}

# ─────────────────────────────────────────────────────────────────
# ═══ DYNAMIC BIVARIATE CHART ═════════════════════════════════════
# ─────────────────────────────────────────────────────────────────
def fig_bivariate(df: pd.DataFrame, x_col: str, y_col: str,
                  chart_type: str, hue_col: str | None = None) -> go.Figure:
    """
    Render a bivariate chart between any two columns with any supported chart type.
    Both axes can be numeric or categorical — logic adapts automatically.
    """
    x_is_num = pd.api.types.is_numeric_dtype(df[x_col])
    y_is_num = pd.api.types.is_numeric_dtype(df[y_col])
    color_seq = PAL

    try:
        # ── Box Plot ──────────────────────────────────────────────
        if chart_type == "Box Plot":
            fig = px.box(df, x=x_col, y=y_col if y_is_num else x_col,
                         color=hue_col if hue_col else x_col if not x_is_num else None,
                         color_discrete_sequence=color_seq, points="outliers",
                         labels={x_col:x_col, y_col:y_col})
            fig.update_traces(marker_size=3)

        # ── Violin ────────────────────────────────────────────────
        elif chart_type == "Violin":
            fig = px.violin(df, x=x_col, y=y_col if y_is_num else x_col,
                            color=hue_col if hue_col else x_col if not x_is_num else None,
                            color_discrete_sequence=color_seq, box=True, points=False)

        # ── Scatter ───────────────────────────────────────────────
        elif chart_type == "Scatter":
            fig = px.scatter(df, x=x_col, y=y_col,
                             color=hue_col if hue_col else None,
                             color_discrete_sequence=color_seq,
                             opacity=0.55, trendline="ols",
                             trendline_color_override="#FF3CAC")

        # ── Bar (mean of y per category x) ────────────────────────
        elif chart_type == "Bar (mean)":
            grp = df.groupby(x_col, observed=True)[y_col].mean().reset_index()
            grp.columns = [x_col, y_col]
            fig = px.bar(grp, x=x_col, y=y_col, color=y_col,
                         color_continuous_scale=[[0,"#0D1E38"],[.5,"#7B61FF"],[1,"#00D4FF"]],
                         text=grp[y_col].round(2))
            fig.update_traces(textposition="outside", textfont=dict(size=10,color="#6A7D9C"))
            fig.update_coloraxes(showscale=False)

        # ── Bar (conversion %) ────────────────────────────────────
        elif chart_type == "Bar (conversion %)":
            target = "ProdTaken" if "ProdTaken" in df.columns else y_col
            grp = df.groupby(x_col, observed=True)[target].mean().reset_index()
            grp[target] = (grp[target]*100).round(2)
            grp[x_col]  = grp[x_col].astype(str)
            fig = px.bar(grp, x=x_col, y=target,
                         color=target,
                         color_continuous_scale=[[0,"#0D1E38"],[.5,"#7B61FF"],[1,"#00E5A0"]],
                         text=grp[target].apply(lambda v: f"{v:.1f}%"))
            fig.update_traces(textposition="outside", textfont=dict(size=11,color="#6A7D9C"))
            fig.update_coloraxes(showscale=False)
            fig.update_yaxes(title_text="Conversion Rate (%)")

        # ── Strip Plot ────────────────────────────────────────────
        elif chart_type == "Strip Plot":
            fig = px.strip(df, x=x_col, y=y_col,
                           color=hue_col if hue_col else x_col if not x_is_num else None,
                           color_discrete_sequence=color_seq)

        # ── Histogram (grouped/overlaid) ──────────────────────────
        elif chart_type == "Histogram (grouped)":
            target = "ProdTaken" if "ProdTaken" in df.columns else None
            grp_col = target or hue_col
            if grp_col:
                fig = go.Figure()
                for val, color in zip(sorted(df[grp_col].dropna().unique()), PAL):
                    sub = df[df[grp_col]==val][x_col if x_is_num else y_col].dropna()
                    lbl = ("Purchased" if val==1 else "Not Purchased") if grp_col=="ProdTaken" else str(val)
                    fig.add_trace(go.Histogram(x=sub, nbinsx=25, name=lbl,
                        marker_color=color, opacity=0.65))
                fig.update_layout(barmode="overlay")
            else:
                col_use = x_col if x_is_num else y_col
                fig = px.histogram(df, x=col_use, nbins=25, color_discrete_sequence=["#7B61FF"])

        # ── KDE Overlay ───────────────────────────────────────────
        elif chart_type == "KDE Overlay":
            num_col = x_col if x_is_num else y_col
            cat_col = y_col if x_is_num else x_col
            fig = go.Figure()
            cats = sorted(df[cat_col].dropna().unique()) if not y_is_num else [None]
            if cats[0] is None:
                data = df[num_col].dropna()
                xs   = np.linspace(data.min(), data.max(), 200)
                sig  = data.std()/2 or 1
                ys   = sum(np.exp(-0.5*((xs-v)/sig)**2) for v in data) / len(data)
                fig.add_trace(go.Scatter(x=xs, y=ys/ys.max(), mode="lines",
                    line=dict(color="#00D4FF",width=2.2), fill="tozeroy",
                    fillcolor="rgba(0,212,255,.08)", name=num_col))
            else:
                for cat, color in zip(cats[:6], PAL):
                    sub = df[df[cat_col]==cat][num_col].dropna()
                    if len(sub) < 5: continue
                    xs  = np.linspace(sub.min(), sub.max(), 200)
                    sig = sub.std()/2 or 1
                    ys  = sum(np.exp(-0.5*((xs-v)/sig)**2) for v in sub) / len(sub)
                    lbl = ("Purchased" if cat==1 else "Not Purchased") if cat_col=="ProdTaken" else str(cat)
                    fig.add_trace(go.Scatter(x=xs, y=ys/ys.max(), mode="lines",
                        line=dict(color=color,width=2.2), fill="tozeroy",
                        fillcolor=color+"18", name=lbl))

        # ── Line (trend) ──────────────────────────────────────────
        elif chart_type == "Line (trend)":
            if x_is_num and y_is_num:
                grp = df.groupby(pd.cut(df[x_col], bins=20), observed=True)[y_col].mean().reset_index()
                grp[x_col] = grp[x_col].apply(lambda i: i.mid if hasattr(i,"mid") else i).astype(float)
                fig = go.Figure(go.Scatter(x=grp[x_col], y=grp[y_col], mode="lines+markers",
                    line=dict(color="#00D4FF",width=2.2),
                    marker=dict(color="#00D4FF",size=6)))
            else:
                grp = df.groupby(x_col, observed=True)[y_col].mean().reset_index() if y_is_num else \
                      df.groupby(y_col, observed=True)[x_col].mean().reset_index()
                fig = go.Figure(go.Scatter(x=grp.iloc[:,0].astype(str), y=grp.iloc[:,1],
                    mode="lines+markers", line=dict(color="#00D4FF",width=2.2),
                    marker=dict(color="#00D4FF",size=7)))

        # ── Heatmap (2D bin) ──────────────────────────────────────
        elif chart_type == "Heatmap (2D bin)":
            if x_is_num and y_is_num:
                h, xe, ye = np.histogram2d(df[x_col].dropna(), df[y_col].dropna(), bins=20)
                fig = go.Figure(go.Heatmap(z=h.T, x=np.round(xe,1), y=np.round(ye,1),
                    colorscale=[[0,"#080E1C"],[.5,"#7B61FF"],[1,"#00D4FF"]]))
            else:
                ct = pd.crosstab(df[y_col], df[x_col])
                fig = go.Figure(go.Heatmap(z=ct.values, x=ct.columns.astype(str),
                    y=ct.index.astype(str),
                    colorscale=[[0,"#080E1C"],[.5,"#7B61FF"],[1,"#00D4FF"]],
                    text=ct.values, texttemplate="%{text}", textfont=dict(size=9)))
        else:
            fig = go.Figure()

    except Exception as e:
        fig = go.Figure()
        fig.add_annotation(text=f"Chart error: {e}", x=0.5, y=0.5,
            xref="paper", yref="paper", showarrow=False,
            font=dict(color="#FF3CAC", size=12))

    fig.update_layout(title_text=f"{x_col}  ×  {y_col}  —  {chart_type}",
                      title_font=dict(size=12, color="#EEF2FF"))
    return _theme(fig, 420)


# ─────────────────────────────────────────────────────────────────
# FIXED BIVARIATE CHARTS  (still used on the dedicated bivariate sub-tabs)
# ─────────────────────────────────────────────────────────────────
def fig_income_vs_conversion(df):
    fig = px.box(df, x="ProdTaken", y="MonthlyIncome", color="ProdTaken",
                 color_discrete_map={0:"#0D1E38",1:"#00D4FF"},
                 labels={"ProdTaken":"Purchased","MonthlyIncome":"Income (₹)"}, points="outliers")
    fig.update_traces(marker_size=3)
    fig.update_xaxes(tickvals=[0,1], ticktext=["Not Purchased","Purchased"])
    return _theme(fig, 330)

def fig_age_group_conversion(df):
    if "AgeGroup" not in df.columns:
        df = df.copy()
        df["AgeGroup"] = pd.cut(df["Age"],bins=[0,25,35,45,55,65,120],
                                 labels=["<25","26-35","36-45","46-55","56-65","65+"])
    ag = df.groupby("AgeGroup",observed=True)["ProdTaken"].mean().reset_index()
    ag.columns = ["AgeGroup","CR"]; ag["CR"]*=100; ag["AgeGroup"]=ag["AgeGroup"].astype(str)
    fig = px.bar(ag, x="AgeGroup", y="CR", color="CR",
                 color_continuous_scale=[[0,"#0D1E38"],[.5,"#7B61FF"],[1,"#00E5A0"]],
                 text=ag["CR"].apply(lambda x:f"{x:.1f}%"))
    fig.update_traces(textposition="outside",textfont=dict(size=11,color="#6A7D9C"))
    fig.update_coloraxes(showscale=False)
    return _theme(fig, 330)

def fig_gender_conversion(df):
    gc = df.groupby("Gender")["ProdTaken"].mean().reset_index()
    gc["ProdTaken"]*=100
    fig = px.bar(gc, x="Gender", y="ProdTaken", color="Gender",
                 color_discrete_sequence=PAL, text=gc["ProdTaken"].apply(lambda x:f"{x:.1f}%"))
    fig.update_traces(textposition="outside",textfont=dict(size=11))
    fig.update_layout(showlegend=False)
    return _theme(fig, 290)

def fig_city_tier_conversion(df):
    cc = df.groupby("CityTier")["ProdTaken"].mean().reset_index()
    cc["ProdTaken"]*=100; cc["CityTier"]=cc["CityTier"].astype(str)
    fig = px.bar(cc, x="CityTier", y="ProdTaken", color="ProdTaken",
                 color_continuous_scale=[[0,"#FF3CAC"],[1,"#FFB547"]],
                 text=cc["ProdTaken"].apply(lambda x:f"{x:.1f}%"))
    fig.update_traces(textposition="outside",textfont=dict(size=11,color="#6A7D9C"))
    fig.update_coloraxes(showscale=False)
    return _theme(fig, 290)

def fig_passport_conversion(df):
    pc = df.groupby("Passport")["ProdTaken"].mean().reset_index()
    pc["ProdTaken"]*=100; pc["label"] = pc["Passport"].map({0:"No Passport",1:"Has Passport"})
    fig = px.bar(pc, x="label", y="ProdTaken", color="label",
                 color_discrete_sequence=["#0D2137","#00E5A0"],
                 text=pc["ProdTaken"].apply(lambda x:f"{x:.1f}%"))
    fig.update_traces(textposition="outside",textfont=dict(size=12,color="#6A7D9C"))
    fig.update_layout(showlegend=False)
    return _theme(fig, 290)

def fig_occupation_conversion(df):
    oc = df.groupby("Occupation")["ProdTaken"].mean().reset_index()
    oc["ProdTaken"]*=100; oc = oc.sort_values("ProdTaken")
    fig = px.bar(oc, y="Occupation", x="ProdTaken", orientation="h", color="ProdTaken",
                 color_continuous_scale=[[0,"#0D1E38"],[.5,"#7B61FF"],[1,"#00D4FF"]],
                 text=oc["ProdTaken"].apply(lambda x:f"{x:.1f}%"))
    fig.update_traces(textposition="outside",textfont=dict(size=10,color="#6A7D9C"))
    fig.update_coloraxes(showscale=False)
    return _theme(fig, 310)

def fig_pitch_satisfaction_conversion(df):
    ps = df.groupby("PitchSatisfactionScore")["ProdTaken"].mean().reset_index()
    ps["ProdTaken"]*=100
    fig = px.line(ps, x="PitchSatisfactionScore", y="ProdTaken", markers=True,
                  color_discrete_sequence=["#00D4FF"],
                  labels={"ProdTaken":"Conversion Rate (%)","PitchSatisfactionScore":"Pitch Satisfaction"})
    fig.update_traces(line_width=2.5, marker_size=9)
    return _theme(fig, 290)

def fig_correlation_heatmap(df):
    num_df = df.select_dtypes(include=np.number).drop(
        columns=[c for c in ["CustomerID"] if c in df.columns]).dropna(axis=1,how="all")
    corr = num_df.corr().round(2)
    fig = go.Figure(data=go.Heatmap(
        z=corr.values, x=corr.columns.tolist(), y=corr.columns.tolist(),
        colorscale=[[0,"#FF3CAC"],[.5,"#080E1C"],[1,"#00D4FF"]], zmid=0,
        text=corr.values.round(2), texttemplate="%{text}",
        textfont=dict(size=9,color="rgba(255,255,255,.65)"), hoverongaps=False))
    return _theme(fig, 460)

# ─────────────────────────────────────────────────────────────────
# DYNAMIC INSIGHTS & RECOMMENDATIONS
# ─────────────────────────────────────────────────────────────────
def get_univariate_insights(df: pd.DataFrame, col: str) -> dict:
    """
    Return a dict with keys:
      - 'insights'      : list of (icon, text) tuples
      - 'recommendations': list of (icon, text) tuples
    for any column selected in the Univariate tab.
    """
    insights = []
    recs     = []

    is_cat = col in get_categorical_cols(df)
    is_num = not is_cat and (col in get_numeric_cols(df) or pd.api.types.is_numeric_dtype(df[col]))

    # ── Target: ProdTaken ─────────────────────────────────────────
    if col == "ProdTaken":
        vc   = df[col].value_counts()
        n0   = vc.get(0, 0)
        n1   = vc.get(1, 0)
        total = n0 + n1
        ratio = round(n0 / max(n1, 1), 1)
        pct_yes = round(n1 / max(total, 1) * 100, 1)
        pct_no  = round(n0 / max(total, 1) * 100, 1)

        insights.append(("⚠️", f"Class imbalance detected — {pct_no}% did NOT purchase vs {pct_yes}% purchased (ratio ≈ {ratio}:1)."))
        insights.append(("📊", f"Out of {total:,} customers, only {n1:,} converted, making this a highly skewed binary target."))
        insights.append(("🔍", "Imbalanced targets cause models to favour the majority class, inflating accuracy while missing true conversions."))

        recs.append(("🔁", "Apply SMOTE (Synthetic Minority Over-sampling Technique) to synthetically generate minority-class samples."))
        recs.append(("⬆️", "Use random over-sampling of the minority class (ProdTaken = 1) to balance training data."))
        recs.append(("⬇️", "Alternatively, apply under-sampling (e.g. NearMiss) on the majority class to reduce imbalance."))
        recs.append(("⚖️", "Use class_weight='balanced' in tree-based models (RandomForest, XGBoost) so minority class gets higher penalty."))
        recs.append(("📐", "Evaluate models with F1-score, ROC-AUC, and Precision-Recall curves rather than raw accuracy."))

    # ── Gender ────────────────────────────────────────────────────
    elif col == "Gender":
        vc    = df[col].value_counts()
        dom   = vc.index[0]
        dom_p = round(vc.iloc[0] / vc.sum() * 100, 1)
        insights.append(("👥", f"'{dom}' is the dominant gender group, accounting for {dom_p}% of the dataset."))
        if "ProdTaken" in df.columns:
            gc = df.groupby("Gender")["ProdTaken"].mean() * 100
            top_g = gc.idxmax()
            insights.append(("🏆", f"'{top_g}' has the highest conversion rate at {gc.max():.1f}%."))
        insights.append(("📊", f"Dataset contains {vc.nunique()} gender categories with a total of {vc.sum():,} records."))

        recs.append(("🎯", f"Focus marketing campaigns on '{top_g if 'ProdTaken' in df.columns else dom}' segment for higher ROI."))
        recs.append(("🔄", "Encode gender as a binary/one-hot variable before feeding into ML models."))
        recs.append(("📢", "Design gender-specific travel packages to improve conversion across all segments."))

    # ── Occupation ────────────────────────────────────────────────
    elif col == "Occupation":
        vc  = df[col].value_counts()
        dom = vc.index[0]
        insights.append(("💼", f"'{dom}' is the most frequent occupation ({round(vc.iloc[0]/vc.sum()*100,1)}% of customers)."))
        if "ProdTaken" in df.columns:
            oc = df.groupby("Occupation")["ProdTaken"].mean() * 100
            top_o = oc.idxmax()
            insights.append(("🏆", f"'{top_o}' occupation has the highest conversion rate at {oc.max():.1f}%."))
            insights.append(("📉", f"'{oc.idxmin()}' occupation converts the least at {oc.min():.1f}%."))

        recs.append(("🎯", "Prioritise outreach to high-converting occupation segments in sales pipelines."))
        recs.append(("🔠", "One-hot encode occupation column — avoid ordinal encoding as there is no natural order."))
        recs.append(("💡", "Create occupation-specific travel bundles (e.g., business travel perks for salaried professionals)."))

    # ── MaritalStatus ─────────────────────────────────────────────
    elif col == "MaritalStatus":
        vc = df[col].value_counts()
        dom = vc.index[0]
        insights.append(("💍", f"'{dom}' is the most common marital status ({round(vc.iloc[0]/vc.sum()*100,1)}%)."))
        if "ProdTaken" in df.columns:
            ms = df.groupby("MaritalStatus")["ProdTaken"].mean() * 100
            top_ms = ms.idxmax()
            insights.append(("🏆", f"'{top_ms}' customers show the highest purchase rate at {ms.max():.1f}%."))

        recs.append(("👨‍👩‍👧", "Market family packages to married customers; solo/adventure packages for singles."))
        recs.append(("🔠", "One-hot encode this feature; consider merging small categories if present."))

    # ── ProductPitched ────────────────────────────────────────────
    elif col == "ProductPitched":
        vc = df[col].value_counts()
        dom = vc.index[0]
        least = vc.index[-1]
        insights.append(("📦", f"'{dom}' is the most frequently pitched product ({round(vc.iloc[0]/vc.sum()*100,1)}%)."))
        insights.append(("📉", f"'{least}' is pitched the least, representing only {round(vc.iloc[-1]/vc.sum()*100,1)}% of pitches."))
        if "ProdTaken" in df.columns:
            pp = df.groupby("ProductPitched")["ProdTaken"].mean() * 100
            top_pp = pp.idxmax()
            insights.append(("🏆", f"'{top_pp}' has the highest conversion rate at {pp.max():.1f}%."))

        recs.append(("🎯", f"Increase pitch frequency for '{top_pp if 'ProdTaken' in df.columns else dom}' as it converts best."))
        recs.append(("🔠", "One-hot encode this feature before ML; avoid label encoding to prevent false ordinal relationships."))
        recs.append(("💰", "Review pricing and positioning of under-performing products to boost their conversion rates."))

    # ── Designation ───────────────────────────────────────────────
    elif col == "Designation":
        vc = df[col].value_counts()
        insights.append(("🏅", f"'{vc.index[0]}' is the most common designation ({round(vc.iloc[0]/vc.sum()*100,1)}%)."))
        if "ProdTaken" in df.columns:
            dg = df.groupby("Designation")["ProdTaken"].mean() * 100
            insights.append(("🏆", f"'{dg.idxmax()}' designation converts best at {dg.max():.1f}%."))
            insights.append(("📉", f"'{dg.idxmin()}' designation has the lowest conversion rate at {dg.min():.1f}%."))

        recs.append(("🎯", "Target senior designations (VP, AVP) with premium packages as they may have higher disposable income."))
        recs.append(("🔠", "Apply ordinal or target encoding based on seniority hierarchy for ML models."))

    # ── TypeofContact ─────────────────────────────────────────────
    elif col == "TypeofContact":
        vc = df[col].value_counts()
        insights.append(("📞", f"'{vc.index[0]}' is the most common contact type ({round(vc.iloc[0]/vc.sum()*100,1)}%)."))
        if "ProdTaken" in df.columns:
            tc = df.groupby("TypeofContact")["ProdTaken"].mean() * 100
            insights.append(("🏆", f"'{tc.idxmax()}' contact type yields the highest conversion at {tc.max():.1f}%."))

        recs.append(("📢", f"Prioritise '{df.groupby('TypeofContact')['ProdTaken'].mean().idxmax() if 'ProdTaken' in df.columns else vc.index[0]}' as the primary channel."))
        recs.append(("🔠", "Binary encode this feature (e.g. Self-Enquiry = 1, Company Invited = 0)."))

    # ── CityTier ──────────────────────────────────────────────────
    elif col == "CityTier":
        vc = df[col].value_counts()
        insights.append(("🏙️", f"City Tier {vc.index[0]} dominates with {round(vc.iloc[0]/vc.sum()*100,1)}% of customers."))
        if "ProdTaken" in df.columns:
            ct = df.groupby("CityTier")["ProdTaken"].mean() * 100
            insights.append(("🏆", f"City Tier {ct.idxmax()} has the highest purchase rate at {ct.max():.1f}%."))

        recs.append(("🏙️", "Invest more in Tier-1 cities for volume; Tier-3 cities may offer untapped growth potential."))
        recs.append(("🔢", "CityTier is already numeric (1/2/3) — use as-is or treat as ordinal categorical in tree models."))

    # ── AgeGroup ──────────────────────────────────────────────────
    elif col == "AgeGroup":
        vc = df[col].value_counts()
        dom = str(vc.index[0])
        insights.append(("👤", f"'{dom}' is the most represented age group ({round(vc.iloc[0]/vc.sum()*100,1)}%)."))
        if "ProdTaken" in df.columns:
            ag = df.groupby("AgeGroup", observed=True)["ProdTaken"].mean() * 100
            insights.append(("🏆", f"Age group '{ag.idxmax()}' has the highest conversion rate at {ag.max():.1f}%."))

        recs.append(("🎯", f"Target age group '{ag.idxmax() if 'ProdTaken' in df.columns else dom}' with targeted campaigns."))
        recs.append(("🔠", "Apply ordinal encoding for AgeGroup since it has a natural order."))

    # ── Numeric: Age ──────────────────────────────────────────────
    elif col == "Age":
        s = df[col].dropna()
        skew = s.skew()
        insights.append(("📅", f"Age ranges from {s.min():.0f} to {s.max():.0f} years with a mean of {s.mean():.1f} years."))
        insights.append(("📐", f"Distribution skewness = {skew:.2f} ({'right-skewed' if skew > 0.5 else 'left-skewed' if skew < -0.5 else 'approximately normal'})."))
        if "ProdTaken" in df.columns:
            med = s.median()
            above = df[df["Age"] > med]["ProdTaken"].mean() * 100
            below = df[df["Age"] <= med]["ProdTaken"].mean() * 100
            insights.append(("🏆", f"Customers above median age ({med:.0f}) convert at {above:.1f}% vs {below:.1f}% below."))

        recs.append(("🔢", "Bin age into groups (e.g. <25, 26-35 …) for tree models or use as continuous for regression."))
        recs.append(("⚠️", "Check and clip outliers using IQR method — extreme ages may skew model performance."))
        recs.append(("🎯", "Design age-specific packages; younger customers may prefer adventure, older may prefer wellness."))

    # ── Numeric: MonthlyIncome ────────────────────────────────────
    elif col == "MonthlyIncome":
        s = df[col].dropna()
        skew = s.skew()
        insights.append(("💰", f"Income ranges ₹{s.min():,.0f}–₹{s.max():,.0f}, median ₹{s.median():,.0f}."))
        insights.append(("📐", f"Distribution skewness = {skew:.2f} ({'right-skewed — consider log transform' if skew > 1 else 'moderate skew' if skew > 0.5 else 'approximately normal'})."))
        if "ProdTaken" in df.columns:
            med = s.median()
            above = df[df["MonthlyIncome"] > med]["ProdTaken"].mean() * 100
            below = df[df["MonthlyIncome"] <= med]["ProdTaken"].mean() * 100
            insights.append(("🏆", f"Customers above median income convert at {above:.1f}% vs {below:.1f}% below."))

        recs.append(("📊", "Apply log/sqrt transform if skew > 1 to normalise for linear/logistic regression models."))
        recs.append(("✂️", "Clip outliers with IQR method to remove extreme income values before training."))
        recs.append(("💎", "Segment into income bands (Low / Mid / High) and create targeted premium packages."))

    # ── Numeric: DurationOfPitch ──────────────────────────────────
    elif col == "DurationOfPitch":
        s = df[col].dropna()
        insights.append(("🕐", f"Pitch durations range from {s.min():.0f} to {s.max():.0f} minutes (mean {s.mean():.1f} min)."))
        if "ProdTaken" in df.columns:
            med = s.median()
            above = df[df["DurationOfPitch"] > med]["ProdTaken"].mean() * 100
            below = df[df["DurationOfPitch"] <= med]["ProdTaken"].mean() * 100
            diff = above - below
            insights.append(("📈" if diff > 0 else "📉", f"Longer pitches (>{med:.0f} min) {'convert better' if diff > 0 else 'convert worse'} — {above:.1f}% vs {below:.1f}%."))
        insights.append(("📐", f"Skewness = {s.skew():.2f} — {'consider capping very long pitch durations' if s.skew() > 1 else 'distribution is reasonably shaped'}."))

        recs.append(("⏱️", f"Aim for pitch durations around {s.median():.0f} min — very long pitches may lose customer interest."))
        recs.append(("✂️", "Cap extreme duration outliers using the 95th percentile before model training."))
        recs.append(("🎯", "Train sales staff to calibrate pitch length based on customer engagement signals."))

    # ── Numeric: NumberOfTrips ────────────────────────────────────
    elif col == "NumberOfTrips":
        s = df[col].dropna()
        insights.append(("✈️", f"Customers take {s.min():.0f}–{s.max():.0f} trips; median = {s.median():.1f} trips."))
        if "ProdTaken" in df.columns:
            med = s.median()
            freq = df[df["NumberOfTrips"] >= med]["ProdTaken"].mean() * 100
            rare = df[df["NumberOfTrips"] <  med]["ProdTaken"].mean() * 100
            insights.append(("🏆", f"Frequent travelers (≥{med:.0f} trips) convert at {freq:.1f}% vs {rare:.1f}% for infrequent."))

        recs.append(("🎯", "Prioritise frequent travelers for premium product pitches — they show higher intent."))
        recs.append(("🔢", "Use as continuous numeric; consider binning into 'Low/Med/High traveler' segments."))
        recs.append(("🌟", "Introduce loyalty rewards for frequent travelers to further boost conversion."))

    # ── Numeric: NumberOfFollowups ────────────────────────────────
    elif col == "NumberOfFollowups":
        s = df[col].dropna()
        insights.append(("📞", f"Follow-ups range from {s.min():.0f} to {s.max():.0f}, with a mean of {s.mean():.1f}."))
        if "ProdTaken" in df.columns:
            fu = df.groupby(df["NumberOfFollowups"].round())["ProdTaken"].mean() * 100
            top_fu = fu.idxmax()
            insights.append(("🏆", f"{top_fu:.0f} follow-ups yields the highest conversion rate at {fu.max():.1f}%."))
            insights.append(("📉", "Too many follow-ups may indicate customer disinterest — monitor diminishing returns."))

        recs.append(("📋", f"Target ~{top_fu:.0f if 'ProdTaken' in df.columns else s.median():.0f} follow-ups as the optimal outreach depth."))
        recs.append(("⚠️", "Flag customers needing >5 follow-ups for re-evaluation of pitch strategy."))
        recs.append(("🔢", "Use as-is numeric feature; consider interaction term with DurationOfPitch."))

    # ── Generic numeric fallback ──────────────────────────────────
    elif is_num:
        s = df[col].dropna()
        skew   = s.skew()
        kurt   = s.kurtosis()
        null_p = round(df[col].isnull().mean() * 100, 1)

        insights.append(("📊", f"Range: {s.min():.2f} – {s.max():.2f} | Mean: {s.mean():.2f} | Median: {s.median():.2f}."))
        insights.append(("📐", f"Skewness = {skew:.2f}, Kurtosis = {kurt:.2f}. {'Heavy-tailed distribution.' if abs(kurt) > 3 else 'Roughly normal tail behaviour.'}"))
        if null_p > 0:
            insights.append(("⚠️", f"{null_p}% missing values detected in this column."))

        if abs(skew) > 1:
            recs.append(("🔄", "Apply log or sqrt transform to reduce skewness before training linear models."))
        if null_p > 0:
            recs.append(("🩹", "Impute missing values using median (robust to outliers) or KNN imputer."))
        recs.append(("✂️", "Check for outliers using IQR or Z-score method and clip before training."))
        recs.append(("🔢", "Scale this feature using StandardScaler or MinMaxScaler for distance-based models."))

    # ── Generic categorical fallback ──────────────────────────────
    else:
        vc = df[col].value_counts()
        null_p = round(df[col].isnull().mean() * 100, 1)

        insights.append(("🏷️", f"'{col}' has {vc.nunique()} unique categories. Dominant: '{vc.index[0]}' ({round(vc.iloc[0]/vc.sum()*100,1)}%)."))
        if null_p > 0:
            insights.append(("⚠️", f"{null_p}% missing values detected — imputation required."))
        if "ProdTaken" in df.columns:
            cr = df.groupby(col)["ProdTaken"].mean() * 100
            insights.append(("🏆", f"Top converting category: '{cr.idxmax()}' at {cr.max():.1f}%."))

        recs.append(("🔠", "Apply one-hot encoding if ≤ 10 categories; use target/frequency encoding for high cardinality."))
        if null_p > 0:
            recs.append(("🩹", "Fill missing values with mode or a dedicated 'Unknown' category."))
        recs.append(("✂️", "Merge rare categories (< 2% frequency) into an 'Other' bucket to reduce noise."))

    return {"insights": insights, "recommendations": recs}


def get_bivariate_insights(df: pd.DataFrame, x_col: str, y_col: str, chart_type: str) -> dict:
    """
    Return insights and recommendations for any bivariate chart combination.
    """
    insights = []
    recs     = []

    x_is_num = pd.api.types.is_numeric_dtype(df[x_col])
    y_is_num = pd.api.types.is_numeric_dtype(df[y_col])

    # ── Correlation for num × num ─────────────────────────────────
    if x_is_num and y_is_num:
        corr = df[[x_col, y_col]].corr().iloc[0, 1]
        abs_c = abs(corr)
        strength  = "strong" if abs_c > 0.6 else "moderate" if abs_c > 0.3 else "weak"
        direction = "positive" if corr > 0 else "negative"

        insights.append(("📐", f"Pearson correlation = {corr:.3f} — {strength} {direction} linear relationship."))
        if abs_c > 0.6:
            insights.append(("⚠️", f"High correlation ({corr:.2f}) may cause multicollinearity if both features are used together in linear models."))
        if abs_c < 0.1:
            insights.append(("🔍", "Near-zero correlation — these variables are largely independent of each other."))

        if "ProdTaken" in [x_col, y_col]:
            target = "ProdTaken"
            feat   = y_col if x_col == "ProdTaken" else x_col
            cr     = df.groupby(pd.qcut(df[feat], q=4, duplicates="drop"))[target].mean() * 100
            best_q = cr.idxmax()
            insights.append(("🏆", f"Highest conversion ({cr.max():.1f}%) occurs in the '{best_q}' range of {feat}."))

        recs.append(("🔢", "Use both features in a scatter plot with trendline (OLS) to visualise the relationship."))
        if abs_c > 0.6:
            recs.append(("✂️", "Consider dropping one of these highly correlated features, or use PCA to combine them."))
        recs.append(("📊", "Bin one feature into quartiles for a grouped bar analysis to reveal non-linear patterns."))

    # ── Cat × Num ─────────────────────────────────────────────────
    elif (not x_is_num and y_is_num) or (x_is_num and not y_is_num):
        cat_col = x_col if not x_is_num else y_col
        num_col = y_col if not x_is_num else x_col

        grp    = df.groupby(cat_col, observed=True)[num_col]
        means  = grp.mean()
        stds   = grp.std()
        top    = means.idxmax()
        bottom = means.idxmin()

        insights.append(("📊", f"'{top}' has the highest mean {num_col} ({means[top]:.2f}) and '{bottom}' the lowest ({means[bottom]:.2f})."))
        insights.append(("📐", f"Spread within groups varies — std ranges from {stds.min():.2f} to {stds.max():.2f}, indicating unequal variance."))

        if cat_col == "ProdTaken":
            insights.append(("🏆", f"Customers who {'purchased' if means[1] > means[0] else 'did not purchase'} show higher average {num_col}."))

        recs.append(("📦", "Use Box or Violin charts to compare distributions — they expose median, spread, and outliers simultaneously."))
        recs.append(("🔬", "Consider a statistical test (e.g. t-test / ANOVA) to confirm whether group differences are significant."))
        recs.append(("🎯", f"Segment customers by '{cat_col}' and tailor {num_col}-driven strategies per group."))

    # ── Cat × Cat ─────────────────────────────────────────────────
    else:
        ct      = pd.crosstab(df[x_col], df[y_col])
        biggest = ct.stack().idxmax()

        insights.append(("🗂️", f"Most common combination: '{biggest[0]}' × '{biggest[1]}' with {ct.loc[biggest]:.0f} occurrences."))
        insights.append(("📊", f"Cross-tab reveals {ct.shape[0]} × {ct.shape[1]} category combinations between '{x_col}' and '{y_col}'."))

        if "ProdTaken" in [x_col, y_col]:
            other = y_col if x_col == "ProdTaken" else x_col
            cr    = df.groupby(other)["ProdTaken"].mean() * 100
            insights.append(("🏆", f"'{cr.idxmax()}' in '{other}' converts best at {cr.max():.1f}%."))

        recs.append(("📊", "Use a grouped/stacked bar chart or heatmap to visualise cross-category frequencies."))
        recs.append(("🔬", "Run a Chi-square test to determine if the association between these categories is statistically significant."))
        recs.append(("🔠", "One-hot encode both features before ML training to avoid ordinal bias."))

    # ── Chart-type specific tips ──────────────────────────────────
    if chart_type == "Scatter":
        recs.append(("🔍", "Add a hue/colour variable to reveal hidden groupings within the scatter cloud."))
    elif chart_type in ("Box Plot", "Violin"):
        recs.append(("📦", "Enable 'points=all' to see raw data density alongside the summary statistics."))
    elif "Heatmap" in chart_type:
        recs.append(("🌡️", "Bright cells in the heatmap mark high-frequency or high-value combinations — investigate those segments."))
    elif "Histogram" in chart_type:
        recs.append(("📊", "Overlapping histograms show distributional differences between groups — look for separation or overlap."))

    return {"insights": insights, "recommendations": recs}


# ─────────────────────────────────────────────────────────────────
# INSIGHTS (original stats helper)
# ─────────────────────────────────────────────────────────────────
def compute_insight_stats(df: pd.DataFrame) -> dict:
    pp = df.groupby("Passport")["ProdTaken"].mean()
    stats = {
        "passport_uplift": round(pp.get(1,0)/max(pp.get(0,1e-9),1e-9), 2),
        "conv_rate":       round(df["ProdTaken"].mean()*100, 1),
    }
    med = df["MonthlyIncome"].median()
    above = df[df["MonthlyIncome"]>med]["ProdTaken"].mean()
    below = df[df["MonthlyIncome"]<=med]["ProdTaken"].mean()
    stats["income_uplift"] = round(above/max(below,1e-9), 2)
    if "AgeGroup" not in df.columns:
        df = df.copy()
        df["AgeGroup"] = pd.cut(df["Age"],bins=[0,25,35,45,55,65,120],
                                 labels=["<25","26-35","36-45","46-55","56-65","65+"])
    stats["top_age_segment"] = str(df.groupby("AgeGroup",observed=True)["ProdTaken"].mean().idxmax())
    return stats
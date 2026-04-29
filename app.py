"""
app.py  —  Trips & Travel Analytics Platform  (v2 — 2-Stage Pipeline)
======================================================================
Run: streamlit run app.py
Requires: travel_dataset.csv in same folder + analysis.py + prediction.py

Stage 1 · Train All Models  → baseline training → best base learner
Stage 2 · Hyperparameter Tuning → tunes ONLY the Stage-1 winner
         Model Analysis     → analyses the final active model
         Predict            → single + batch inference
"""

import warnings
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import analysis   as ana
import prediction as pred

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════
st.set_page_config(page_title="Trips & Travel — Analytics", page_icon="✈️",
                   layout="wide", initial_sidebar_state="expanded")

# ══════════════════════════════════════════════════════════════════
# CSS  (unchanged from v1 — all existing styling preserved)
# ══════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:opsz,wght@9..40,300;9..40,400;9..40,500;9..40,600&display=swap');
:root{--deep:#03060F;--surface:rgba(9,15,30,0.92);--border:rgba(255,255,255,0.07);--c-cyan:#00D4FF;--c-violet:#7B61FF;--c-rose:#FF3CAC;--c-amber:#FFB547;--c-green:#00E5A0;--txt1:#EEF2FF;--txt2:#6A7D9C;--txt3:#2E4060;}
html,body,[class*="css"],.stApp{font-family:'DM Sans',sans-serif !important;background:var(--deep) !important;color:var(--txt1) !important;}
.main .block-container{background:transparent !important;padding:1.4rem 2rem 5rem !important;max-width:1600px !important;}
::-webkit-scrollbar{width:4px}::-webkit-scrollbar-track{background:var(--deep)}::-webkit-scrollbar-thumb{background:var(--c-violet);border-radius:99px}
.orb{position:fixed;border-radius:50%;filter:blur(110px);pointer-events:none;z-index:0;animation:orbPulse ease-in-out infinite alternate}
.orb-a{width:500px;height:500px;background:#00D4FF;opacity:.04;top:-150px;left:-90px;animation-duration:13s}
.orb-b{width:420px;height:420px;background:#7B61FF;opacity:.05;bottom:-130px;right:-50px;animation-duration:18s;animation-delay:-6s}
.orb-c{width:300px;height:300px;background:#FF3CAC;opacity:.035;top:40%;left:43%;animation-duration:10s;animation-delay:-3s}
@keyframes orbPulse{from{transform:translate(0,0) scale(1)}to{transform:translate(20px,-26px) scale(1.07)}}
.pg-eye{font-size:.62rem;font-weight:600;letter-spacing:.3em;text-transform:uppercase;color:var(--c-cyan);margin-bottom:.3rem}
.pg-h1{font-family:'Syne',sans-serif;font-size:clamp(1.9rem,3vw,2.9rem);font-weight:800;line-height:1.05;color:#FFFFFF;margin:0 0 .3rem}
.pg-h1 span{color:var(--c-cyan)}
.pg-sub{color:var(--txt2);font-size:.91rem;font-weight:300;margin-bottom:1rem}
.pg-rule{width:52px;height:2px;margin-bottom:1.5rem;background:linear-gradient(90deg,var(--c-cyan),transparent)}
.kpi-grid{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:.75rem;margin-bottom:1.4rem}
.kpi{position:relative;background:var(--surface);border:1px solid var(--border);border-radius:14px;padding:1rem .95rem;overflow:visible;backdrop-filter:blur(22px);transition:transform .32s cubic-bezier(.16,1,.3,1),box-shadow .32s ease,border-color .28s;cursor:default;min-width:0}
.kpi::before{content:'';position:absolute;top:0;left:0;right:0;height:2px;border-radius:14px 14px 0 0;background:var(--kbar,linear-gradient(90deg,#00D4FF,transparent))}
.kpi:hover{transform:translateY(-5px);border-color:var(--kborder,rgba(0,212,255,.25));box-shadow:0 16px 38px var(--kglow,rgba(0,212,255,.12))}
.kc{--kbar:linear-gradient(90deg,#00D4FF,transparent);--kborder:rgba(0,212,255,.24);--kglow:rgba(0,212,255,.12)}
.kg{--kbar:linear-gradient(90deg,#00E5A0,transparent);--kborder:rgba(0,229,160,.24);--kglow:rgba(0,229,160,.12)}
.kv{--kbar:linear-gradient(90deg,#7B61FF,transparent);--kborder:rgba(123,97,255,.24);--kglow:rgba(123,97,255,.12)}
.ka{--kbar:linear-gradient(90deg,#FFB547,transparent);--kborder:rgba(255,181,71,.24);--kglow:rgba(255,181,71,.12)}
.kr{--kbar:linear-gradient(90deg,#FF3CAC,transparent);--kborder:rgba(255,60,172,.24);--kglow:rgba(255,60,172,.12)}
.kpi-ic{display:inline-flex;align-items:center;justify-content:center;width:30px;height:30px;border-radius:7px;background:rgba(255,255,255,.05);font-size:.9rem;margin-bottom:.65rem}
.kpi-val{font-family:'Syne',sans-serif;font-size:clamp(1.1rem,2vw,1.55rem);font-weight:800;line-height:1.1;display:block;margin-bottom:.22rem;white-space:normal;word-break:break-all}
.kpi-lbl{font-size:.66rem;font-weight:600;letter-spacing:.07em;text-transform:uppercase;color:var(--txt1);display:block;margin-bottom:.1rem;line-height:1.3}
.kpi-sub{font-size:.62rem;color:var(--txt3);display:block;line-height:1.3}
.cc{background:var(--surface);border:1px solid var(--border);border-radius:16px;padding:1.1rem 1.05rem .45rem;backdrop-filter:blur(20px);position:relative;overflow:hidden;transition:box-shadow .32s ease,border-color .32s;margin-bottom:.85rem}
.cc::before{content:'';position:absolute;top:0;left:0;right:0;height:1px;background:linear-gradient(90deg,transparent,rgba(255,255,255,.07),transparent)}
.cc:hover{border-color:rgba(0,212,255,.11);box-shadow:0 8px 32px rgba(0,212,255,.06)}
.cc-lbl{font-family:'Syne',sans-serif;font-size:.85rem;font-weight:700;color:var(--txt1);margin:0 0 .65rem;display:flex;align-items:center;gap:.4rem}
.cc-pill{font-size:.57rem;font-weight:700;letter-spacing:.12em;text-transform:uppercase;padding:.15rem .48rem;border-radius:99px;background:rgba(0,212,255,.09);color:var(--c-cyan);border:1px solid rgba(0,212,255,.18)}
.dp-wrap{background:var(--surface);border:1px solid var(--border);border-radius:16px;padding:1.1rem 1.1rem .6rem;backdrop-filter:blur(20px);margin-bottom:.85rem}
.dp-title{font-family:'Syne',sans-serif;font-size:.85rem;font-weight:700;color:var(--txt1);display:flex;align-items:center;gap:.4rem}
.dp-badge{font-size:.63rem;font-weight:600;padding:.2rem .55rem;border-radius:99px;background:rgba(0,229,160,.09);color:var(--c-green);border:1px solid rgba(0,229,160,.2)}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#030810 0%,#060f1d 100%) !important;border-right:1px solid var(--border) !important}
[data-testid="stSidebarContent"]{background:transparent !important;padding:1.1rem .95rem !important}
.sb-brand{text-align:center;padding:.65rem 0 1.1rem;border-bottom:1px solid var(--border);margin-bottom:1.15rem}
.sb-ico{font-size:1.9rem;display:block;animation:sbRock 3.5s ease-in-out infinite}
@keyframes sbRock{0%,100%{transform:rotate(0) translateY(0)}50%{transform:rotate(9deg) translateY(-3px)}}
.sb-name{font-family:'Syne',sans-serif;font-size:.92rem;font-weight:800;background:linear-gradient(90deg,#00D4FF,#7B61FF);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;display:block;margin-top:.28rem}
.sb-sub{font-size:.57rem;color:var(--txt3);letter-spacing:.22em;text-transform:uppercase;display:block;margin-top:.12rem}
.sb-lbl{font-size:.6rem;font-weight:600;letter-spacing:.18em;text-transform:uppercase;color:var(--txt3);margin:.8rem 0 .28rem;display:block}
.active-tab-bar{margin:.5rem 0 1.2rem;padding:.55rem .9rem;background:rgba(0,212,255,.05);border:1px solid rgba(0,212,255,.15);border-radius:10px;font-size:.8rem;color:var(--c-cyan);font-family:'Syne',sans-serif;font-weight:700}
.metric-card{background:var(--surface);border:1px solid var(--border);border-radius:14px;padding:.85rem 1rem;margin-bottom:.5rem;transition:transform .25s,border-color .25s}
.metric-card:hover{transform:translateX(4px);border-color:var(--mc,rgba(0,212,255,.3))}
.metric-card-icon{font-size:1.2rem;margin-bottom:.3rem}
.metric-card-val{font-family:'Syne',sans-serif;font-size:1.3rem;font-weight:800;display:block}
.metric-card-label{font-size:.7rem;font-weight:700;text-transform:uppercase;letter-spacing:.08em;color:var(--txt1);display:block;margin:.1rem 0}
.metric-card-desc{font-size:.68rem;color:var(--txt2);line-height:1.5}
.best-badge{display:inline-block;padding:.12rem .5rem;border-radius:99px;background:rgba(0,229,160,.15);color:var(--c-green);border:1px solid rgba(0,229,160,.3);font-size:.6rem;font-weight:700;letter-spacing:.1em;margin-left:.4rem}
.model-info-card{background:var(--surface);border:1px solid rgba(0,229,160,.2);border-left:3px solid var(--c-green);border-radius:14px;padding:1rem 1.1rem;margin-bottom:1rem}
.model-info-h{font-family:'Syne',sans-serif;font-size:.85rem;font-weight:700;color:var(--c-green);margin:0 0 .55rem}
.model-info-row{display:flex;justify-content:space-between;padding:.22rem 0;border-bottom:1px solid rgba(255,255,255,.04);font-size:.76rem}
.model-info-row:last-child{border-bottom:none}
.model-info-k{color:var(--txt2)}.model-info-v{color:var(--txt1);font-weight:500;word-break:break-all;text-align:right;max-width:60%}
.pred-result{border-radius:18px;padding:1.6rem 2rem;text-align:center;margin-top:1rem}
.pred-result.yes{background:linear-gradient(135deg,rgba(0,229,160,.09),rgba(0,212,255,.06));border:1px solid rgba(0,229,160,.3)}
.pred-result.no{background:linear-gradient(135deg,rgba(255,60,172,.09),rgba(255,181,71,.06));border:1px solid rgba(255,60,172,.3)}
.pred-ico{font-size:2.8rem;display:block;margin-bottom:.5rem}
.pred-label{font-family:'Syne',sans-serif;font-size:1.5rem;font-weight:800;display:block;margin-bottom:.25rem}
.pred-desc{font-size:.8rem;color:var(--txt2)}
.prob-bar-wrap{margin:.4rem 0;display:flex;align-items:center;gap:.7rem;font-size:.76rem}
.prob-bar-track{flex:1;height:7px;background:rgba(255,255,255,.07);border-radius:99px;overflow:hidden}
.prob-bar-fill{height:100%;border-radius:99px;transition:width .8s cubic-bezier(.16,1,.3,1)}
.prob-label{min-width:60px;color:var(--txt2)}.prob-val{min-width:42px;text-align:right;font-weight:600;font-family:'Syne',sans-serif}
.chart-ctrl{background:var(--surface);border:1px solid var(--border);border-radius:12px;padding:.9rem 1rem;margin-bottom:.8rem}
.fin-card{background:var(--surface);border:1px solid var(--border);border-radius:14px;padding:.9rem 1.05rem;margin-bottom:.6rem;display:flex;gap:.75rem;align-items:flex-start;position:relative;overflow:hidden;cursor:default;transition:transform .28s,border-color .28s,box-shadow .28s}
.fin-card::before{content:'';position:absolute;left:0;top:0;bottom:0;width:3px;background:var(--fc,#00D4FF);border-radius:14px 0 0 14px}
.fin-card:hover{transform:translateX(5px);border-color:var(--fc,#00D4FF)}
.fin-ico{font-size:1.05rem;min-width:34px;height:34px;display:flex;align-items:center;justify-content:center;border-radius:8px;background:var(--fc-bg,rgba(0,212,255,.08));flex-shrink:0}
.fin-h{font-family:'Syne',sans-serif;font-size:.82rem;font-weight:700;color:var(--txt1);margin:0 0 .18rem}
.fin-p{font-size:.76rem;color:var(--txt2);line-height:1.58;margin:0}
.q-card{background:var(--surface);border:1px solid var(--border);border-top:2px solid var(--qc,#00D4FF);border-radius:14px;padding:1rem 1.1rem;margin-bottom:.65rem;transition:transform .28s,box-shadow .28s}
.q-card:hover{transform:translateY(-3px);box-shadow:0 12px 30px var(--qglow,rgba(0,212,255,.1))}
.q-header{display:flex;align-items:center;gap:.55rem;margin-bottom:.55rem}
.q-badge{font-family:'Syne',sans-serif;font-size:.68rem;font-weight:700;padding:.18rem .55rem;border-radius:99px;background:var(--qbg,rgba(0,212,255,.1));color:var(--qc,#00D4FF);border:1px solid var(--qborder,rgba(0,212,255,.2));letter-spacing:.05em}
.q-title{font-family:'Syne',sans-serif;font-size:.85rem;font-weight:700;color:var(--txt1)}
.q-item{display:flex;align-items:center;gap:.5rem;padding:.28rem 0;border-bottom:1px solid rgba(255,255,255,.04);font-size:.77rem;color:var(--txt2);transition:color .2s,transform .2s;cursor:default}
.q-item:last-child{border-bottom:none}
.q-item:hover{color:var(--txt1);transform:translateX(3px)}
.q-arrow{font-size:.7rem;color:var(--qc,#00D4FF);flex-shrink:0}
.stat-counters{display:grid;grid-template-columns:repeat(4,1fr);gap:.7rem;margin-bottom:1.2rem}
.sc{background:rgba(9,15,30,0.92);border:1px solid rgba(255,255,255,.07);border-radius:13px;padding:.85rem .9rem;text-align:center;position:relative;overflow:hidden;transition:transform .28s,box-shadow .28s;cursor:default}
.sc:hover{transform:translateY(-3px)}
.sc::before{content:'';position:absolute;bottom:0;left:0;right:0;height:2px;background:var(--sc-bar,linear-gradient(90deg,#00D4FF,transparent))}
.sc-val{font-family:'Syne',sans-serif;font-size:1.7rem;font-weight:800;display:block;line-height:1}
.sc-lbl{font-size:.66rem;color:#4A5A6E;margin-top:.3rem;display:block;text-transform:uppercase;letter-spacing:.08em}
div[data-baseweb="select"]>div{background:rgba(255,255,255,.03) !important;border:1px solid var(--border) !important;border-radius:10px !important}
.stDataFrame{border-radius:12px !important;overflow:hidden !important}
.stDataFrame thead th{background:rgba(0,212,255,.05) !important;color:var(--c-cyan) !important;font-family:'Syne',sans-serif !important;font-size:.72rem !important;font-weight:700 !important;letter-spacing:.1em !important;text-transform:uppercase !important}
hr{border:none !important;border-top:1px solid var(--border) !important;margin:1rem 0 !important}
div[data-baseweb="tab-list"]{background:rgba(255,255,255,.02) !important;border-radius:11px !important;padding:4px !important;border:1px solid var(--border) !important;gap:3px !important;margin-bottom:1.05rem !important}
button[data-baseweb="tab"]{background:transparent !important;border-radius:8px !important;color:var(--txt2) !important;font-weight:500 !important;font-size:.8rem !important;padding:.4rem .88rem !important}
button[aria-selected="true"][data-baseweb="tab"]{background:linear-gradient(135deg,rgba(0,212,255,.13),rgba(123,97,255,.13)) !important;color:var(--c-cyan) !important;font-weight:700 !important;border:1px solid rgba(0,212,255,.18) !important}
.target-banner{background:linear-gradient(135deg,rgba(0,212,255,.07),rgba(123,97,255,.07));border:1px solid rgba(0,212,255,.2);border-left:4px solid #00D4FF;border-radius:14px;padding:1rem 1.3rem;margin-bottom:1.2rem}
.target-banner-title{font-family:'Syne',sans-serif;font-size:.95rem;font-weight:800;color:#00D4FF;margin:0 0 .3rem;display:flex;align-items:center;gap:.5rem}
.target-banner-body{font-size:.8rem;color:#8A9AB8;line-height:1.65;margin:0}
.imbalance-chip{display:inline-block;padding:.18rem .6rem;border-radius:99px;background:rgba(255,60,172,.12);color:#FF3CAC;border:1px solid rgba(255,60,172,.3);font-size:.65rem;font-weight:700;letter-spacing:.1em;margin-left:.5rem}
/* ── Pipeline status strip ── */
.pipeline-strip{display:flex;gap:0;margin-bottom:1.2rem;border:1px solid rgba(255,255,255,.07);border-radius:10px;overflow:hidden;background:rgba(255,255,255,.02)}
.ps{flex:1;padding:.6rem .9rem;font-size:.75rem;font-weight:600;border-right:1px solid rgba(255,255,255,.07);display:flex;align-items:center;gap:.4rem;color:#2E4060}
.ps:last-child{border-right:none}
.ps.done{color:#00E5A0;background:rgba(0,229,160,.04)}
.ps.active{color:#00D4FF;background:rgba(0,212,255,.06)}
/* ── tuned badge ── */
.tuned-badge{display:inline-block;padding:.14rem .55rem;border-radius:99px;background:rgba(0,229,160,.13);color:#00E5A0;border:1px solid rgba(0,229,160,.3);font-size:.65rem;font-weight:700;letter-spacing:.1em;vertical-align:middle;margin-left:.35rem}
.base-badge{display:inline-block;padding:.14rem .55rem;border-radius:99px;background:rgba(0,212,255,.1);color:#00D4FF;border:1px solid rgba(0,212,255,.25);font-size:.65rem;font-weight:700;letter-spacing:.1em;vertical-align:middle;margin-left:.35rem}
@media(max-width:1150px){.kpi-grid{grid-template-columns:repeat(3,1fr)}.stat-counters{grid-template-columns:repeat(2,1fr)}}
@media(max-width:760px){.kpi-grid{grid-template-columns:repeat(2,1fr)}}
@media(max-width:480px){.kpi-grid{grid-template-columns:1fr}}
</style>
<div class="orb orb-a"></div><div class="orb orb-b"></div><div class="orb orb-c"></div>
""", unsafe_allow_html=True)

# ── helpers ──────────────────────────────────────────────────────
def card(label, pill=None):
    ph = f"<span class='cc-pill'>{pill}</span>" if pill else ""
    st.markdown(f"""<div class="cc"><p class="cc-lbl">{label} {ph}</p>""", unsafe_allow_html=True)
    c = st.container()
    st.markdown("</div>", unsafe_allow_html=True)
    return c

def _insight_panel(ins_list, rec_list):
    if not (ins_list or rec_list):
        return
    col_left, col_right = st.columns(2, gap="medium")
    with col_left:
        st.markdown("""<div style="background:rgba(0,212,255,0.04);border:1px solid
            rgba(0,212,255,0.15);border-left:3px solid #00D4FF;border-radius:14px;padding:1rem 1.1rem">
            <p style="font-family:'Syne',sans-serif;font-size:.82rem;font-weight:700;
            color:#00D4FF;margin:0 0 .6rem">🔍 Key Insights</p>""", unsafe_allow_html=True)
        for ico, txt in ins_list:
            st.markdown(f"""<div style="display:flex;gap:.5rem;align-items:flex-start;
                padding:.3rem 0;border-bottom:1px solid rgba(255,255,255,.04);font-size:.76rem;
                color:#8A9AB8;line-height:1.55"><span style="flex-shrink:0">{ico}</span>
                <span>{txt}</span></div>""", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with col_right:
        st.markdown("""<div style="background:rgba(0,229,160,0.04);border:1px solid
            rgba(0,229,160,0.15);border-left:3px solid #00E5A0;border-radius:14px;padding:1rem 1.1rem">
            <p style="font-family:'Syne',sans-serif;font-size:.82rem;font-weight:700;
            color:#00E5A0;margin:0 0 .6rem">💡 Recommendations</p>""", unsafe_allow_html=True)
        for ico, txt in rec_list:
            st.markdown(f"""<div style="display:flex;gap:.5rem;align-items:flex-start;
                padding:.3rem 0;border-bottom:1px solid rgba(255,255,255,.04);font-size:.76rem;
                color:#8A9AB8;line-height:1.55"><span style="flex-shrink:0">{ico}</span>
                <span>{txt}</span></div>""", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

def _locked_state(icon, title, body):
    st.markdown(f"""<div style="text-align:center;padding:3.5rem 1rem;color:var(--txt2)">
      <div style="font-size:3rem;margin-bottom:.8rem">{icon}</div>
      <div style="font-family:'Syne',sans-serif;font-size:.95rem;font-weight:700;
        color:var(--txt1);margin-bottom:.5rem">{title}</div>
      <div style="font-size:.82rem;max-width:380px;margin:0 auto;line-height:1.6">{body}</div>
    </div>""", unsafe_allow_html=True)

def _pipeline_strip(base_done: bool, tuned_done: bool):
    """Visual status bar showing which pipeline stages are complete."""
    s1 = "done" if base_done  else "active"
    s2 = "done" if tuned_done else ("active" if base_done else "")
    s3 = "done" if (base_done or tuned_done) else ""
    s4 = "done" if (base_done or tuned_done) else ""
    st.markdown(f"""<div class="pipeline-strip">
      <div class="ps {s1}">{"✅" if base_done else "🔄"} Stage 1 · Baseline Training</div>
      <div class="ps {s2}">{"✅" if tuned_done else "⚙️"} Stage 2 · Hyperparameter Tuning</div>
      <div class="ps {s3}">{"✅" if (base_done or tuned_done) else "🔒"} Model Analysis</div>
      <div class="ps {s4}">{"✅" if (base_done or tuned_done) else "🔒"} Predict</div>
    </div>""", unsafe_allow_html=True)

# ── data ─────────────────────────────────────────────────────────
@st.cache_data
def _get_data(): return ana.load_cleaned()
df = _get_data()

# ── session state ────────────────────────────────────────────────
DEFAULTS = {
    "page"          : "overview",
    "analysis_tab"  : "univariate",
    "train_result"  : None,   # Stage-1 result dict (in memory)
    "tune_result"   : None,   # Stage-2 result dict (in memory)
}
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── auto-load saved artefacts on every script rerun ──────────────
# Guard directly on file existence — avoids the old _prod gate that
# could silently swallow load errors and leave session_state None.

if st.session_state.train_result is None and pred.base_artefacts_exist():
    try:
        _base_auto = pred.load_base_artefacts()
        if _base_auto:
            _meta = _base_auto["meta"]
            st.session_state.train_result = {
                "best_model"    : _base_auto["model"],
                "best_name"     : _meta["best_name"],
                "models"        : {_meta["best_name"]: _base_auto["model"]},
                "results_df"    : _meta.get("results_df", pd.DataFrame()),
                "scaler"        : _base_auto["scaler"],
                "le_dict"       : _base_auto["le_dict"],
                "feature_names" : _meta.get("feature_names", _base_auto.get("feature_names", [])),
                "X_test"        : _meta.get("X_test"),
                "X_test_sc"     : _meta.get("X_test_sc"),
                "best_X_test"   : _meta.get("best_X_test"),
                "y_test"        : _meta.get("y_test"),
                "balance_method": _meta.get("balance_method", "N/A"),
                "before_balance": _meta.get("before_balance", {}),
                "after_balance" : _meta.get("after_balance", {}),
                "stage"         : _meta.get("stage", "baseline"),
                "tuned"         : False,
                "X_train_raw"   : _meta.get("X_train_raw"),
                "y_train_raw"   : _meta.get("y_train_raw"),
            }
    except Exception:
        pass  # silently skip — user sees "train first" prompt

if st.session_state.tune_result is None and pred.tuned_artefacts_exist():
    try:
        _tuned_auto = pred.load_tuned_artefacts()
        if _tuned_auto:
            _tmeta = _tuned_auto["meta"]
            st.session_state.tune_result = {
                "tuned_model"        : _tuned_auto["model"],
                "best_name"          : _tmeta["best_name"],
                "best_params"        : _tmeta.get("best_params", {}),
                "best_cv_f1"         : _tmeta.get("best_cv_f1"),
                "scaler"             : _tuned_auto["scaler"],
                "le_dict"            : _tuned_auto["le_dict"],
                "feature_names"      : _tmeta.get("feature_names", _tuned_auto.get("feature_names", [])),
                "X_test"             : _tmeta.get("X_test"),
                "X_test_sc"          : _tmeta.get("X_test_sc"),
                "best_X_test"        : _tmeta.get("best_X_test"),
                "y_test"             : _tmeta.get("y_test"),
                "tuning_search_space": _tmeta.get("tuning_search_space", {}),
                "comparison"         : _tmeta.get("comparison", {}),
                "baseline_metrics"   : _tmeta.get("baseline_metrics", {}),
                "tuned_metrics"      : _tmeta.get("tuned_metrics", {}),
                "balance_method"     : _tmeta.get("balance_method", "N/A"),
                "before_balance"     : _tmeta.get("before_balance", {}),
                "after_balance"      : _tmeta.get("after_balance", {}),
                "stage"              : "tuned",
                "tuned"              : True,
            }
    except Exception:
        pass  # silently skip

# ── helpers to get current production model info ──────────────────
def _get_prod_model():
    """Returns (model, name, scaler, le_dict, feature_names, is_tuned)."""
    if st.session_state.tune_result:
        r = st.session_state.tune_result
        return (r["tuned_model"], r["best_name"], r["scaler"],
                r["le_dict"], r["feature_names"], True)
    if st.session_state.train_result:
        r = st.session_state.train_result
        return (r["best_model"], r["best_name"], r["scaler"],
                r["le_dict"], r["feature_names"], False)
    return (None, None, None, None, None, False)

# ══════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""<div class="sb-brand"><span class="sb-ico">✈</span>
    <span class="sb-name">Trips &amp; Travel</span>
    <span class="sb-sub">Analytics Platform</span></div>""", unsafe_allow_html=True)

    st.markdown("<span class='sb-lbl'>Navigation</span>", unsafe_allow_html=True)
    for key, ico, label in [("overview","🏠","Overview"), ("analysis","📊","Analysis"), ("prediction","🤖","Prediction")]:
        if st.button(f"{ico}  {label}", key=f"nav_{key}", width='stretch'):
            st.session_state.page = key; st.rerun()

    st.markdown("<hr>", unsafe_allow_html=True)
    if st.session_state.page in ("overview", "analysis"):
        st.markdown("<span class='sb-lbl'>Filters</span>", unsafe_allow_html=True)
        gender_f = st.multiselect("Gender", df["Gender"].unique(), default=list(df["Gender"].unique()), label_visibility="collapsed")
        st.markdown("<span class='sb-lbl'>Gender</span>", unsafe_allow_html=True)
        city_f   = st.multiselect("City Tier", sorted(df["CityTier"].unique()), default=list(df["CityTier"].unique()), label_visibility="collapsed")
        st.markdown("<span class='sb-lbl'>City Tier</span>", unsafe_allow_html=True)
        occ_f    = st.multiselect("Occupation", df["Occupation"].unique(), default=list(df["Occupation"].unique()), label_visibility="collapsed")
        st.markdown("<span class='sb-lbl'>Occupation</span>", unsafe_allow_html=True)
        imin, imax = int(df["MonthlyIncome"].min()), int(df["MonthlyIncome"].max())
        inc_r = st.slider("Income", imin, imax, (imin, imax), step=1000, label_visibility="collapsed")
        st.markdown("<span class='sb-lbl'>Monthly Income (₹)</span>", unsafe_allow_html=True)
    else:
        gender_f = list(df["Gender"].unique())
        city_f   = list(df["CityTier"].unique())
        occ_f    = list(df["Occupation"].unique())
        imin, imax = int(df["MonthlyIncome"].min()), int(df["MonthlyIncome"].max())
        inc_r = (imin, imax)

    # ── Pipeline status in sidebar ────────────────────────────────
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("<span class='sb-lbl'>ML Pipeline</span>", unsafe_allow_html=True)

    _base_done  = st.session_state.train_result is not None
    _tuned_done = st.session_state.tune_result  is not None
    _, _active_name, _, _, _, _is_tuned = _get_prod_model()

    if _active_name:
        _badge = "<span class='tuned-badge'>TUNED ✓</span>" if _is_tuned else "<span class='base-badge'>BASELINE</span>"
        st.markdown(f"""<div style="background:rgba(0,229,160,.06);border:1px solid rgba(0,229,160,.25);
            border-radius:10px;padding:.55rem .75rem;font-size:.76rem;color:#00E5A0;font-weight:600">
            🏆 Active Model<br>
            <span style="font-family:'Syne',sans-serif;font-size:.85rem;font-weight:800">
            {_active_name}</span>{_badge}</div>""", unsafe_allow_html=True)

        _s1_clr = "#00E5A0" if _base_done  else "#2E4060"
        _s2_clr = "#00E5A0" if _tuned_done else "#2E4060"
        st.markdown(f"""<div style="margin-top:.5rem;display:grid;grid-template-columns:1fr 1fr;gap:.4rem">
          <div style="background:rgba(0,0,0,.3);border:1px solid rgba(255,255,255,.06);border-radius:8px;
            padding:.4rem .6rem;text-align:center;font-size:.65rem;color:{_s1_clr};font-weight:700">
            {"✅" if _base_done else "○"} Stage 1<br>Baseline</div>
          <div style="background:rgba(0,0,0,.3);border:1px solid rgba(255,255,255,.06);border-radius:8px;
            padding:.4rem .6rem;text-align:center;font-size:.65rem;color:{_s2_clr};font-weight:700">
            {"✅" if _tuned_done else "○"} Stage 2<br>Tuned</div>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown("""<div style="background:rgba(255,181,71,.06);border:1px solid rgba(255,181,71,.2);
            border-radius:10px;padding:.55rem .75rem;font-size:.75rem;color:#FFB547">
            ⚠️ No model trained yet<br>
            <span style="font-size:.68rem;color:#4A5A6E">Go to Prediction → Train All Models</span>
            </div>""", unsafe_allow_html=True)

# ── filter ───────────────────────────────────────────────────────
dff  = df[df["Gender"].isin(gender_f) & df["CityTier"].isin(city_f) &
          df["Occupation"].isin(occ_f) &
          (df["MonthlyIncome"].between(inc_r[0], inc_r[1]) | df["MonthlyIncome"].isna())]
kpis = ana.compute_kpis(dff)

# ══════════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════
if st.session_state.page == "overview":
    st.markdown("""<div class="pg-eye">✦ Marketing Analytics Dashboard</div>
    <h1 class="pg-h1">Wellness Tourism <span>Intelligence</span></h1>
    <p class="pg-sub">Real-time customer behaviour &amp; conversion insights for strategic growth</p>
    <div class="pg-rule"></div>""", unsafe_allow_html=True)

    st.markdown(f"""<div class="kpi-grid">
      <div class="kpi kc"><div class="kpi-ic">👤</div>
        <span class="kpi-val" style="color:#00D4FF;-webkit-text-fill-color:#00D4FF">{kpis['total']:,}</span>
        <span class="kpi-lbl">Total Customers</span><span class="kpi-sub">in filtered view</span></div>
      <div class="kpi kg"><div class="kpi-ic">🎯</div>
        <span class="kpi-val" style="color:#00E5A0;-webkit-text-fill-color:#00E5A0">{kpis['purchased']:,}</span>
        <span class="kpi-lbl">ProdTaken = 1</span><span class="kpi-sub">purchased travel package</span></div>
      <div class="kpi kv"><div class="kpi-ic">📈</div>
        <span class="kpi-val" style="color:#7B61FF;-webkit-text-fill-color:#7B61FF">{kpis['conv_rate']:.1f}%</span>
        <span class="kpi-lbl">Conversion Rate</span><span class="kpi-sub">target = 1 ratio</span></div>
      <div class="kpi ka"><div class="kpi-ic">💰</div>
        <span class="kpi-val" style="color:#FFB547;-webkit-text-fill-color:#FFB547">₹{kpis['avg_inc']:,.0f}</span>
        <span class="kpi-lbl">Avg Monthly Income</span><span class="kpi-sub">mean across segment</span></div>
      <div class="kpi kr"><div class="kpi-ic">🛂</div>
        <span class="kpi-val" style="color:#FF3CAC;-webkit-text-fill-color:#FF3CAC">{kpis['pasp_pct']:.1f}%</span>
        <span class="kpi-lbl">Passport Holders</span><span class="kpi-sub">travel ready</span></div>
    </div>""", unsafe_allow_html=True)

    c1, c2 = st.columns(2, gap="medium")
    with c1:
        with card("🎯 ProdTaken — Target Variable Distribution", "Binary"):
            st.plotly_chart(ana.fig_purchase_donut(dff), width='stretch', key="plotly_chart_1")
    with c2:
        with card("📦 Products Pitched", "Inventory"):
            st.plotly_chart(ana.fig_products_bar(dff), width='stretch', key="plotly_chart_2")
    c3, c4, c5 = st.columns(3, gap="medium")
    with c3:
        with card("👤 Age Distribution"):
            st.plotly_chart(ana.fig_age_histogram(dff), width='stretch', key="plotly_chart_3")
    with c4:
        with card("⚧ Gender Split"):
            st.plotly_chart(ana.fig_gender_pie(dff), width='stretch', key="plotly_chart_4")
    with c5:
        with card("🏙 City Tier"):
            st.plotly_chart(ana.fig_city_tier_bar(dff), width='stretch', key="plotly_chart_5")

    cl, cr = st.columns([3, 1])
    with cl:
        all_cols = dff.columns.tolist()
        sel_cols = st.multiselect("cols", options=all_cols, default=all_cols[:8], label_visibility="collapsed")
    with cr:
        n_rows = st.select_slider("rows", options=[10, 25, 50, 100], value=10, label_visibility="collapsed")
    st.dataframe((dff[sel_cols] if sel_cols else dff).head(n_rows), width='stretch', height=280)

    n0 = int((dff["ProdTaken"] == 0).sum())
    n1 = int((dff["ProdTaken"] == 1).sum())
    total_tv = n0 + n1
    pct1 = round(n1 / max(total_tv, 1) * 100, 1)
    pct0 = round(n0 / max(total_tv, 1) * 100, 1)
    ratio = round(n0 / max(n1, 1), 2)
    st.markdown(f"""<div class="target-banner" style="padding:.75rem 1.1rem">
      <div style="display:flex;align-items:center;gap:.7rem;flex-wrap:wrap">
        <div class="target-banner-title" style="margin:0">🎯 Target Variable:
          <code style="background:rgba(0,212,255,.12);padding:.1rem .4rem;border-radius:5px;font-size:.82rem">ProdTaken</code>
          <span class="imbalance-chip">⚠️ Class Imbalance {ratio}:1</span>
        </div>
        <div style="display:flex;align-items:center;gap:1.2rem;margin-left:auto;flex-wrap:wrap">
          <div style="display:flex;align-items:center;gap:.45rem;font-size:.78rem">
            <div style="width:10px;height:10px;border-radius:50%;background:#00E5A0;flex-shrink:0"></div>
            <span style="color:#00E5A0;font-weight:700">{n1:,}</span>
            <span style="color:#6A7D9C">Purchased ({pct1}%)</span>
          </div>
          <div style="display:flex;align-items:center;gap:.45rem;font-size:.78rem">
            <div style="width:10px;height:10px;border-radius:50%;background:#FF3CAC;flex-shrink:0"></div>
            <span style="color:#FF3CAC;font-weight:700">{n0:,}</span>
            <span style="color:#6A7D9C">Not Purchased ({pct0}%)</span>
          </div>
        </div>
      </div>
    </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# PAGE 2 — ANALYSIS
# ══════════════════════════════════════════════════════════════════
elif st.session_state.page == "analysis":
    st.markdown("""<div class="pg-eye">✦ Deep Analysis Suite</div>
    <h1 class="pg-h1">Customer <span>Analysis</span></h1>
    <p class="pg-sub">Interactive univariate, bivariate, and strategic insight explorer</p>
    <div class="pg-rule"></div>""", unsafe_allow_html=True)

    btncols = st.columns(3, gap="small")
    for col, (key, label) in zip(btncols, [("univariate","📊 Univariate"), ("bivariate","🔗 Bivariate"), ("insights","💡 Insights")]):
        with col:
            if st.button(label, key=f"tab_{key}", width='stretch'):
                st.session_state.analysis_tab = key; st.rerun()

    anames = {"univariate":"📊 Univariate Analysis", "bivariate":"🔗 Bivariate Analysis", "insights":"💡 Insights & Recommendations"}
    st.markdown(f"""<div class="active-tab-bar">▶ Active: {anames[st.session_state.analysis_tab]}</div>""", unsafe_allow_html=True)

    # ═══ UNIVARIATE ═══
    if st.session_state.analysis_tab == "univariate":
        tab_num, tab_cat, tab_stats = st.tabs(["📊 Numeric","🏷 Categorical","📋 Stats"])
        with tab_num:
            num_cols = ana.get_numeric_cols(dff)
            st.markdown("<div class='chart-ctrl'>", unsafe_allow_html=True)
            cc1, cc2, cc3 = st.columns([2, 2, 1])
            with cc1: sel_col   = st.selectbox("📌 Select Column", num_cols, key="uni_num_col")
            with cc2: sel_chart = st.selectbox("📊 Chart Type", ana.UNIVARIATE_NUM_CHARTS, key="uni_num_chart")
            with cc3:
                col_colors = {"Age":"#7B61FF","MonthlyIncome":"#FFB547","DurationOfPitch":"#00D4FF","NumberOfTrips":"#FF3CAC","NumberOfFollowups":"#00E5A0"}
                chart_color = st.color_picker("Color", col_colors.get(sel_col, "#7B61FF"), key="uni_num_color")
            st.markdown("</div>", unsafe_allow_html=True)
            if sel_col:
                with card(f"📊 {sel_col} — {sel_chart}"):
                    st.plotly_chart(ana.fig_univariate(dff, sel_col, sel_chart, chart_color), width='stretch', key="plotly_chart_6")
                    s = ana.get_numeric_summary(dff, sel_col)
                    for col_s, (k, v) in zip(st.columns(6), s.items()):
                        with col_s: st.metric(k.capitalize(), v)
                st.markdown("---")
                st.markdown("#### 📊 Insights & Recommendations")
                _insight_panel(*ana.get_univariate_insights(dff, sel_col).values())

        with tab_cat:
            cat_cols = ana.get_categorical_cols(dff)
            st.markdown("<div class='chart-ctrl'>", unsafe_allow_html=True)
            cc1, cc2, cc3, cc4 = st.columns([2, 2, 1, 1])
            with cc1: sel_col   = st.selectbox("📌 Select Column", cat_cols, key="uni_cat_col")
            with cc2: sel_chart = st.selectbox("📊 Chart Type", ana.UNIVARIATE_CAT_CHARTS, key="uni_cat_chart")
            with cc3: start_color = st.color_picker("Start Color", "#0D1E38", key="start_color")
            with cc4: end_color   = st.color_picker("End Color", "#00D4FF", key="end_color")
            st.markdown("</div>", unsafe_allow_html=True)
            if sel_col:
                with card(f"🏷 {sel_col} — {sel_chart}"):
                    kwargs = {"color": end_color, "start_color": start_color} if sel_chart in ["Bar Chart","Horizontal Bar"] else {"color": end_color}
                    st.plotly_chart(ana.fig_univariate(dff, sel_col, sel_chart, **kwargs), width='stretch', key="plotly_chart_7")
                vc = dff[sel_col].value_counts()
                st.markdown(f"""<div style="display:flex;flex-wrap:wrap;gap:.6rem;margin-top:.3rem">
                {''.join(f'<span style="font-size:.75rem;background:rgba(0,212,255,.07);border:1px solid rgba(0,212,255,.15);padding:.18rem .55rem;border-radius:99px;color:#EEF2FF"><strong>{v}</strong> <span style="color:#6A7D9C">{k}</span></span>' for k, v in vc.items())}
                </div>""", unsafe_allow_html=True)
                st.markdown("---")
                _insight_panel(*ana.get_univariate_insights(dff, sel_col).values())

        with tab_stats:
            with card("📋 Descriptive Statistics", "Full Dataset"):
                st.dataframe(dff.describe().round(2), width='stretch')

    # ═══ BIVARIATE ═══
    elif st.session_state.analysis_tab == "bivariate":
        biv_tab1, biv_tab2 = st.tabs(["🔧 Custom Explorer","📌 Pre-built Charts"])
        with biv_tab1:
            all_cols = ana.get_all_cols(dff)
            cat_cols = ana.get_categorical_cols(dff)
            st.markdown("<div class='chart-ctrl'>", unsafe_allow_html=True)
            bcc1, bcc2, bcc3, bcc4 = st.columns([2, 2, 2, 1])
            with bcc1: x_col      = st.selectbox("X Axis", all_cols, index=all_cols.index("MonthlyIncome") if "MonthlyIncome" in all_cols else 0, key="biv_x")
            with bcc2: y_col      = st.selectbox("Y Axis", all_cols, index=all_cols.index("ProdTaken") if "ProdTaken" in all_cols else 1, key="biv_y")
            with bcc3: chart_type = st.selectbox("📊 Chart Type", ana.BIVARIATE_CHART_TYPES, key="biv_chart")
            with bcc4: hue_col    = st.selectbox("🎨 Colour by", ["None"]+cat_cols, key="biv_hue"); hue_col = None if hue_col == "None" else hue_col
            st.markdown("</div>", unsafe_allow_html=True)
            with card(f"🔗 {x_col}  ×  {y_col}", chart_type):
                if x_col == y_col: st.warning("X and Y columns are the same.")
                else: st.plotly_chart(ana.fig_bivariate(dff, x_col, y_col, chart_type, hue_col), width='stretch', key="plotly_chart_8")
            if pd.api.types.is_numeric_dtype(dff[x_col]) and pd.api.types.is_numeric_dtype(dff[y_col]):
                corr_val = dff[[x_col, y_col]].corr().iloc[0, 1]
                strength = "strong" if abs(corr_val) > .6 else "moderate" if abs(corr_val) > .3 else "weak"
                direction = "positive" if corr_val > 0 else "negative"
                st.markdown(f"""<div style="font-size:.75rem;color:var(--txt2);margin-top:-.4rem;padding:.4rem .8rem;background:rgba(0,212,255,.04);border-radius:8px;border:1px solid rgba(0,212,255,.1)">
                    📐 Pearson correlation: <strong style="color:{'#00E5A0' if abs(corr_val)>.3 else '#6A7D9C'}">{corr_val:.3f}</strong> — {strength} {direction} relationship</div>""", unsafe_allow_html=True)
            if x_col != y_col:
                _insight_panel(*ana.get_bivariate_insights(dff, x_col, y_col, chart_type).values())

        with biv_tab2:
            c1, c2 = st.columns(2, gap="medium")
            with c1:
                with card("💰 Income vs Conversion"): st.plotly_chart(ana.fig_income_vs_conversion(dff), width='stretch', key="plotly_chart_9")
            with c2:
                with card("👶 Age Group vs Conversion"): st.plotly_chart(ana.fig_age_group_conversion(dff), width='stretch', key="plotly_chart_10")
            c3, c4 = st.columns(2, gap="medium")
            with c3:
                with card("⚧ Gender vs Conversion"): st.plotly_chart(ana.fig_gender_conversion(dff), width='stretch', key="plotly_chart_11")
            with c4:
                with card("🏙 City Tier vs Conversion"): st.plotly_chart(ana.fig_city_tier_conversion(dff), width='stretch', key="plotly_chart_12")
            c5, c6 = st.columns(2, gap="medium")
            with c5:
                with card("🛂 Passport vs Conversion"): st.plotly_chart(ana.fig_passport_conversion(dff), width='stretch', key="plotly_chart_13")
            with c6:
                with card("⭐ Pitch Satisfaction vs Conversion"): st.plotly_chart(ana.fig_pitch_satisfaction_conversion(dff), width='stretch', key="plotly_chart_14")
            with card("🔥 Correlation Matrix", "Numeric"):
                st.plotly_chart(ana.fig_correlation_heatmap(dff), width='stretch', key="plotly_chart_15")

    # ═══ INSIGHTS ═══
    elif st.session_state.analysis_tab == "insights":
        stats = ana.compute_insight_stats(dff)
        st.markdown(f"""<div class="stat-counters">
          <div class="sc" style="--sc-bar:linear-gradient(90deg,#00D4FF,transparent)"><span class="sc-val" style="color:#00D4FF;-webkit-text-fill-color:#00D4FF">{stats['conv_rate']}%</span><span class="sc-lbl">Conversion Rate</span></div>
          <div class="sc" style="--sc-bar:linear-gradient(90deg,#00E5A0,transparent)"><span class="sc-val" style="color:#00E5A0;-webkit-text-fill-color:#00E5A0">{stats['passport_uplift']}×</span><span class="sc-lbl">Passport Uplift</span></div>
          <div class="sc" style="--sc-bar:linear-gradient(90deg,#7B61FF,transparent)"><span class="sc-val" style="color:#7B61FF;-webkit-text-fill-color:#7B61FF">{stats['income_uplift']}×</span><span class="sc-lbl">Income Uplift</span></div>
          <div class="sc" style="--sc-bar:linear-gradient(90deg,#FFB547,transparent)"><span class="sc-val" style="color:#FFB547;-webkit-text-fill-color:#FFB547">{stats['top_age_segment']}</span><span class="sc-lbl">Top Age Segment</span></div>
        </div>""", unsafe_allow_html=True)
        col_l, col_r = st.columns([1.1, 1], gap="large")
        with col_l:
            st.markdown("""<p style="font-family:'Syne',sans-serif;font-size:.85rem;font-weight:700;color:#EEF2FF;margin-bottom:.7rem">💡 Key Findings</p>""", unsafe_allow_html=True)
            for clr, bg, ico, ttl, body in [
                ("#00D4FF","rgba(0,212,255,.08)","🎯","High-Value Age Bracket","Customers aged 26–35 show the highest conversion rates."),
                ("#7B61FF","rgba(123,97,255,.08)","💰","Income Threshold Effect","A clear inflection at ₹35,000/month — purchase probability increases ~2.4×."),
                ("#FF3CAC","rgba(255,60,172,.08)","⚧","Gender Dynamics","Female customers exhibit marginally higher conversion."),
                ("#FFB547","rgba(255,181,71,.08)","🛂","Passport Advantage","Passport holders are significantly more likely to purchase."),
                ("#00E5A0","rgba(0,229,160,.08)","🏙","Tier-2 City Opportunity","Tier-2 cities show growing intent with lower saturation."),
            ]:
                st.markdown(f"""<div class="fin-card" style="--fc:{clr};--fc-bg:{bg}"><div class="fin-ico">{ico}</div><div><p class="fin-h">{ttl}</p><p class="fin-p">{body}</p></div></div>""", unsafe_allow_html=True)
        with col_r:
            st.markdown("""<p style="font-family:'Syne',sans-serif;font-size:.85rem;font-weight:700;color:#EEF2FF;margin-bottom:.7rem">🚀 Action Playbook</p>""", unsafe_allow_html=True)
            for clr, bg, border, glow, q, ttl, items in [
                ("#00D4FF","rgba(0,212,255,.09)","rgba(0,212,255,.18)","rgba(0,212,255,.1)","Q1","Segment &amp; Target",["Build income micro-segments","Launch email flows for 26–35","A/B test wellness vs adventure messaging"]),
                ("#7B61FF","rgba(123,97,255,.09)","rgba(123,97,255,.2)","rgba(123,97,255,.1)","Q2","Channel Expansion",["Invest in Tier-2 digital","Referral incentives for passport holders","Partner with travel influencers"]),
                ("#FF3CAC","rgba(255,60,172,.09)","rgba(255,60,172,.2)","rgba(255,60,172,.1)","Q3","Retention &amp; Upsell",["Loyalty tiers for repeat purchasers","Cross-sell premium packages","30-day post-purchase engagement"]),
            ]:
                bullets = "".join([f"<div class='q-item'><span class='q-arrow'>→</span>{i}</div>" for i in items])
                st.markdown(f"""<div class="q-card" style="--qc:{clr};--qbg:{bg};--qborder:{border};--qglow:{glow}"><div class="q-header"><span class="q-badge">{q}</span><span class="q-title">{ttl}</span></div>{bullets}</div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# PAGE 3 — PREDICTION  (4 tabs: Train | Tune | Analyse | Predict)
# ══════════════════════════════════════════════════════════════════
elif st.session_state.page == "prediction":
    st.markdown("""<div class="pg-eye">✦ 2-Stage ML Pipeline</div>
    <h1 class="pg-h1">Customer <span>Prediction</span></h1>
    <p class="pg-sub">Stage 1: baseline multi-model training · Stage 2: single-model hyperparameter tuning · Analysis · Predict</p>
    <div class="pg-rule"></div>""", unsafe_allow_html=True)

    _base_done  = st.session_state.train_result is not None
    _tuned_done = st.session_state.tune_result  is not None
    _pipeline_strip(_base_done, _tuned_done)

    tab1, tab2, tab3, tab4 = st.tabs([
        "🏋️ Train All Models",
        "⚙️ Hyperparameter Tuning",
        "📊 Model Analysis",
        "🎯 Predict",
    ])

    # ══════════════════════════════════════════════════════════════
    # TAB 1 — TRAIN ALL MODELS  (Stage 1: baseline only)
    # ══════════════════════════════════════════════════════════════
    with tab1:
        st.markdown("""<div style="background:rgba(0,212,255,.04);border:1px solid rgba(0,212,255,.15);
            border-radius:14px;padding:1rem 1.2rem;margin-bottom:1rem">
            <p style="font-family:'Syne',sans-serif;font-size:.88rem;font-weight:700;color:#00D4FF;margin:0 0 .4rem">
            🔧 Stage 1 — Baseline Training Pipeline</p>
            <p style="font-size:.77rem;color:#8A9AB8;margin:0;line-height:1.7">
            Trains <strong style="color:#EEF2FF">all models with default (baseline) parameters</strong> — no tuning yet.
            Applies <strong style="color:#00E5A0">BorderlineSMOTE + Tomek Links</strong> to fix the 4.31:1 class imbalance.
            CV scores are computed leakage-safe (balancing inside each fold). Produces a leaderboard and
            <strong style="color:#FFB547">identifies the best base learner</strong> by F1 score.<br>
            <strong style="color:#FF3CAC">↳ Hyperparameter Tuning (Tab 2) will then tune only that winner.</strong></p>
        </div>""", unsafe_allow_html=True)

        # Models info row
        n_models = 5 + (1 if pred.HAS_XGB else 0)
        _model_names = (["Logistic Regression", "Decision Tree", "Random Forest",
                          "Gradient Boosting", "AdaBoost"] + (["XGBoost"] if pred.HAS_XGB else []))
        _model_pills = "".join(
            '<span style="font-size:.72rem;background:rgba(0,212,255,.07);border:1px solid '
            'rgba(0,212,255,.15);padding:.2rem .55rem;border-radius:99px;color:#8A9AB8">'
            '<strong style="color:#EEF2FF">' + m + '</strong></span>'
            for m in _model_names
        )
        st.markdown(
            '<div style="display:flex;gap:.6rem;flex-wrap:wrap;margin-bottom:1rem">'
            + _model_pills
            + '<span style="font-size:.72rem;background:rgba(0,229,160,.07);border:1px solid '
            'rgba(0,229,160,.15);padding:.2rem .55rem;border-radius:99px;color:#00E5A0">'
            + str(n_models) + ' models · default params · balanced training</span></div>',
            unsafe_allow_html=True
        )

        if st.button("🚀 Train All Models  (Baseline — No Tuning)", width='stretch', type="primary"):
            prog_bar  = st.progress(0)
            prog_text = st.empty()
            steps = (["Logistic Regression","Decision Tree","Random Forest","Gradient Boosting","AdaBoost"]
                     + (["XGBoost"] if pred.HAS_XGB else []))
            step_count = [0]
            def _prog_cb(name):
                step_count[0] += 1
                prog_bar.progress(min(step_count[0] / len(steps), 0.95))
                prog_text.markdown(f"<span style='color:var(--txt2);font-size:.8rem'>🔄 Training {name} (baseline)…</span>", unsafe_allow_html=True)

            with st.spinner("Balancing + baseline training…"):
                try:
                    result = pred.train_all_models(ana.load_cleaned(), progress_cb=_prog_cb)
                    pred.save_base_artefacts(result)
                    st.session_state.train_result = result
                    # Clear any stale tuning result if model winner changed
                    if (st.session_state.tune_result and
                            st.session_state.tune_result.get("best_name") != result["best_name"]):
                        st.session_state.tune_result = None
                    prog_bar.progress(1.0); prog_text.empty()
                    best_row = result["results_df"].iloc[0]
                    st.success(
                        f"✅ Stage 1 complete!  Best base learner: **{result['best_name']}**  |  "
                        f"F1: {best_row['F1']}  |  ROC-AUC: {best_row['ROC-AUC']}  |  PR-AUC: {best_row['PR-AUC']}"
                    )
                    st.info("💾 Saved to disk. Go to **Hyperparameter Tuning** tab to tune the winner.")
                except Exception as e:
                    prog_bar.empty(); prog_text.empty()
                    st.error(f"Training failed: {e}")

        # ── Results ──
        if st.session_state.train_result:
            res = st.session_state.train_result
            best_name  = res["best_name"]
            results_df = res["results_df"]

            st.markdown("---")

            # Best model banner
            best_row = results_df.iloc[0]
            _banner_metrics = "".join(
                '<div style="text-align:center">'
                '<div style="font-family:\'Syne\',sans-serif;font-size:1.1rem;font-weight:800;color:#00D4FF">'
                + str(best_row.get(m, "—")) +
                '</div><div style="font-size:.65rem;color:#6A7D9C;text-transform:uppercase;letter-spacing:.08em">'
                + m + '</div></div>'
                for m in ["F1", "ROC-AUC", "PR-AUC", "Balanced Accuracy"]
            )
            st.markdown(
                '<div style="background:linear-gradient(135deg,rgba(0,229,160,.07),rgba(0,212,255,.05));'
                'border:1px solid rgba(0,229,160,.25);border-left:4px solid #00E5A0;border-radius:14px;'
                'padding:.9rem 1.2rem;margin-bottom:1rem;display:flex;align-items:center;gap:1rem;flex-wrap:wrap">'
                '<div>'
                '<div style="font-family:\'Syne\',sans-serif;font-size:.78rem;font-weight:700;color:#00E5A0;margin-bottom:.2rem">'
                '🏆 Best Base Learner — ready for Stage 2 tuning</div>'
                '<div style="font-family:\'Syne\',sans-serif;font-size:1.3rem;font-weight:800;color:#EEF2FF">'
                + best_name +
                '</div></div>'
                '<div style="display:flex;gap:1rem;margin-left:auto;flex-wrap:wrap">'
                + _banner_metrics +
                '</div></div>',
                unsafe_allow_html=True
            )

            # Balancing results
            bef = res.get("before_balance", {}); aft = res.get("after_balance", {}); bm = res.get("balance_method", "N/A")
            c_bal1, c_bal2 = st.columns([1, 1], gap="medium")
            with c_bal1:
                with card("⚖️ Class Balancing Results", bm):
                    st.plotly_chart(pred.fig_balance_pie(bef, aft, bm), width='stretch', key="balance_pie_train")
            with c_bal2:
                st.markdown(f"""<div style="background:rgba(0,229,160,.04);border:1px solid rgba(0,229,160,.15);border-left:3px solid #00E5A0;border-radius:14px;padding:1rem 1.1rem;margin-top:.85rem">
                    <p style="font-family:'Syne',sans-serif;font-size:.82rem;font-weight:700;color:#00E5A0;margin:0 0 .6rem">🧪 Balancing Details</p>
                    <div style="font-size:.76rem;color:#8A9AB8;line-height:1.75">
                    <div><span style="color:#6A7D9C">Method:</span>&nbsp;<strong style="color:#00D4FF">{bm}</strong></div>
                    <div><span style="color:#6A7D9C">Before:</span>&nbsp;<span style="color:#FF3CAC">Class 0 = {bef.get(0,"?")}</span>&nbsp;/&nbsp;<span style="color:#00E5A0">Class 1 = {bef.get(1,"?")}</span></div>
                    <div><span style="color:#6A7D9C">After:</span>&nbsp;<span style="color:#FF3CAC">Class 0 = {aft.get(0,"?")}</span>&nbsp;/&nbsp;<span style="color:#00E5A0">Class 1 = {aft.get(1,"?")}</span></div>
                    </div>
                    <div style="margin-top:.5rem;font-size:.73rem;color:#6A7D9C;line-height:1.6">{pred.BALANCE_RATIONALE}</div>
                </div>""", unsafe_allow_html=True)

            # Leaderboard
            with card("📊 Baseline Model Leaderboard", "Ranked by F1"):
                display_cols = [c for c in ["Model","F1","ROC-AUC","PR-AUC","Accuracy","Balanced Accuracy",
                                            "Precision","Recall","MCC","Cohen Kappa","Log Loss","Brier Score",
                                            "CV ROC-AUC","CV Std"] if c in results_df.columns]
                styled = results_df[display_cols].style.highlight_max(
                    subset=[c for c in ["F1","ROC-AUC","PR-AUC","Accuracy","Balanced Accuracy","Precision","Recall","MCC","Cohen Kappa","CV ROC-AUC"] if c in display_cols],
                    color="rgba(0,229,160,.18)"
                ).highlight_min(
                    subset=[c for c in ["Log Loss","Brier Score","CV Std"] if c in display_cols],
                    color="rgba(0,229,160,.18)")
                st.dataframe(styled, width='stretch', height=260)

            # Comparison chart
            with card("📊 Baseline Model Comparison"):
                st.plotly_chart(pred.fig_model_comparison(results_df), width='stretch', key="plotly_chart_16")

            with st.expander("📖 Which metric should I trust?", expanded=False):
                st.markdown(pred.BEST_METRIC_NOTE)
                m_cols = st.columns(4)
                for i, (k, (ico, clr, desc)) in enumerate(pred.METRIC_INFO.items()):
                    best_tag = '<span class="best-badge">★ BEST</span>' if k in ("F1","PR-AUC") else ""
                    with m_cols[i % 4]:
                        st.markdown(f"""<div class="metric-card" style="--mc:{clr}">
                        <div class="metric-card-icon">{ico}</div>
                        <span class="metric-card-label">{k}{best_tag}</span>
                        <span class="metric-card-desc">{desc}</span></div>""", unsafe_allow_html=True)

            # Artefact status
            st.markdown("---")
            st.markdown("""<p style="font-family:'Syne',sans-serif;font-size:.82rem;font-weight:700;color:var(--txt1);margin-bottom:.5rem">📁 Saved Artefacts</p>""", unsafe_allow_html=True)
            art_cols = st.columns(5)
            for ac, fname, desc in zip(art_cols,
                ["base_model.pkl","scaler.pkl","feature_names.pkl","label_encoders.pkl","base_meta.pkl"],
                ["Best baseline model","StandardScaler","Feature column list","LabelEncoders","Stage-1 metadata"]):
                with ac:
                    exists = pred.base_artefacts_exist()
                    clr = "#00E5A0" if exists else "#FF3CAC"
                    ico = "✅" if exists else "❌"
                    st.markdown(f"""<div style="background:rgba(255,255,255,.03);border:1px solid rgba(255,255,255,.08);border-radius:10px;padding:.6rem .7rem;text-align:center">
                        <div style="font-size:1.1rem">{ico}</div>
                        <div style="font-size:.68rem;font-weight:700;color:{clr};margin:.2rem 0">{fname}</div>
                        <div style="font-size:.62rem;color:#4A5A6E">{desc}</div>
                    </div>""", unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════
    # TAB 2 — HYPERPARAMETER TUNING  (Stage 2: tune the winner)
    # ══════════════════════════════════════════════════════════════
    with tab2:
        if not _base_done:
            _locked_state("⚙️", "Run Stage 1 First",
                          "Go to <strong>Train All Models</strong> tab to train and identify the best base learner before tuning.")
        else:
            _base_res   = st.session_state.train_result
            best_name   = _base_res["best_name"]
            base_row    = _base_res["results_df"].iloc[0].to_dict()

            st.markdown(f"""<div class="active-tab-bar">⚙️ Stage 2 — Tuning: {best_name}
                <span class="base-badge">STAGE 1 WINNER</span></div>""", unsafe_allow_html=True)

            # Explainer + controls
            c_info, c_ctrl = st.columns([1.4, 1], gap="medium")
            with c_info:
                st.markdown(f"""<div style="background:rgba(123,97,255,.05);border:1px solid rgba(123,97,255,.16);border-radius:14px;padding:1rem 1.1rem;margin-bottom:1rem">
                    <p style="font-family:'Syne',sans-serif;font-size:.84rem;font-weight:700;color:#7B61FF;margin:0 0 .5rem">🎯 What happens here?</p>
                    <div style="font-size:.76rem;color:#8A9AB8;line-height:1.7">
                    <strong style="color:#EEF2FF">Only {best_name}</strong> is tuned — the Stage-1 winner.
                    <code>RandomizedSearchCV</code> with <strong>StratifiedKFold (k=5)</strong> searches for
                    better hyperparameters using <strong>F1 scoring</strong>. BorderlineSMOTE + Tomek Links
                    runs <em>inside each CV fold</em> to prevent data leakage. The best configuration is then
                    refit on the full balanced training set and evaluated on the same held-out test set.
                    </div></div>""", unsafe_allow_html=True)
            with c_ctrl:
                tune_iters = st.slider("🔍 Search iterations (RandomizedSearchCV)", 10, 50, 25, 5, key="tune_iters_v2")
                st.markdown(f"""<div style="font-size:.73rem;color:#6A7D9C;line-height:1.6;margin-top:.4rem;padding:.5rem .75rem;background:rgba(123,97,255,.04);border:1px solid rgba(123,97,255,.14);border-radius:8px">
                  {tune_iters} candidate configs × 5 CV folds = <strong style="color:#EEF2FF">{tune_iters*5} fits</strong><br>
                  Scoring: <strong style="color:#7B61FF">F1</strong> · n_jobs=-1 (parallel)
                </div>""", unsafe_allow_html=True)

            # Base model stats
            st.markdown(f"""<div class="model-info-card">
              <p class="model-info-h">📊 {best_name} — Baseline Metrics (pre-tuning)</p>
              {"".join(f'<div class="model-info-row"><span class="model-info-k">{k}</span><span class="model-info-v">{base_row.get(k,"—")}</span></div>' for k in ["F1","ROC-AUC","PR-AUC","Balanced Accuracy","Recall","Precision"])}
            </div>""", unsafe_allow_html=True)

            if st.button(f"⚙️ Tune {best_name}", width='stretch', type="primary"):
                if _base_res.get("X_train_raw") is None:
                    st.error("⚠️ Training data not available. Please re-run **Train All Models** with the updated prediction.py to store train data in memory.")
                else:
                    tune_prog = st.empty()
                    def _tune_cb(msg):
                        tune_prog.markdown(f"<span style='color:var(--txt2);font-size:.8rem'>🔄 {msg}</span>", unsafe_allow_html=True)
                    with st.spinner(f"Tuning {best_name} — this may take a moment…"):
                        try:
                            tune_res = pred.tune_best_model(_base_res, tune_iters=tune_iters, progress_cb=_tune_cb)
                            pred.save_tuned_artefacts(tune_res)
                            st.session_state.tune_result = tune_res
                            tune_prog.empty()
                            tm = tune_res["tuned_metrics"]
                            st.success(
                                f"✅ Stage 2 complete! Tuned {best_name}  |  "
                                f"F1: {tm.get('F1','—')}  |  Recall: {tm.get('Recall','—')}  |  ROC-AUC: {tm.get('ROC-AUC','—')}"
                            )
                        except Exception as e:
                            tune_prog.empty()
                            st.error(f"Tuning failed: {e}")

            # ── Tuning results ──
            _tun = st.session_state.tune_result
            if _tun and _tun.get("best_name") == best_name:
                st.markdown("---")
                search_space = _tun.get("tuning_search_space", {})
                comparison   = _tun.get("comparison", {})
                best_params  = _tun.get("best_params", {})
                best_cv_f1   = _tun.get("best_cv_f1")

                # Best params table
                if best_params:
                    c_p1, c_p2 = st.columns([1, 1.5], gap="medium")
                    with c_p1:
                        with card(f"🎛 Best Hyperparameters — {best_name}", "Tuned"):
                            params_df = pd.DataFrame([{"Parameter": k, "Best Value": str(v)} for k, v in best_params.items()])
                            st.dataframe(params_df, width='stretch', hide_index=True)
                            if isinstance(best_cv_f1, float):
                                st.markdown(f"""<div style="margin-top:.5rem;padding:.5rem .75rem;background:rgba(0,229,160,.06);border:1px solid rgba(0,229,160,.2);border-radius:8px;font-size:.77rem;color:#00E5A0;font-weight:600">
                                  🏆 Best CV F1 (during search): <strong>{best_cv_f1:.4f}</strong></div>""", unsafe_allow_html=True)
                    with c_p2:
                        if search_space:
                            with card("📊 Best Parameter Profile"):
                                st.plotly_chart(pred.fig_best_params_bar(search_space, best_name), width='stretch', key="plotly_chart_17")

                # Search space heatmap
                if search_space:
                    with card("🗺 Search Space Heatmap", "★ = selected"):
                        st.plotly_chart(pred.fig_tuning_search_space(search_space, best_name), width='stretch', key="plotly_chart_18")

                # Comparison charts
                if comparison:
                    c_cmp1, c_cmp2 = st.columns(2, gap="medium")
                    with c_cmp1:
                        with card("📈 Baseline vs Tuned"):
                            st.plotly_chart(pred.fig_tuning_comparison(comparison, best_name), width='stretch', key="plotly_chart_19")
                    with c_cmp2:
                        with card("📉 Metric Delta (Tuned − Baseline)"):
                            st.plotly_chart(pred.fig_tuning_delta(comparison, best_name), width='stretch', key="plotly_chart_20")

                    with st.expander("📋 Tuning Comparison Table", expanded=False):
                        st.dataframe(pred.get_tuning_comparison(comparison), width='stretch', hide_index=True)

                # Tuned artefacts status
                st.markdown("---")
                st.markdown("""<p style="font-family:'Syne',sans-serif;font-size:.82rem;font-weight:700;color:var(--txt1);margin-bottom:.5rem">📁 Tuned Artefacts</p>""", unsafe_allow_html=True)
                art_cols2 = st.columns(3)
                for ac, fname, desc in zip(art_cols2,
                    ["tuned_model.pkl","tuned_meta.pkl","(scaler/encoders shared)"],
                    ["Tuned model object","Stage-2 metadata + params","Re-used from Stage 1"]):
                    with ac:
                        exists = pred.tuned_artefacts_exist() if "tuned_meta" in fname or "tuned_model" in fname else True
                        clr = "#00E5A0" if exists else "#FF3CAC"
                        ico = "✅" if exists else "❌"
                        st.markdown(f"""<div style="background:rgba(255,255,255,.03);border:1px solid rgba(255,255,255,.08);border-radius:10px;padding:.6rem .7rem;text-align:center">
                            <div style="font-size:1.1rem">{ico}</div>
                            <div style="font-size:.68rem;font-weight:700;color:{clr};margin:.2rem 0">{fname}</div>
                            <div style="font-size:.62rem;color:#4A5A6E">{desc}</div>
                        </div>""", unsafe_allow_html=True)
            elif not _tun:
                st.info("Run the tuner above to see results. The tuned model will automatically be used for Model Analysis and Predict.")

    # ══════════════════════════════════════════════════════════════
    # TAB 3 — MODEL ANALYSIS  (active = tuned > base)
    # ══════════════════════════════════════════════════════════════
    with tab3:
        if not _base_done:
            _locked_state("🔒", "No Trained Model Found",
                          "Go to <strong>Train All Models</strong> tab to train a model first.")
        else:
            # Prefer tuned model; fall back to base winner
            _tun = st.session_state.tune_result
            _bas = st.session_state.train_result

            if _tun:
                analysis_model   = _tun["tuned_model"]
                analysis_name    = _tun["best_name"]
                analysis_Xtest   = _tun["best_X_test"]
                analysis_ytest   = _tun["y_test"]
                analysis_feats   = _tun["feature_names"]
                analysis_row     = _tun["tuned_metrics"]
                analysis_Xtest_r = _tun["X_test"]
                analysis_Xtsc    = _tun["X_test_sc"]
                analysis_bef     = _tun["before_balance"]
                analysis_aft     = _tun["after_balance"]
                analysis_bm      = _tun["balance_method"]
                stage_label      = f"<span class='tuned-badge'>TUNED ✓</span>"
                models_dict      = {analysis_name: analysis_model}
                results_df_src   = _bas["results_df"]   # leaderboard still from baseline
            else:
                analysis_model   = _bas["best_model"]
                analysis_name    = _bas["best_name"]
                analysis_Xtest   = _bas["best_X_test"]
                analysis_ytest   = _bas["y_test"]
                analysis_feats   = _bas["feature_names"]
                analysis_row     = _bas["results_df"].iloc[0].to_dict()
                analysis_Xtest_r = _bas["X_test"]
                analysis_Xtsc    = _bas["X_test_sc"]
                analysis_bef     = _bas.get("before_balance", {})
                analysis_aft     = _bas.get("after_balance",  {})
                analysis_bm      = _bas.get("balance_method", "N/A")
                stage_label      = "<span class='base-badge'>BASELINE</span>"
                models_dict      = _bas.get("models", {analysis_name: analysis_model})
                results_df_src   = _bas["results_df"]

            st.markdown(f"""<div class="active-tab-bar">📊 Analysing: {analysis_name} {stage_label}</div>""",
                        unsafe_allow_html=True)

            if not _tun:
                st.info("💡 Tip: Run **Hyperparameter Tuning** (Tab 2) to analyse the tuned production model instead of the baseline.")

            # Metric cards
            primary_m = ["Accuracy","Balanced Accuracy","Precision","Recall","F1","ROC-AUC","PR-AUC","MCC"]
            mc = st.columns(4)
            for i, k in enumerate(primary_m):
                v = analysis_row.get(k)
                if v is None: continue
                ico, clr, desc = pred.METRIC_INFO.get(k, ("📊","#00D4FF",""))
                best_tag = '<span class="best-badge">★ BEST</span>' if k in ("F1","PR-AUC") else ""
                with mc[i % 4]:
                    st.markdown(f"""<div class="metric-card" style="--mc:{clr}">
                    <div class="metric-card-icon">{ico}</div>
                    <span class="metric-card-val" style="color:{clr};-webkit-text-fill-color:{clr}">{v}</span>
                    <span class="metric-card-label">{k}{best_tag}</span>
                    <span class="metric-card-desc">{desc}</span></div>""", unsafe_allow_html=True)

            # Class balancing chart
            with card(f"⚖️ Class Distribution Before vs After Balancing", analysis_bm):
                st.plotly_chart(pred.fig_balance_pie(analysis_bef, analysis_aft, analysis_bm),
                                width='stretch', key="balance_pie_analysis")

            # Main charts
            c1, c2 = st.columns(2, gap="medium")
            with c1:
                with card("📈 Baseline Model Comparison"):
                    st.plotly_chart(pred.fig_model_comparison(results_df_src), width='stretch', key="plotly_chart_22")
            with c2:
                with card("📉 ROC Curves (all baseline models)"):
                    st.plotly_chart(pred.fig_roc_curves(models_dict, analysis_Xtest_r, analysis_ytest, analysis_Xtsc),
                                    width='stretch', key="roc_curves_analysis")
            c3, c4 = st.columns(2, gap="medium")
            with c3:
                with card(f"🧮 Confusion Matrix — {analysis_name} {stage_label if _tun else ''}"):
                    st.plotly_chart(pred.fig_confusion_matrix(analysis_model, analysis_Xtest, analysis_ytest, analysis_name),
                                    width='stretch', key="confusion_matrix_analysis")
            with c4:
                with card(f"📉 Precision-Recall Curve — {analysis_name}"):
                    st.plotly_chart(pred.fig_pr_curve(analysis_model, analysis_Xtest, analysis_ytest, analysis_name),
                                    width='stretch', key="pr_curve_analysis")
            c5, c6 = st.columns(2, gap="medium")
            with c5:
                with card(f"🕸 Metric Radar — {analysis_name}"):
                    st.plotly_chart(pred.fig_metrics_radar(analysis_row, analysis_name), width='stretch', key="plotly_chart_26")
            with c6:
                with card("📊 Probability Distribution"):
                    st.plotly_chart(pred.fig_prob_distribution(analysis_model, analysis_Xtest, analysis_ytest),
                                    width='stretch', key="prob_distribution_analysis")

            fi_fig = pred.fig_feature_importance(analysis_model, analysis_feats, analysis_name)
            if fi_fig:
                with card("🔍 Feature Importances"):
                    st.plotly_chart(fi_fig, width='stretch', key="plotly_chart_28")

            with st.expander("📖 Metric Guide", expanded=False):
                st.markdown(pred.BEST_METRIC_NOTE)

    # ══════════════════════════════════════════════════════════════
    # TAB 4 — PREDICT  (uses production model: tuned > base)
    # ══════════════════════════════════════════════════════════════
    with tab4:
        active_model, active_name, active_scaler, active_le, active_feats, is_tuned = _get_prod_model()

        if active_model is None:
            _locked_state("🔒", "No Model Available",
                          "Train a model on the <strong>Train All Models</strong> tab first.")
        else:
            prod_badge = "<span class='tuned-badge'>TUNED ✓</span>" if is_tuned else "<span class='base-badge'>BASELINE</span>"
            st.markdown(f"""<div class="active-tab-bar">🤖 Production Model: {active_name} {prod_badge}</div>""",
                        unsafe_allow_html=True)

            if not is_tuned:
                st.info("💡 Tip: Run **Hyperparameter Tuning** (Tab 2) to upgrade to the tuned production model.")

            predict_mode = st.radio("Prediction Mode", ["👤 Single Customer","📂 Batch CSV Upload"], horizontal=True)

            # ── SINGLE ──────────────────────────────────────────────
            if predict_mode == "👤 Single Customer":
                st.markdown("""<div style="background:var(--surface);border:1px solid var(--border);
                    border-radius:16px;padding:1.2rem 1.1rem;margin-bottom:1rem">
                    <p style="font-family:'Syne',sans-serif;font-size:.85rem;font-weight:700;
                    color:var(--txt1);margin:0 0 1rem;display:flex;align-items:center;gap:.4rem">
                    🧾 Manual Customer Input</p>""", unsafe_allow_html=True)

                r1 = st.columns(3)
                with r1[0]: age    = st.number_input("Age", 18, 70, 30, 1)
                with r1[1]: gender = st.selectbox("Gender", df["Gender"].unique().tolist())
                with r1[2]: marital = st.selectbox("Marital Status", df["MaritalStatus"].unique().tolist() if "MaritalStatus" in df.columns else ["Single","Married","Divorced"])
                r2 = st.columns(3)
                with r2[0]: income     = st.number_input("Monthly Income (₹)", 10000, 100000, 35000, 1000)
                with r2[1]: occupation = st.selectbox("Occupation", df["Occupation"].unique().tolist())
                with r2[2]: city_tier  = st.selectbox("City Tier", sorted(df["CityTier"].unique().tolist()))
                r3 = st.columns(3)
                with r3[0]: passport   = st.selectbox("Passport", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
                with r3[1]: num_trips  = st.number_input("No. of Trips", 0, 22, int(df["NumberOfTrips"].median()) if "NumberOfTrips" in df.columns else 3)
                with r3[2]: pitch_sat  = st.slider("Pitch Satisfaction", 1, 5, 3)
                r4 = st.columns(3)
                with r4[0]: type_contact     = st.selectbox("Type of Contact", df["TypeofContact"].unique().tolist() if "TypeofContact" in df.columns else ["Self Enquiry","Company Invited"])
                with r4[1]: product_pitched  = st.selectbox("Product Pitched", df["ProductPitched"].unique().tolist() if "ProductPitched" in df.columns else ["Basic","Standard","Deluxe","Super Deluxe","King"])
                with r4[2]: designation      = st.selectbox("Designation", df["Designation"].unique().tolist() if "Designation" in df.columns else ["Executive","Manager","Senior Manager","AVP","VP"])
                r5 = st.columns(3)
                with r5[0]: num_person   = st.number_input("Persons Visiting", 1, 10, 2)
                with r5[1]: num_followups = st.number_input("Follow-ups", 0, 10, 3)
                with r5[2]: num_children = st.number_input("Children Visiting", 0, 5, 1)
                r6 = st.columns(3)
                with r6[0]: prop_star = st.slider("Preferred Property Star", 1, 5, 3)
                with r6[1]: own_car   = st.selectbox("Own Car", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
                with r6[2]: duration  = st.number_input("Pitch Duration (min)", 5, 60, 15)
                st.markdown("</div>", unsafe_allow_html=True)

                if st.button("🎯 Predict Conversion", width='stretch', type="primary"):
                    input_dict = {
                        "Age": age, "MonthlyIncome": income, "Passport": passport,
                        "NumberOfTrips": num_trips, "PitchSatisfactionScore": pitch_sat,
                        "CityTier": city_tier, "Gender": gender, "Occupation": occupation,
                        "MaritalStatus": marital, "ProductPitched": product_pitched,
                        "Designation": designation, "TypeofContact": type_contact,
                        "NumberOfPersonVisiting": num_person, "NumberOfFollowups": num_followups,
                        "NumberOfChildrenVisiting": num_children, "PreferredPropertyStar": prop_star,
                        "OwnCar": own_car, "DurationOfPitch": duration,
                    }
                    try:
                        result = pred.predict_single(active_model, input_dict, df,
                                                     le_dict=active_le,
                                                     scaler=active_scaler,
                                                     feature_names=active_feats)
                        p = result["prediction"]; py = result["proba_yes"]; pn = result["proba_no"]
                        if p == 1:
                            st.markdown(f"""<div class="pred-result yes">
                            <span class="pred-ico">✅</span>
                            <span class="pred-label" style="color:#00E5A0;-webkit-text-fill-color:#00E5A0">Will Purchase</span>
                            <span class="pred-desc">High likelihood to convert — prioritise for follow-up</span>
                            </div>""", unsafe_allow_html=True)
                        else:
                            st.markdown(f"""<div class="pred-result no">
                            <span class="pred-ico">❌</span>
                            <span class="pred-label" style="color:#FF3CAC;-webkit-text-fill-color:#FF3CAC">Will Not Purchase</span>
                            <span class="pred-desc">Low probability — consider a nurture campaign</span>
                            </div>""", unsafe_allow_html=True)

                        st.markdown(f"""<div style="margin-top:1rem;background:var(--surface);
                            border:1px solid var(--border);border-radius:14px;padding:1rem 1.1rem">
                            <p style="font-family:'Syne',sans-serif;font-size:.8rem;font-weight:700;
                            color:var(--txt1);margin:0 0 .7rem">📊 Prediction Confidence
                            <span style="font-size:.65rem;color:#6A7D9C;font-weight:400;margin-left:.5rem">
                            via {active_name} {("(tuned)" if is_tuned else "(baseline)")}</span></p>
                            <div class="prob-bar-wrap"><span class="prob-label">Will Buy</span>
                            <div class="prob-bar-track"><div class="prob-bar-fill"
                              style="width:{py*100:.1f}%;background:linear-gradient(90deg,#00E5A0,#00D4FF)">
                            </div></div><span class="prob-val" style="color:#00E5A0">{py*100:.1f}%</span></div>
                            <div class="prob-bar-wrap"><span class="prob-label">Won't Buy</span>
                            <div class="prob-bar-track"><div class="prob-bar-fill"
                              style="width:{pn*100:.1f}%;background:linear-gradient(90deg,#FF3CAC,#FFB547)">
                            </div></div><span class="prob-val" style="color:#FF3CAC">{pn*100:.1f}%</span></div>
                            </div>""", unsafe_allow_html=True)
                        with st.expander("📋 View Input Summary"):
                            st.dataframe(result["input_df"], width='stretch')
                    except Exception as e:
                        st.error(f"Prediction failed: {e}")
                        st.info("Tip: ensure the model was trained on the same dataset schema.")

            # ── BATCH ───────────────────────────────────────────────
            else:
                st.markdown("""<div style="background:var(--surface);border:1px solid var(--border);
                    border-radius:16px;padding:1.2rem 1.1rem;margin-bottom:1rem">
                    <p style="font-family:'Syne',sans-serif;font-size:.85rem;font-weight:700;
                    color:var(--txt1);margin:0 0 .5rem">📂 Upload Customer CSV</p>
                    <p style="font-size:.75rem;color:var(--txt2);margin:0 0 .8rem;line-height:1.6">
                    Upload a CSV with the same columns as training data (excluding ProdTaken).
                    The model will predict conversion probability for each row.</p>""", unsafe_allow_html=True)

                uploaded = st.file_uploader("Upload CSV", type=["csv"], label_visibility="collapsed")
                st.markdown("</div>", unsafe_allow_html=True)

                if uploaded:
                    try:
                        batch_df = pd.read_csv(uploaded)
                        st.markdown(f"<span style='font-size:.75rem;color:var(--txt2)'>📋 {len(batch_df):,} rows × {len(batch_df.columns)} columns</span>", unsafe_allow_html=True)
                        st.dataframe(batch_df.head(5), width='stretch')
                        if st.button("🎯 Run Batch Predictions", width='stretch', type="primary"):
                            with st.spinner("Running batch predictions…"):
                                try:
                                    out_df = pred.predict_batch(active_model, batch_df,
                                                                le_dict=active_le,
                                                                scaler=active_scaler,
                                                                feature_names=active_feats)
                                    n_yes  = int((out_df["Prediction"] == 1).sum())
                                    n_no   = int((out_df["Prediction"] == 0).sum())
                                    conv_r = round(n_yes / max(len(out_df), 1) * 100, 1)

                                    c_b1, c_b2, c_b3 = st.columns(3)
                                    with c_b1:
                                        st.markdown(f"""<div class="kpi kg"><div class="kpi-ic">✅</div>
                                          <span class="kpi-val" style="color:#00E5A0;-webkit-text-fill-color:#00E5A0">{n_yes:,}</span>
                                          <span class="kpi-lbl">Will Purchase</span></div>""", unsafe_allow_html=True)
                                    with c_b2:
                                        st.markdown(f"""<div class="kpi kr"><div class="kpi-ic">❌</div>
                                          <span class="kpi-val" style="color:#FF3CAC;-webkit-text-fill-color:#FF3CAC">{n_no:,}</span>
                                          <span class="kpi-lbl">Will Not Purchase</span></div>""", unsafe_allow_html=True)
                                    with c_b3:
                                        st.markdown(f"""<div class="kpi kv"><div class="kpi-ic">📈</div>
                                          <span class="kpi-val" style="color:#7B61FF;-webkit-text-fill-color:#7B61FF">{conv_r}%</span>
                                          <span class="kpi-lbl">Predicted Conversion</span></div>""", unsafe_allow_html=True)

                                    with card("📊 Batch Prediction Results"):
                                        st.dataframe(out_df, width='stretch', height=320)

                                    csv_bytes = out_df.to_csv(index=False).encode()
                                    st.download_button("⬇️ Download Predictions CSV", csv_bytes,
                                                       "predictions.csv", "text/csv",
                                                       width='stretch')
                                    .\ollama
                                    serve      except Exception as e:
                                    st.error(f"Batch prediction failed: {e}")
                    except Exception as e:
                        st.error(f"Could not read CSV: {e}")
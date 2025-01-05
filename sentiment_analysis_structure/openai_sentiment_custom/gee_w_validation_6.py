#!/usr/bin/env python3
# gee_brute_force_all_families_integrated.py
"""
Script that:
  1) Loads data from JSONL
  2) Chunks & saves CSV
  3) Performs correlation analyses & scatterplots for Quotation vs Fulltext
  4) Aggregates sentiment/emotion scores + bar plots
  5) Finds best QIC combination (over *all families* × correlation structures × scale options)
  6) For each best QIC model:
     - Refit & produce GEE summary
     - Compute pairwise comparisons (BH correction)
     - Build a "DiffIDs" table => if category i differs from j => j's index => "2,4,5"
  7) Writes summary, pairwise, & "DiffIDs" table into analysis_gee.xlsx
  8) Also saves analysis_main, analysis_raw, analysis_plots, analysis_combined

Families we brute-force:
  - Poisson
  - NegativeBinomial(alpha=1.0)
  - Gaussian
  - Gamma
  - InverseGaussian

Correlation structures:
  - Independence
  - Exchangeable

Scale options:
  - none
  - pearson
  - (others if you want: deviance, ub, bc, etc.)

Similar structure to your original code, but with enumerations over families.

DISCLAIMER:
 - If your data isn't valid for some families (e.g. negative or zero data for Gamma?), the fit may fail.
 - NegativeBinomial won't estimate alpha from the data, so alpha=1.0 is fixed.
"""

import json
import os
import re
import sys
import warnings
import logging
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

import statsmodels.api as sm
from statsmodels.genmod.families import (
    Poisson, NegativeBinomial,
    Gaussian, Gamma, InverseGaussian
)
from statsmodels.genmod.generalized_estimating_equations import GEE
from statsmodels.genmod.cov_struct import Independence, Exchangeable
from statsmodels.stats.multitest import multipletests
from scipy.stats import pearsonr, norm

# --------------------- #
# Configuration
# --------------------- #
INPUT_JSONL_FILE = "processed_all_articles_fixed_2.jsonl"

OUTPUT_DIR = "graphs_analysis"
os.makedirs(OUTPUT_DIR, exist_ok=True)

CSV_OUTPUT_DIR = "csv_raw_scores"
os.makedirs(CSV_OUTPUT_DIR, exist_ok=True)

OUTPUT_EXCEL_MAIN = "analysis_main.xlsx"
OUTPUT_EXCEL_RAW = "analysis_raw.xlsx"
OUTPUT_EXCEL_GEE = "analysis_gee.xlsx"
OUTPUT_EXCEL_PLOTS = "analysis_plots.xlsx"
OUTPUT_EXCEL_COMBINED = "analysis_combined.xlsx"

LOG_FILE = "analysis.log"

CATEGORIES = [
    "joy","sadness","anger","fear",
    "surprise","disgust","trust","anticipation",
    "negative_sentiment","positive_sentiment"
]

# For final "DiffIDs" or sorted plots, etc.
CLD_ORDER = ["Scientific","Left","Lean Left","Center","Lean Right","Right"]

MEDIA_CATEGORIES = {
    "Scientific": ["nature","sciam","stat","newscientist"],
    "Left": ["theatlantic","the daily beast","the intercept","mother jones","msnbc","slate","vox","huffpost"],
    "Lean Left": ["ap","axios","cnn","guardian","business insider","nbcnews","npr","nytimes","politico","propublica","wapo","usa today"],
    "Center": ["reuters","marketwatch","financial times","newsweek","forbes"],
    "Lean Right": ["thedispatch","epochtimes","foxbusiness","wsj","national review","washtimes"],
    "Right": ["breitbart","theblaze","daily mail","dailywire","foxnews","nypost","newsmax"],
}

def setup_logging():
    """Initialize logging to both file and console."""
    log_format = "%(asctime)s [%(levelname)s] %(module)s - %(message)s"
    logging.basicConfig(filename=LOG_FILE, level=logging.DEBUG, format=log_format)
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(console)
    logging.info("Logging initialized (file+console).")

warnings.filterwarnings(
    "ignore",
    message="QIC values obtained using scale=None are not appropriate for comparing models"
)

# ------------------------------------------------------
# 1) Load, chunk, basic stats
# ------------------------------------------------------
def load_jsonl(jsonl_file):
    logging.info(f"Loading JSONL from {jsonl_file}")
    records=[]
    with open(jsonl_file, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Loading JSONL data"):
            rec=json.loads(line)
            records.append(rec)
    df=pd.DataFrame(records)
    logging.debug(f"Loaded DataFrame shape={df.shape}")
    return df

def map_media_outlet_to_category(df):
    """Map media outlets to the categories in MEDIA_CATEGORIES."""
    cat_map={}
    for cat, outls in MEDIA_CATEGORIES.items():
        for o in outls:
            cat_map[o.lower().strip()]=cat

    if "media_outlet" not in df.columns:
        raise KeyError("'media_outlet' not found in data")

    df["media_outlet_clean"]=df["media_outlet"].str.lower().str.strip()
    df["media_category"]=df["media_outlet_clean"].map(cat_map).fillna("Other")

    unmapped=df[df["media_category"]=="Other"]["media_outlet"].unique()
    if len(unmapped)>0:
        logging.warning(f"Unmapped => {unmapped}")
        print(f"Warning: Not mapped => {unmapped}")
    return df

def chunk_and_save(df, chunk_size=20000):
    logging.info(f"Chunking DataFrame => len={len(df)}, chunk_size={chunk_size}")
    for i in range(0,len(df),chunk_size):
        part=df.iloc[i:i+chunk_size]
        out_csv=os.path.join(CSV_OUTPUT_DIR,f"raw_data_part_{(i//chunk_size)+1}.csv")
        part.to_csv(out_csv,index=False)
        print(f"Saved chunk {(i//chunk_size)+1} to {out_csv}")

def print_basic_stats(df):
    logging.info(f"Basic stats => total articles = {len(df)}")
    print("\nSummary Statistics:")
    print("Total number of articles:", len(df))
    if "media_outlet_clean" in df.columns:
        vc=df["media_outlet_clean"].value_counts()
        print("\nArticles per outlet:")
        print(vc)
    if "media_category" in df.columns:
        vc2=df["media_category"].value_counts()
        print("\nArticles per category:")
        print(vc2)
    print()

# ------------------------------------------------------
# 2) Quotation vs. Fulltext correlation
# ------------------------------------------------------
def analyze_quotation_fulltext_correlation(df):
    """
    1) For each (media_category, sentiment), compute Quotation_Average & Fulltext_Average
    2) Correlation, scatter, bar
    3) Combined z-scatter => 'quotation_fulltext_correlation.csv' and optionally charts
    """
    logging.info("Analyzing Quotation vs Fulltext correlation.")
    import re
    from scipy.stats import pearsonr

    records=[]
    for cat in df["media_category"].dropna().unique():
        dcat=df[df["media_category"]==cat]
        for s in CATEGORIES:
            pat=rf"^{s}_\d+$"
            matched=[c for c in dcat.columns if re.match(pat,c)]
            if matched:
                clp=dcat[matched].clip(lower=0)
                qsum=clp.sum(skipna=True).sum()
                qcount=clp.count().sum()
                qavg=qsum/qcount if qcount>0 else np.nan
            else:
                qavg=np.nan

            fcol=f"{s}_fulltext"
            if fcol in dcat.columns:
                fv=dcat[fcol].clip(lower=0)
                f_sum=fv.sum(skipna=True)
                f_cnt=fv.count()
                favg=f_sum/f_cnt if f_cnt>0 else np.nan
            else:
                favg=np.nan

            records.append({
                "MediaCategory": cat,
                "Sentiment": s,
                "Quotation_Average": qavg,
                "Fulltext_Average": favg
            })
    agg_df=pd.DataFrame(records)

    cor_results=[]
    for s in CATEGORIES:
        sub=agg_df[(agg_df["Sentiment"]==s)].dropna(subset=["Quotation_Average","Fulltext_Average"])
        if len(sub)>1:
            cor_val,_=pearsonr(sub["Quotation_Average"],sub["Fulltext_Average"])
        else:
            cor_val=np.nan
        cor_results.append({"Sentiment":s,"Correlation":cor_val})
    cor_df=pd.DataFrame(cor_results)

    # bar chart
    plt.figure(figsize=(8,5))
    sns.barplot(x="Sentiment", y="Correlation", data=cor_df, color="gray")
    plt.title("Correlation (Quotation vs. Fulltext) per Sentiment")
    plt.xticks(rotation=45, ha="right")
    plt.ylim(-1,1)
    plt.tight_layout()
    outfig=os.path.join(OUTPUT_DIR,"correlation_quotation_fulltext_bar.png")
    try:
        plt.savefig(outfig)
    except:
        pass
    plt.close()

    outcsv=os.path.join(CSV_OUTPUT_DIR,"quotation_fulltext_correlation.csv")
    cor_df.to_csv(outcsv, index=False)
    logging.info(f"Correlation data => {outcsv}")

# ------------------------------------------------------
# 3) Aggregation & Stats
# ------------------------------------------------------
def aggregate_sentiment_scores(df, sentiments):
    logging.info("Aggregating sentiment/emotion scores by category + sentiment.")
    recs=[]
    for cat in MEDIA_CATEGORIES.keys():
        sub=df[df["media_category"]==cat]
        for s in sentiments:
            pat=rf"^{re.escape(s)}_\d+$"
            matched=[c for c in sub.columns if re.match(pat,c)]
            if matched:
                qsum=sub[matched].clip(lower=0).sum(skipna=True).sum()
                qcount=sub[matched].clip(lower=0).count().sum()
            else:
                qsum,qcount=(0,0)

            fcol=f"{s}_fulltext"
            if fcol in sub.columns:
                fv=sub[fcol].clip(lower=0)
                f_sum=fv.sum(skipna=True)
                f_cnt=fv.count()
            else:
                f_sum,f_cnt=(0,0)

            recs.append({
                "Media Category": cat,
                "Sentiment/Emotion": s,
                "Quotation_Sum": qsum,
                "Quotation_Count": qcount,
                "Fulltext_Sum": f_sum,
                "Fulltext_Count": f_cnt
            })
    return pd.DataFrame(recs)

def calculate_averages(agg_df):
    logging.info("Calculating Quotation_Average & Fulltext_Average.")
    def sdiv(a,b): return a/b if b>0 else None
    agg_df["Quotation_Average"]=agg_df.apply(lambda r: sdiv(r["Quotation_Sum"],r["Quotation_Count"]),axis=1)
    agg_df["Fulltext_Average"]=agg_df.apply(lambda r: sdiv(r["Fulltext_Sum"],r["Fulltext_Count"]),axis=1)
    return agg_df

def calculate_mean_median(agg_df):
    logging.info("Computing global mean/median of Quotation/Fulltext averages.")
    rows=[]
    for s in CATEGORIES:
        sub=agg_df[agg_df["Sentiment/Emotion"]==s]
        qa=sub["Quotation_Average"].dropna()
        fa=sub["Fulltext_Average"].dropna()
        mean_q=qa.mean() if len(qa)>0 else None
        med_q=qa.median() if len(qa)>0 else None
        mean_f=fa.mean() if len(fa)>0 else None
        med_f=fa.median() if len(fa)>0 else None
        rows.append({
            "Sentiment/Emotion": s,
            "Mean_Quotation_Average": mean_q,
            "Median_Quotation_Average": med_q,
            "Mean_Fulltext_Average": mean_f,
            "Median_Fulltext_Average": med_f
        })
    return pd.DataFrame(rows)

def save_aggregated_scores_to_csv(agg_df, out_dir):
    fn=os.path.join(out_dir,"aggregated_sentiment_emotion_scores.csv")
    agg_df.to_csv(fn,index=False)
    print(f"Aggregated sentiment/emotion scores => {fn}")
    logging.info(f"Aggregated => {fn}")

def plot_statistics(agg_df, out_dir):
    logging.info("Plotting bar charts for Quotation/Fulltext across categories.")
    sns.set_style("whitegrid")
    for s in CATEGORIES:
        sub=agg_df[agg_df["Sentiment/Emotion"]==s]
        # Quotation
        plt.figure(figsize=(10,6))
        sns.barplot(x="Media Category", y="Quotation_Average", data=sub, color="steelblue")
        plt.title(f"Mean Quotation-Based '{s.capitalize()}' Scores")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        out1=os.path.join(out_dir,f"quote_{s}.png")
        try:
            plt.savefig(out1)
        except:
            pass
        plt.close()

        # Fulltext
        plt.figure(figsize=(10,6))
        sns.barplot(x="Media Category", y="Fulltext_Average", data=sub, color="darkorange")
        plt.title(f"Mean Fulltext-Based '{s.capitalize()}' Scores")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        out2=os.path.join(out_dir,f"fulltext_{s}.png")
        try:
            plt.savefig(out2)
        except:
            pass
        plt.close()

# ------------------------------------------------------
# 4) GEE scale computations (like original)
# ------------------------------------------------------
def compute_pearson_scale(y, mu, df_resid):
    if df_resid<=0: return np.nan
    y=np.asarray(y)
    mu=np.asarray(mu)
    r=(y-mu)/np.sqrt(mu+1e-9)
    return np.sum(r**2)/df_resid

def compute_deviance_scale(y, mu, df_resid):
    if df_resid<=0: return np.nan
    arr=[]
    for obs,lam in zip(y,mu):
        if obs>0 and lam>0:
            arr.append(obs*math.log(obs/lam)-(obs-lam))
        elif obs==0:
            arr.append(-(obs-lam))
        else:
            arr.append(np.nan)
    dev=2*np.nansum(arr)
    return dev/df_resid if df_resid>0 else np.nan

def compute_ub_scale(y, mu, df_resid):
    p=compute_pearson_scale(y, mu, df_resid)
    if np.isnan(p): return p
    return 1.1*p

def compute_bc_scale(y, mu, df_resid):
    d=compute_deviance_scale(y, mu, df_resid)
    if np.isnan(d): return d
    return 0.9*d

# ------------------------------------------------------
# 5) All Families => enumerations
# ------------------------------------------------------
def try_all_families_and_scales(df, formula, group_col):
    """
    We define which families, correlation structures, scales we'll try.
    Return the best QIC + details.
    """
    # families to brute force
    from statsmodels.genmod.families import Poisson, Gaussian, Gamma, InverseGaussian
    # NegativeBinomial from statsmodels => fix alpha=1.0 for example
    from statsmodels.genmod.families.family import NegativeBinomial

    families = [
        Poisson(),
        NegativeBinomial(alpha=1.0),
        Gaussian(),
        Gamma(),
        InverseGaussian()
    ]

    # correlation structures
    cor_structs = [Independence(), Exchangeable()]

    # scale options
    scale_opts = ["none", "pearson"]  # expand if needed: ["none","pearson","deviance","ub","bc"]

    best_qic = np.inf
    best_tuple = None

    for fam in families:
        fam_name = fam.__class__.__name__
        for cor_obj in cor_structs:
            cor_name = cor_obj.__class__.__name__
            for sc_opt in scale_opts:
                # build base model => fit => handle scale
                model=GEE.from_formula(formula, groups=group_col, data=df,
                                       family=fam, cov_struct=cor_obj)
                try:
                    base_res = model.fit(scale=None)
                    y=base_res.model.endog
                    mu=base_res.fittedvalues
                    n=len(y)
                    p=len(base_res.params)
                    dfresid=n-p
                    final_res = base_res
                    if sc_opt!="none" and dfresid>0:
                        # compute scale if possible
                        if sc_opt=="pearson":
                            val=compute_pearson_scale(y, mu, dfresid)
                        elif sc_opt=="deviance":
                            val=compute_deviance_scale(y, mu, dfresid)
                        elif sc_opt=="ub":
                            val=1.1*compute_pearson_scale(y, mu, dfresid)
                        elif sc_opt=="bc":
                            val=0.9*compute_deviance_scale(y, mu, dfresid)
                        else:
                            val=None
                        if val is not None and not np.isnan(val):
                            final_res=model.fit(scale=val)

                    qic_val = final_res.qic()
                    if isinstance(qic_val, tuple):
                        qic_val = qic_val[0]

                    if qic_val<best_qic:
                        best_qic = qic_val
                        best_tuple = (fam_name, cor_name, sc_opt, best_qic)
                except Exception as e:
                    logging.warning(f"Failed to fit {fam_name} + {cor_name} + scale={sc_opt}: {e}")
                    pass

    return best_tuple  # (famName, corName, scaleOpt, QICValue)


def run_gee_for_sentiment_measure_best_qic(df, sentiment, measure):
    """
    Instead of only Poisson and correlation structures, we do:
      - families = [Poisson, NegativeBinomial(alpha=1), Gaussian, Gamma, InverseGaussian]
      - cor structures = [Independence, Exchangeable]
      - scale opts = ["none","pearson",...]
    Return info about best QIC combination or None if can't fit anything.
    """
    d2=df.copy()
    if measure=="Quotation":
        pat=rf"^{re.escape(sentiment)}_\d+$"
        matched=[c for c in d2.columns if re.match(pat,c)]
        if not matched:
            return None
        d2["_score_col"]=d2[matched].clip(lower=0).mean(axis=1)
    else:
        fcol=f"{sentiment}_fulltext"
        if fcol not in d2.columns:
            return None
        d2["_score_col"]=d2[fcol].clip(lower=0)

    needed=["_score_col","media_outlet_clean","media_category"]
    d2=d2.dropna(subset=needed)
    if len(d2)<2 or d2["media_category"].nunique()<2:
        return None

    best_tuple = try_all_families_and_scales(
        df=d2,
        formula="_score_col ~ media_category",
        group_col="media_outlet_clean"
    )
    if best_tuple is None:
        return None
    # best_tuple => (famName, corName, scaleOpt, bestQICVal)
    famName, corName, scOpt, bestQICVal = best_tuple
    return {
        "Sentiment": sentiment,
        "Measure": measure,
        "Best_Family": famName,
        "Best_Structure": corName,
        "Best_Scale": scOpt,
        "Best_QIC_main": bestQICVal,
        "AllCombos": pd.DataFrame()  # if you want to store combos, you'd do so above
    }


def run_gee_analyses_best_qic(df):
    logging.info("Running best QIC approach for each sentiment–measure (OVER MULTIPLE FAMILIES).")
    best_list=[]
    all_combos=[]
    for s in CATEGORIES:
        for meas in ["Quotation","Fulltext"]:
            info=run_gee_for_sentiment_measure_best_qic(df, s, meas)
            if info is not None:
                best_list.append({
                    "Sentiment": info["Sentiment"],
                    "Measure": info["Measure"],
                    "Best_Family": info["Best_Family"],
                    "Best_Structure": info["Best_Structure"],
                    "Best_Scale": info["Best_Scale"],
                    "Best_QIC_main": info["Best_QIC_main"]
                })
                # info["AllCombos"] if you had them
    df_best=pd.DataFrame(best_list)
    df_all=pd.DataFrame()  # if not storing combos
    return df_best, df_all

# ------------------------------------------------------
# 6) Pairwise + DiffIDs (unchanged from your original)
# ------------------------------------------------------
def refit_best_gee_with_scale(df, sentiment, measure, family_name, struct, scale_name):
    """
    Build final GEE => apply scale => return fitted result or None
    We parse family_name => create actual family obj
    """
    # parse family
    from statsmodels.genmod.families import (
        Poisson, Gaussian, Gamma, InverseGaussian
    )
    from statsmodels.genmod.families.family import NegativeBinomial

    if family_name=="Poisson":
        fam_obj=Poisson()
    elif family_name=="NegativeBinomial":
        fam_obj=NegativeBinomial(alpha=1.0)
    elif family_name=="Gaussian":
        fam_obj=Gaussian()
    elif family_name=="Gamma":
        fam_obj=Gamma()
    elif family_name=="InverseGaussian":
        fam_obj=InverseGaussian()
    else:
        logging.warning(f"Unknown family={family_name}")
        return None

    d2=df.copy()
    if measure=="Quotation":
        pat=rf"^{re.escape(sentiment)}_\d+$"
        matched=[c for c in d2.columns if re.match(pat,c)]
        if not matched:
            return None
        d2["_score_col"]=d2[matched].clip(lower=0).mean(axis=1)
    else:
        fcol=f"{sentiment}_fulltext"
        if fcol not in d2.columns:
            return None
        d2["_score_col"]=d2[fcol].clip(lower=0)

    needed=["_score_col","media_outlet_clean","media_category"]
    d2=d2.dropna(subset=needed)
    if len(d2)<2 or d2["media_category"].nunique()<2:
        return None

    d2["media_category"]=d2["media_category"].astype("category")
    if struct=="Independence":
        cov_obj=Independence()
    else:
        cov_obj=Exchangeable()

    model=GEE.from_formula("_score_col ~ media_category",
        groups="media_outlet_clean",data=d2,
        family=fam_obj, cov_struct=cov_obj)
    bres=model.fit(scale=None)

    # handle scale
    y=bres.model.endog
    mu=bres.fittedvalues
    n=len(y)
    p=len(bres.params)
    dfresid=n-p
    if scale_name=="none" or dfresid<=0:
        return bres
    else:
        if scale_name=="pearson":
            scv=compute_pearson_scale(y,mu,dfresid)
        elif scale_name=="deviance":
            scv=compute_deviance_scale(y,mu,dfresid)
        elif scale_name=="ub":
            scv=1.1*compute_pearson_scale(y,mu,dfresid)
        elif scale_name=="bc":
            scv=0.9*compute_deviance_scale(y,mu,dfresid)
        else:
            scv=None
        if scv is None or np.isnan(scv):
            return bres
        return model.fit(scale=scv)


def pairwise_and_cld(df, sentiment, measure, family_name, struct, scale_name):
    """
    Refit best GEE => final_fit => GEE summary => Pairwise => build DiffIDs.
    """
    final_fit=refit_best_gee_with_scale(df, sentiment, measure, family_name, struct, scale_name)
    if final_fit is None:
        return None,None,None
    summary_txt=final_fit.summary().as_text()

    # Pairwise
    params=final_fit.params
    cov=final_fit.cov_params()
    mdf=final_fit.model.data.frame
    cats=mdf["media_category"].cat.categories
    ref=cats[0]
    idx_map={ref:0}
    for c in cats[1:]:
        nm=f"media_category[T.{c}]"
        idx_map[c]=final_fit.model.exog_names.index(nm)

    pair_list=[]
    for i in range(len(cats)):
        for j in range(i+1,len(cats)):
            ca,cb=cats[i],cats[j]
            con=np.zeros(len(params))
            if ca==ref and cb!=ref:
                con[idx_map[cb]]=-1.0
            elif cb==ref and ca!=ref:
                con[idx_map[ca]]=1.0
            else:
                con[idx_map[ca]]=1.0
                con[idx_map[cb]]=-1.0
            diff_est=con@params
            diff_var=con@cov@con
            diff_se=np.sqrt(diff_var)
            z=diff_est/diff_se
            pval=2*(1-norm.cdf(abs(z)))
            pair_list.append((ca,cb,diff_est,diff_se,z,pval))

    pair_df=pd.DataFrame(pair_list,columns=["CategoryA","CategoryB","Difference","SE","Z","p_value"])
    rej, p_adj,_,_= multipletests(pair_df["p_value"], method="fdr_bh")
    pair_df["p_value_adj"]=p_adj
    pair_df["reject_H0"]=rej

    # build DiffIDs
    cat_list=list(cats)
    cat_index_map={cat_list[i]: i+1 for i in range(len(cat_list))}
    diff_map={c: set() for c in cat_list}
    for i,row in pair_df.iterrows():
        A=row["CategoryA"]
        B=row["CategoryB"]
        if row["reject_H0"]:
            diff_map[A].add(cat_index_map[B])
            diff_map[B].add(cat_index_map[A])

    rows=[]
    for c in cat_list:
        diffs=sorted(list(diff_map[c]))
        diffs_str=",".join(str(x) for x in diffs)
        rows.append((c,diffs_str))
    diffIDs_df=pd.DataFrame(rows,columns=["Category","DiffIDs"])

    return summary_txt, pair_df, diffIDs_df

# ------------------------------------------------------
# 7) Compile results
# ------------------------------------------------------
def compile_results_into_multiple_workbooks(
    aggregated_df, stats_df, raw_df,
    df_best_qic, df_all_combos,
    plots_dir,
    main_excel, raw_excel, gee_excel, plots_excel, combined_excel,
    df_full
):
    # analysis_main
    with pd.ExcelWriter(main_excel, engine="openpyxl") as w:
        aggregated_df.to_excel(w,"Aggregated_Scores",index=False)
        stats_df.to_excel(w,"Mean_Median_Statistics",index=False)

    # analysis_raw
    with pd.ExcelWriter(raw_excel, engine="openpyxl") as w:
        raw_df.to_excel(w,"Raw_Data",index=False)
        for s in CATEGORIES:
            s_cols=[c for c in raw_df.columns if c.startswith(s+"_")]
            s_df=raw_df[["media_category","media_outlet"]+s_cols].copy()
            sheet_name=f"Raw_{s[:29]}"
            s_df.to_excel(w,sheet_name,index=False)

    # analysis_gee => best QIC w/ pairwise
    with pd.ExcelWriter(gee_excel, engine="openpyxl") as writer:
        idx_rows=[]
        for i, row in df_best_qic.iterrows():
            s=row["Sentiment"]
            meas=row["Measure"]
            fam=row["Best_Family"]
            st=row["Best_Structure"]
            sc=row["Best_Scale"]
            sh_name=f"BestQIC_{s[:8]}_{meas[:8]}"

            summary_txt, pair_df, diffIDs_df = pairwise_and_cld(df_full, s, meas, fam, st, sc)
            if summary_txt is None:
                tmp_df=pd.DataFrame({"Summary":["No valid model or not enough data."]})
                tmp_df.to_excel(writer, sh_name, index=False)
                continue

            # Summaries
            lines=summary_txt.split("\n")
            sumdf=pd.DataFrame({"GEE_Summary":lines})
            sumdf.to_excel(writer, sh_name, index=False)

            sr=len(sumdf)+2
            pair_df.to_excel(writer, sh_name, index=False, startrow=sr)

            ws=writer.sheets[sh_name]
            for row_idx in range(len(pair_df)):
                if pair_df.loc[row_idx,"reject_H0"]:
                    excel_row=sr+1 + row_idx
                    for col_idx in range(1, pair_df.shape[1] + 1):
                        cell=ws.cell(row=excel_row+1, column=col_idx)
                        cell.fill=PatternFill(fill_type="solid", start_color="FFFF0000", end_color="FFFF0000")

            sr2=sr+len(pair_df)+2
            if diffIDs_df is not None and not diffIDs_df.empty:
                diffIDs_df.to_excel(writer, sh_name, index=False, startrow=sr2)

            idx_rows.append({
                "Sentiment": s,
                "Measure": meas,
                "Family": fam,
                "Structure": st,
                "Scale": sc,
                "BestQIC": row["Best_QIC_main"]
            })
        idxdf=pd.DataFrame(idx_rows)
        idxdf.to_excel(writer,"BestQIC_Index",index=False)

        if not df_all_combos.empty:
            df_all_combos.to_excel(writer,"All_Combos",index=False)

    # analysis_plots
    wb_plots=Workbook()
    if "Sheet" in wb_plots.sheetnames:
        wb_plots.remove(wb_plots["Sheet"])
    any_sheets=False
    for s in CATEGORIES:
        q_path=os.path.join(plots_dir,f"quote_{s}.png")
        if os.path.exists(q_path):
            st=f"Quote_{s[:28]}"
            ws=wb_plots.create_sheet(title=st)
            try:
                img=ExcelImage(q_path)
                img.anchor="A1"
                ws.add_image(img)
            except:
                pass
            any_sheets=True
        f_path=os.path.join(plots_dir,f"fulltext_{s}.png")
        if os.path.exists(f_path):
            st2=f"Fulltext_{s[:25]}"
            ws2=wb_plots.create_sheet(title=st2)
            try:
                img2=ExcelImage(f_path)
                img2.anchor="A1"
                ws2.add_image(img2)
            except:
                pass
            any_sheets=True

    cbar=os.path.join(plots_dir,"correlation_quotation_fulltext_bar.png")
    if os.path.exists(cbar):
        ws3=wb_plots.create_sheet("Correlation_Bar")
        try:
            ig3=ExcelImage(cbar)
            ig3.anchor="A1"
            ws3.add_image(ig3)
        except:
            pass
        any_sheets=True

    combp=os.path.join(plots_dir,"combined_normalized_scatter.png")
    if os.path.exists(combp):
        ws4=wb_plots.create_sheet("Combined_ZScatter")
        try:
            ig4=ExcelImage(combp)
            ig4.anchor="A1"
            ws4.add_image(ig4)
        except:
            pass
        any_sheets=True

    if not any_sheets:
        wb_plots.create_sheet("DummySheet")
    wb_plots.save(plots_excel)

    # analysis_combined
    raw_clean=raw_df.copy()
    raw_clean=raw_clean.applymap(lambda x:", ".join(x) if isinstance(x,list) else x)
    wb_comb=Workbook()
    if "Sheet" in wb_comb.sheetnames:
        wb_comb.remove(wb_comb["Sheet"])

    ws_agg=wb_comb.create_sheet("Aggregated_Scores")
    for r in dataframe_to_rows(aggregated_df,index=False,header=True):
        ws_agg.append(r)

    ws_stats=wb_comb.create_sheet("Mean_Median_Statistics")
    for r in dataframe_to_rows(stats_df,index=False,header=True):
        ws_stats.append(r)

    ws_raw=wb_comb.create_sheet("Raw_Data")
    for r in dataframe_to_rows(raw_clean,index=False,header=True):
        ws_raw.append(r)

    ws_best=wb_comb.create_sheet("BestQIC_Table")
    for r in dataframe_to_rows(df_best_qic,index=False,header=True):
        ws_best.append(r)

    wb_comb.save(combined_excel)

# ------------------------------------------------------
def main():
    setup_logging()
    logging.info("Starting best QIC GEE approach (with multiple families) + LSD-based DiffIDs (final).")

    # 1) Load
    df=load_jsonl(INPUT_JSONL_FILE)
    print(f"Total articles loaded: {len(df)}")

    # 2) map + chunk + stats
    df=map_media_outlet_to_category(df)
    chunk_and_save(df,20000)
    print_basic_stats(df)

    # 3) correlation
    print("Performing Quotation vs Fulltext correlation analysis...")
    analyze_quotation_fulltext_correlation(df)

    # 4) aggregator
    print("Aggregating sentiment/emotion scores per media category...")
    agg_df=aggregate_sentiment_scores(df,CATEGORIES)
    agg_df=calculate_averages(agg_df)
    stats_df=calculate_mean_median(agg_df)
    save_aggregated_scores_to_csv(agg_df,CSV_OUTPUT_DIR)
    plot_statistics(agg_df,OUTPUT_DIR)

    # 5) best QIC approach
    print("Fitting best QIC approach for each sentiment–measure (with BH pairwise, among multiple families).")
    df_best, df_allcombos = run_gee_analyses_best_qic(df)
    print("Best QIC approach completed.\n")
    print(df_best)

    # 6) compile => summary, pairwise, & DiffIDs
    compile_results_into_multiple_workbooks(
        aggregated_df=agg_df,
        stats_df=stats_df,
        raw_df=df,
        df_best_qic=df_best,
        df_all_combos=df_allcombos,
        plots_dir=OUTPUT_DIR,
        main_excel=OUTPUT_EXCEL_MAIN,
        raw_excel=OUTPUT_EXCEL_RAW,
        gee_excel=OUTPUT_EXCEL_GEE,
        plots_excel=OUTPUT_EXCEL_PLOTS,
        combined_excel=OUTPUT_EXCEL_COMBINED,
        df_full=df
    )

    print("Analysis completed successfully with multi-family GEE approach.")
    logging.info("Analysis completed successfully.")


if __name__=="__main__":
    main()

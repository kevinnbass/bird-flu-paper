#!/usr/bin/env python3
# gee_final_best_qic_unstructured_fix_dtype.py
"""
Script that:
  1) Loads data from JSONL, parses 'date' -> integer days => 'time_int'
  2) Ensures all columns used by Unstructured are numeric
  3) Chunks & saves CSV
  4) Performs correlation analyses & scatterplots for Quotation vs Fulltext
  5) Aggregates sentiment/emotion scores + bar plots
  6) Finds best QIC combination (Ind/Exch/Unstructured × scale=[none,pearson,deviance,ub,bc]),
     passing 'time_int' only for Unstructured
  7) For each best QIC model:
     - Refit & produce GEE summary
     - Compute pairwise comparisons (Holm correction)
     - Build LSD-like CLD that merges letter sets if bridging multiple groups
  8) Writes summary, pairwise, & CLD into analysis_gee.xlsx (one sheet per sentiment×measure)
  9) Also saves analysis_main.xlsx, analysis_raw.xlsx, analysis_plots.xlsx, analysis_combined.xlsx

Fixes warnings of the form:
  "Pandas data cast to numpy dtype of object. Check input data with np.asarray(data)."
by explicitly converting exog/endog/time_ to numeric, ensuring no object dtypes remain.
"""

import json
import os
import re
import sys
import warnings
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from openpyxl import Workbook
from openpyxl.drawing.image import Image as ExcelImage
from openpyxl.utils.dataframe import dataframe_to_rows

import statsmodels.api as sm
from statsmodels.genmod.generalized_estimating_equations import GEE
from statsmodels.genmod.families import Poisson
from statsmodels.genmod.cov_struct import Independence, Exchangeable, Unstructured
from statsmodels.stats.multitest import multipletests
from statsmodels.tools.tools import add_constant
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

# ------------------------------
# 1) Load, parse date->time_int, chunk, basic stats
# ------------------------------
def load_jsonl_and_prepare_time(jsonl_file):
    logging.info(f"Loading JSONL from {jsonl_file}")
    recs=[]
    with open(jsonl_file,"r",encoding="utf-8") as f:
        for line in tqdm(f, desc="Loading JSONL data"):
            rec=json.loads(line)
            recs.append(rec)
    df=pd.DataFrame(recs)
    logging.debug(f"Loaded DataFrame shape={df.shape}")

    if "date" not in df.columns:
        logging.warning("No 'date' column => fallback time_int=1.")
        df["time_int"]=1
        return df

    df["date_parsed"] = pd.to_datetime(df["date"], errors="coerce")
    if df["date_parsed"].isnull().all():
        logging.warning("All 'date' invalid => fallback time_int=1.")
        df["time_int"]=1
    else:
        min_date = df["date_parsed"].min()
        df["time_int"]=(df["date_parsed"]-min_date).dt.days + 1
        df["time_int"]=df["time_int"].fillna(1).astype(int)
    return df

def map_media_outlet_to_category(df):
    cat_map={}
    for cat,outls in MEDIA_CATEGORIES.items():
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
    logging.info(f"Chunking => len={len(df)}, chunk_size={chunk_size}")
    for i in range(0,len(df),chunk_size):
        part=df.iloc[i:i+chunk_size]
        out_csv=os.path.join(CSV_OUTPUT_DIR, f"raw_data_part_{(i//chunk_size)+1}.csv")
        part.to_csv(out_csv,index=False)
        print(f"Saved chunk {(i//chunk_size)+1} to {out_csv}")

def print_basic_stats(df):
    logging.info(f"Basic stats => total articles={len(df)}")
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

# ------------------------------
# 2) Quotation vs. Fulltext correlation
# ------------------------------
def analyze_quotation_fulltext_correlation(df):
    logging.info("Analyzing Quotation vs Fulltext correlation.")
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
    scatter_map={}
    for s in CATEGORIES:
        sub=agg_df[(agg_df["Sentiment"]==s)].dropna(subset=["Quotation_Average","Fulltext_Average"])
        if len(sub)>1:
            cor_val,_=pearsonr(sub["Quotation_Average"], sub["Fulltext_Average"])
        else:
            cor_val=np.nan
        cor_results.append({"Sentiment":s,"Correlation":cor_val})

        if not sub.empty:
            plt.figure(figsize=(6,5))
            sns.scatterplot(x="Quotation_Average",y="Fulltext_Average",data=sub,hue="MediaCategory",s=50)
            plt.title(f"{s.capitalize()} (Quotation vs Fulltext)\nr={cor_val:.3f}")
            plt.tight_layout()
            outp=os.path.join(OUTPUT_DIR,f"scatter_{s}.png")
            try:
                plt.savefig(outp)
            except:
                pass
            plt.close()
        scatter_map[s]=sub.copy()

    cor_df=pd.DataFrame(cor_results)
    plt.figure(figsize=(8,5))
    sns.barplot(x="Sentiment",y="Correlation",data=cor_df,color="gray")
    plt.title("Correlation (Quotation vs. Fulltext) per Sentiment")
    plt.xticks(rotation=45,ha="right")
    plt.ylim(-1,1)
    plt.tight_layout()
    cbar=os.path.join(OUTPUT_DIR,"correlation_quotation_fulltext_bar.png")
    try:
        plt.savefig(cbar)
    except:
        pass
    plt.close()

    combo=[]
    for s,sd in scatter_map.items():
        if not sd.empty:
            cpy=sd.copy()
            cpy["Sentiment"]=s
            combo.append(cpy)
    if combo:
        allc=pd.concat(combo,ignore_index=True)
        allc["Qmean"]=allc.groupby("Sentiment")["Quotation_Average"].transform("mean")
        allc["Qstd"]=allc.groupby("Sentiment")["Quotation_Average"].transform("std")
        allc["Fmean"]=allc.groupby("Sentiment")["Fulltext_Average"].transform("mean")
        allc["Fstd"]=allc.groupby("Sentiment")["Fulltext_Average"].transform("std")
        allc["Quotation_Z"]=(allc["Quotation_Average"]-allc["Qmean"])/allc["Qstd"]
        allc["Fulltext_Z"]=(allc["Fulltext_Average"]-allc["Fmean"])/allc["Fstd"]
        v=allc.dropna(subset=["Quotation_Z","Fulltext_Z"])
        if len(v)>1:
            r_val,_=pearsonr(v["Quotation_Z"],v["Fulltext_Z"])
        else:
            r_val=np.nan
        plt.figure(figsize=(7,5))
        sns.regplot(x="Quotation_Z",y="Fulltext_Z",data=v,
                    scatter_kws={"color":"black","alpha":0.6},line_kws={"color":"red"})
        plt.title(f"All Sentiments Combined (Z-scores)\nr={r_val:.3f}")
        plt.tight_layout()
        csc=os.path.join(OUTPUT_DIR,"combined_normalized_scatter.png")
        try:
            plt.savefig(csc)
        except:
            pass
        plt.close()

    out_csv=os.path.join(CSV_OUTPUT_DIR,"quotation_fulltext_correlation.csv")
    cor_df.to_csv(out_csv,index=False)
    logging.info(f"Correlation data => {out_csv}")

# ------------------------------
# 4) Aggregation & Stats
# ------------------------------
def aggregate_sentiment_scores(df,sentiments):
    logging.info("Aggregating sentiment/emotion scores by category + sentiment.")
    ...
    # unchanged, same as your code

def calculate_averages(agg_df):
    logging.info("Calculating Quotation_Average & Fulltext_Average.")
    ...
    # unchanged

def calculate_mean_median(agg_df):
    logging.info("Computing global mean/median of Quotation/Fulltext averages.")
    ...
    # unchanged

def save_aggregated_scores_to_csv(agg_df,out_dir):
    ...
    # unchanged

def plot_statistics(agg_df,out_dir):
    ...
    # unchanged

# ------------------------------
# 5) GEE + time dimension => fix dtype
# ------------------------------
def fit_and_compute_scales(model):
    base_res=model.fit(scale=None)
    base_qic=base_res.qic()
    y=base_res.model.endog
    mu=base_res.fittedvalues
    n=len(y)
    p=len(base_res.params)
    dfresid=n-p

    pear=compute_pearson_scale(y,mu,dfresid)
    dev=compute_deviance_scale(y,mu,dfresid)
    ubv=compute_ub_scale(y,mu,dfresid)
    bcv=compute_bc_scale(y,mu,dfresid)

    results={}
    if isinstance(base_qic, tuple):
        q_m,q_a=base_qic
    else:
        q_m,q_a=base_qic,None
    results["none"]=(q_m,q_a,None)

    from math import isnan
    for (nm,val) in [("pearson",pear),("deviance",dev),("ub",ubv),("bc",bcv)]:
        if (not isnan(val)) and (dfresid>0):
            re2=model.fit(scale=val)
            r_qic=re2.qic()
            if isinstance(r_qic,tuple):
                rm,ra=r_qic
            else:
                rm,ra=r_qic,None
            results[nm]=(rm,ra,val)
        else:
            results[nm]=(np.nan,None,val)
    return results

def compute_pearson_scale(y,mu,df_resid):
    ...
def compute_deviance_scale(y,mu,df_resid):
    ...
def compute_ub_scale(y,mu,df_resid):
    ...
def compute_bc_scale(y,mu,df_resid):
    ...

# ------------------------------
# 6) Best QIC approach => fix data cast object
# ------------------------------
def run_gee_for_sentiment_measure_best_qic(df, sentiment, measure):
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

    needed=["_score_col","media_outlet_clean","media_category","time_int"]
    d2=d2.dropna(subset=needed)
    if len(d2)<2 or d2["media_category"].nunique()<2:
        return None

    best_qic_main=np.inf
    best_tuple=None
    combos=[]
    structures=[Independence(), Exchangeable(), Unstructured()]

    for cov_obj in structures:
        st_name=cov_obj.__class__.__name__
        try:
            if st_name=="Unstructured":
                # build numeric endog,exog
                endog = d2["_score_col"].astype(float).to_numpy()
                cat_dummies = pd.get_dummies(d2["media_category"].astype("category"), drop_first=False)
                cat_dummies = cat_dummies.astype(float)  # ensure float
                exog = add_constant(cat_dummies)         # also float
                groups = d2["media_outlet_clean"].values
                # groups can be object => that's okay
                time_  = d2["time_int"].astype(int).values

                base_model = GEE(endog, exog, groups=groups, time=time_,
                                 family=Poisson(), cov_struct=cov_obj)
            else:
                base_model=GEE.from_formula(
                    "_score_col ~ media_category",
                    groups="media_outlet_clean",
                    data=d2,
                    family=Poisson(),
                    cov_struct=cov_obj
                )
            sc_map=fit_and_compute_scales(base_model)
        except Exception as e:
            logging.warning(f"Structure={st_name} => fit failed => {e}")
            continue

        for sm,(qm,qa,sc_val) in sc_map.items():
            combos.append({
                "Sentiment":sentiment,
                "Measure":measure,
                "Structure":st_name,
                "ScaleMethod":sm,
                "NumericScale":sc_val,
                "QIC_main":qm,
                "QIC_alt":qa
            })
            if (qm<best_qic_main) and (not np.isnan(qm)):
                best_qic_main=qm
                best_tuple=(st_name,sm,qm,sc_val)

    if best_tuple is None:
        return None
    stn, sm, qicv, scale_num=best_tuple
    return {
        "Sentiment":sentiment,
        "Measure":measure,
        "Best_Structure":stn,
        "Best_Scale":sm,
        "Best_QIC_main":qicv,
        "AllCombos":pd.DataFrame(combos),
        "Summary":f"Best scale={sm}, numeric={scale_num}"
    }

def run_gee_analyses_best_qic(df):
    logging.info("Running best QIC approach (Ind/Exch/Unstructured).")
    best_list=[]
    all_combos=[]
    for s in CATEGORIES:
        for meas in ["Quotation","Fulltext"]:
            info=run_gee_for_sentiment_measure_best_qic(df,s,meas)
            if info is not None:
                best_list.append({
                    "Sentiment": info["Sentiment"],
                    "Measure": info["Measure"],
                    "Best_Structure": info["Best_Structure"],
                    "Best_Scale": info["Best_Scale"],
                    "Best_QIC_main": info["Best_QIC_main"]
                })
                all_combos.append(info["AllCombos"])
    df_best=pd.DataFrame(best_list)
    df_all=pd.concat(all_combos, ignore_index=True) if all_combos else pd.DataFrame()
    return df_best, df_all

# ------------------------------
# 7) Pairwise + LSD-based CLD
# ------------------------------
def refit_best_gee_with_scale(df, sentiment, measure, struct, scale_name):
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

    needed=["_score_col","media_outlet_clean","media_category","time_int"]
    d2=d2.dropna(subset=needed)
    if len(d2)<2 or d2["media_category"].nunique()<2:
        return None

    from statsmodels.genmod.cov_struct import Unstructured,Independence,Exchangeable
    if struct=="Unstructured":
        try:
            endog = d2["_score_col"].astype(float).to_numpy()
            cat_dummies = pd.get_dummies(d2["media_category"].astype("category"), drop_first=False)
            cat_dummies=cat_dummies.astype(float)
            exog = add_constant(cat_dummies)
            groups = d2["media_outlet_clean"].values
            time_  = d2["time_int"].astype(int).values
            base_model = GEE(endog, exog, groups=groups, time=time_,
                             family=Poisson(), cov_struct=Unstructured())
            bres=base_model.fit(scale=None)
        except Exception as e:
            logging.warning(f"Refit base => Unstructured => fail => {e}")
            return None
    elif struct=="Independence":
        base_model=GEE.from_formula(
            "_score_col ~ media_category",
            groups="media_outlet_clean",
            data=d2,
            family=Poisson(),
            cov_struct=Independence()
        )
        try:
            bres=base_model.fit(scale=None)
        except Exception as e:
            logging.warning(f"Refit =>Independence => fail => {e}")
            return None
    else:
        # Exchangeable
        base_model=GEE.from_formula(
            "_score_col ~ media_category",
            groups="media_outlet_clean",
            data=d2,
            family=Poisson(),
            cov_struct=Exchangeable()
        )
        try:
            bres=base_model.fit(scale=None)
        except Exception as e:
            logging.warning(f"Refit => Exchangeable => fail => {e}")
            return None

    if scale_name=="none":
        return bres

    y=bres.model.endog
    mu=bres.fittedvalues
    n=len(y)
    p=len(bres.params)
    dfresid=n-p
    if dfresid<=0:
        return bres
    # numeric scale
    scv=None
    if scale_name=="pearson":
        scv=compute_pearson_scale(y,mu,dfresid)
    elif scale_name=="deviance":
        scv=compute_deviance_scale(y,mu,dfresid)
    elif scale_name=="ub":
        scv=compute_ub_scale(y,mu,dfresid)
    elif scale_name=="bc":
        scv=compute_bc_scale(y,mu,dfresid)

    if scv is None or np.isnan(scv):
        return bres

    try:
        final_fit=base_model.fit(scale=scv)
        return final_fit
    except Exception as e:
        logging.warning(f"Refit scale={scale_name}, val={scv} => fail =>{e}")
        return bres

def pairwise_and_cld(df, sentiment, measure, struct, scale_name):
    final_fit=refit_best_gee_with_scale(df,sentiment,measure,struct,scale_name)
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
    rej,p_adj,_,_ = multipletests(pair_df["p_value"],method="holm")
    pair_df["p_value_adj"]=p_adj
    pair_df["reject_H0"]=rej

    cld_df=build_lsd_cld(final_fit, pair_df)
    return summary_txt, pair_df, cld_df

def build_lsd_cld(fit_result, pair_df):
    ...
    # LSD-based merging logic => same as prior code

# ------------------------------
# 8) Compile results
# ------------------------------
def compile_results_into_multiple_workbooks(
    aggregated_df, stats_df, raw_df,
    df_best_qic, df_all_combos,
    plots_dir,
    main_excel, raw_excel, gee_excel, plots_excel, combined_excel,
    df_full
):
    ...
    # same code for output xlsx, no changes needed

# ------------------------------
def main():
    setup_logging()
    logging.info("Starting best QIC approach => fix 'object' dtype => convert exog/endog/time_ to numeric if Unstructured")

    # 1) load, parse time
    df=load_jsonl_and_prepare_time(INPUT_JSONL_FILE)

    # 2) map + chunk + stats
    df=map_media_outlet_to_category(df)
    chunk_and_save(df,20000)
    print_basic_stats(df)

    # 3) correlation
    print("Performing Quotation vs Fulltext correlation analysis...")
    analyze_quotation_fulltext_correlation(df)

    # 4) aggregation
    print("Aggregating sentiment/emotion scores per media category...")
    agg_df=aggregate_sentiment_scores(df,CATEGORIES)
    agg_df=calculate_averages(agg_df)
    stats_df=calculate_mean_median(agg_df)
    save_aggregated_scores_to_csv(agg_df,CSV_OUTPUT_DIR)
    plot_statistics(agg_df,OUTPUT_DIR)

    # 5) best QIC => Ind/Exch/Unstructured => ensure numeric
    print("Fitting best QIC approach... with LSD-based CLD.")
    df_best, df_all=run_gee_analyses_best_qic(df)
    print("Best QIC approach completed.\n")

    compile_results_into_multiple_workbooks(
        aggregated_df=agg_df,
        stats_df=stats_df,
        raw_df=df,
        df_best_qic=df_best,
        df_all_combos=df_all,
        plots_dir=OUTPUT_DIR,
        main_excel=OUTPUT_EXCEL_MAIN,
        raw_excel=OUTPUT_EXCEL_RAW,
        gee_excel=OUTPUT_EXCEL_GEE,
        plots_excel=OUTPUT_EXCEL_PLOTS,
        combined_excel=OUTPUT_EXCEL_COMBINED,
        df_full=df
    )

    print("Analysis completed successfully.")
    logging.info("Analysis completed successfully.")

if __name__=="__main__":
    main()

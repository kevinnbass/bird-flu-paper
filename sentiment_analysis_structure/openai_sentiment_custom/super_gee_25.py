#!/usr/bin/env python3
# super_gee_all_validations_dual_cats_sens_in_same_xlsx_labeled_collapsed_subdir.py
"""
A complete Python script that:
  1) Loads data from JSONL
  2) Normalizes high_rate_2 => 'yes'/'no'
  3) Runs the pipeline FOUR times:
       (A) Original categories => "Yes"
       (B) Original categories => "All"
       (C) Collapsed categories => "Yes_collapsed"
       (D) Collapsed categories => "All_collapsed"

Small clusters are merged in a "Sens" run, appended to the same XLSX in new tabs.
All output files are placed into a subdirectory "analysis_output/",
and any run with "_collapsed" in its prefix is labeled "analysis_XXX_collapsed_YYY.xlsx"
"""

import json
import os
import re
import sys
import warnings
import logging
import math
import gc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Statsmodels & related
import statsmodels.api as sm
from statsmodels.genmod.families import (
    Poisson, NegativeBinomial,
    Gaussian, Gamma, InverseGaussian
)
import statsmodels.genmod.families.links as ln
from statsmodels.genmod.generalized_estimating_equations import GEE
from statsmodels.genmod.cov_struct import Independence, Exchangeable
from statsmodels.stats.multitest import multipletests
from scipy.stats import pearsonr, norm

from openpyxl import Workbook, load_workbook
from openpyxl.drawing.image import Image as ExcelImage
from openpyxl.utils.dataframe import dataframe_to_rows
from openpyxl.styles import PatternFill

###############################################################################
# CONFIG
###############################################################################
INPUT_JSONL_FILE = "processed_all_articles_fixed_4.jsonl"

# Subdirectory for final Excel outputs
OUTPUT_DIR = "analysis_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# We'll store all base file names here:
BASE_MAIN_FILE = "analysis_main.xlsx"
BASE_RAW_FILE  = "analysis_raw.xlsx"
BASE_GEE_FILE  = "analysis_gee.xlsx"
BASE_PLOTS_FILE= "analysis_plots.xlsx"
BASE_COMBINED_FILE="analysis_combined.xlsx"

LOG_FILE = "analysis.log"

os.makedirs("graphs_analysis", exist_ok=True)  # for graphs
os.makedirs("csv_raw_scores", exist_ok=True)  # for CSV

CATEGORIES = [
    "joy","sadness","anger","fear",
    "surprise","disgust","trust","anticipation",
    "negative_sentiment","positive_sentiment"
]

# Original 6-cat scheme
MEDIA_CATEGORIES_ORIG = {
    "Scientific": ["nature","sciam","stat","newscientist"],
    "Left": ["theatlantic","the daily beast","the intercept","mother jones","msnbc","slate","vox","huffpost"],
    "Lean Left": ["ap","axios","cnn","guardian","business insider","nbcnews","npr","nytimes","politico","propublica","wapo","usa today"],
    "Center": ["reuters","marketwatch","financial times","newsweek","forbes"],
    "Lean Right": ["thedispatch","epochtimes","foxbusiness","wsj","national review","washtimes"],
    "Right": ["breitbart","theblaze","daily mail","dailywire","foxnews","nypost","newsmax"],
}

# Collapsed 4-cat scheme
MEDIA_CATEGORIES_COLLAPSED = {
    "Scientific": ["nature","sciam","stat","newscientist"],
    "Left": [
        "theatlantic","the daily beast","the intercept","mother jones",
        "msnbc","slate","vox","huffpost",
        "ap","axios","cnn","guardian","business insider","nbcnews",
        "npr","nytimes","politico","propublica","wapo","usa today"
    ],
    "Center": ["reuters","marketwatch","financial times","newsweek","forbes"],
    "Right": [
        "thedispatch","epochtimes","foxbusiness","wsj","national review","washtimes",
        "breitbart","theblaze","daily mail","dailywire","foxnews","nypost","newsmax"
    ],
}

# For Compare_Imbalance
COMPARE_IMBALANCE = []

###############################################################################
# LOGGING
###############################################################################
def setup_logging():
    log_format = "%(asctime)s [%(levelname)s] %(module)s - %(message)s"
    logging.basicConfig(filename=LOG_FILE, level=logging.DEBUG, format=log_format)
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(console)
    logging.info("Logging initialized.")

warnings.filterwarnings(
    "ignore",
    message="QIC values obtained using scale=None are not appropriate for comparing models"
)

###############################################################################
# rename_files_for_prefix => put them into OUTPUT_DIR, plus handle "collapsed"
###############################################################################
def rename_files_for_prefix(base_name, prefix):
    """
    - Takes a base filename like "analysis_main.xlsx"
    - Puts it in the subdirectory OUTPUT_DIR
    - If prefix ends with "_collapsed", insert "_collapsed_" before the final prefix
      E.g. prefix="Yes_collapsed" => "analysis_main_collapsed_Yes.xlsx"
    - If prefix is empty => no change
    Otherwise => "analysis_main_<prefix>.xlsx"
    """
    if not prefix:
        # no prefix => just subdir
        return os.path.join(OUTPUT_DIR, base_name)
    # e.g. base_name= "analysis_main.xlsx"
    root, ext = os.path.splitext(base_name)  # "analysis_main", ".xlsx"

    if prefix.endswith("_collapsed"):
        # e.g. "Yes_collapsed" => short_pre="Yes"
        short_pre = prefix.replace("_collapsed","")  # => "Yes"
        new_fn = f"{root}_collapsed_{short_pre}{ext}"
    else:
        new_fn = f"{root}_{prefix}{ext}"

    return os.path.join(OUTPUT_DIR, new_fn)

###############################################################################
# 1) Load + map
###############################################################################
def load_jsonl(jsonl_file):
    logging.info(f"Loading JSONL => {jsonl_file}")
    records=[]
    with open(jsonl_file,"r",encoding="utf-8") as f:
        for line in tqdm(f,desc="Loading JSONL"):
            rec=json.loads(line)
            records.append(rec)
    df=pd.DataFrame(records)
    return df

def map_media_outlet_to_category(df, cat_dict):
    cat_map={}
    for cat, outls in cat_dict.items():
        for o in outls:
            cat_map[o.lower().strip()] = cat

    df2 = df.copy()
    df2["media_outlet_clean"] = df2["media_outlet"].str.lower().str.strip()
    df2["media_category"] = df2["media_outlet_clean"].map(cat_map).fillna("Other")
    return df2

###############################################################################
# chunk + stats
###############################################################################
def chunk_and_save(df, chunk_size=20000, prefix=""):
    from tqdm import trange
    logging.info(f"Chunking => len={len(df)}, chunk_size={chunk_size}, prefix={prefix}")
    for i in range(0,len(df),chunk_size):
        part=df.iloc[i:i+chunk_size]
        suffix=f"_{prefix}" if prefix else ""
        out_csv=os.path.join("csv_raw_scores",f"raw_data_part_{(i//chunk_size)+1}{suffix}.csv")
        part.to_csv(out_csv,index=False)
        print(f"Saved chunk {(i//chunk_size)+1} => {out_csv}")

def print_basic_stats(df, prefix=""):
    logging.info(f"Basic stats => total articles={len(df)}, prefix={prefix}")
    print(f"\nSummary Stats (prefix={prefix or '(None)'}) =>")
    print("Total articles:", len(df))
    if "media_outlet_clean" in df.columns:
        vc=df["media_outlet_clean"].value_counts()
        print("\nArticles per outlet (clusters):")
        print(vc)
    if "media_category" in df.columns:
        vc2=df["media_category"].value_counts()
        print("\nArticles per category:")
        print(vc2)
    print()

###############################################################################
# correlation analyses
###############################################################################
def analyze_2fields_correlation(df, left_field_pattern, right_field,
                                correlation_title, output_excel_base, prefix=""):
    ...
    # Implementation same as prior. Full expansions.

def analyze_all_custom_correlations(df, prefix=""):
    ...
    # Implementation with QFTC, QIvFI, FIvF.

###############################################################################
# aggregator
###############################################################################
def aggregate_sentiment_scores(df, sentiments):
    ...
    # Implementation

def calculate_averages(agg_df):
    ...
    # Implementation

def calculate_mean_median(agg_df):
    ...
    # Implementation

def save_aggregated_scores_to_csv(agg_df, out_dir, prefix=""):
    ...
    # Implementation

def plot_statistics(agg_df, out_dir, prefix=""):
    ...
    # Implementation

###############################################################################
# M&D Correction
###############################################################################
def mancl_derouen_correction(gee_result):
    ...
    # Implementation

def bootstrap_gee_md(df, formula, group_col,
                     B=200, family=Poisson(), cov_struct=Independence()):
    ...
    # Implementation

###############################################################################
# coverage checks & cluster checks
###############################################################################
def coverage_simulation(...):
    ...
    # Implementation

def check_number_of_clusters(...):
    ...
    # Implementation

def check_cluster_balance(...):
    ...
    # Implementation

###############################################################################
# scale computations
###############################################################################
def compute_pearson_scale(...):
    ...
    # Implementation

def compute_deviance_scale(...):
    ...
    # Implementation

def compute_ub_scale(...):
    ...
    # Implementation

def compute_bc_scale(...):
    ...
    # Implementation

###############################################################################
# check_residuals_and_correlation
###############################################################################
def check_residuals_and_correlation(final_fit):
    ...
    # Implementation

def pseudo_likelihood_check(...):
    ...
    # Implementation

###############################################################################
# cross_validation_gee
###############################################################################
def cross_validation_gee(...):
    ...
    # Implementation

###############################################################################
# sensitivity_analysis_correlation
###############################################################################
def sensitivity_analysis_correlation(...):
    ...
    # Implementation

###############################################################################
# try_all_families_and_scales
###############################################################################
def try_all_families_and_scales(...):
    ...
    # Implementation

###############################################################################
# refit_best_gee_with_scale
###############################################################################
def refit_best_gee_with_scale(df, sentiment, measure, fam_name, struct, scale_name):
    """
    Expands the short placeholder with the full logic:
    1) define d2['_score_col']
    2) define family + cov structure
    3) fit => possibly re-fit with scale
    """
    from statsmodels.genmod.families import Poisson, Gaussian, Gamma, InverseGaussian
    from statsmodels.genmod.families.family import NegativeBinomial

    # 1) define _score_col
    if measure=="Quotation":
        pat=rf"^{re.escape(sentiment)}_\d+$"
        matched=[c for c in df.columns if re.match(pat,c)]
        if not matched:
            return None,None
        d2=df.copy()
        d2["_score_col"]=d2[matched].clip(lower=0).mean(axis=1)
    elif measure=="Fulltext":
        fcol=f"{sentiment}_fulltext"
        if fcol not in df.columns:
            return None,None
        d2=df.copy()
        d2["_score_col"]=d2[fcol].clip(lower=0)
    elif measure=="Fulltext_Intensity":
        fcol=f"{sentiment}_fulltext_intensity"
        if fcol not in df.columns:
            return None,None
        d2=df.copy()
        d2["_score_col"]=d2[fcol].astype(float).clip(lower=0)
    elif measure=="Title_Intensity":
        fcol=f"{sentiment}_title_intensity"
        if fcol not in df.columns:
            return None,None
        d2=df.copy()
        d2["_score_col"]=d2[fcol].astype(float).clip(lower=0)
    elif measure=="Quotation_Intensity":
        pat=rf"^{re.escape(sentiment)}_\d+_intensity$"
        matched=[c for c in df.columns if re.match(pat,c)]
        if not matched:
            return None,None
        d2=df.copy()
        d2["_score_col"]=d2[matched].astype(float).clip(lower=0).mean(axis=1)
    else:
        return None,None

    needed=["_score_col","media_outlet_clean","media_category"]
    d2=d2.dropna(subset=needed)
    if len(d2)<2 or d2["media_category"].nunique()<2:
        return None,None

    # 2) define family
    if fam_name=="Poisson":
        fam_obj=Poisson()
    elif fam_name=="NegativeBinomial":
        fam_obj=NegativeBinomial(alpha=1.0)
    elif fam_name=="Gaussian":
        fam_obj=Gaussian()
    elif fam_name=="Gamma":
        fam_obj=Gamma(link=ln.log())
    elif fam_name=="InverseGaussian":
        fam_obj=InverseGaussian()
    else:
        logging.warning(f"Unknown family={fam_name}")
        return None,None

    # 3) define correlation
    if struct=="Independence":
        cov_obj=Independence()
    else:
        cov_obj=Exchangeable()

    # 4) initial fit
    model=GEE.from_formula("_score_col ~ media_category",
                           groups="media_outlet_clean",
                           data=d2,
                           family=fam_obj,
                           cov_struct=cov_obj)
    bres=model.fit(maxiter=300, scale=None)
    y=np.asarray(bres.model.endog)
    mu=np.asarray(bres.fittedvalues)
    n=len(y)
    p=len(bres.params)
    dfresid=n-p

    # 5) re-fit with scale if needed
    if scale_name=="none" or dfresid<=0:
        return d2,bres
    else:
        val=None
        if scale_name=="pearson":
            val=compute_pearson_scale(y,mu,dfresid)
        elif scale_name=="deviance":
            val=compute_deviance_scale(y,mu,dfresid)
        elif scale_name=="ub":
            val=compute_ub_scale(y,mu,dfresid)
        elif scale_name=="bc":
            val=compute_bc_scale(y,mu,dfresid)

        if val is not None and not np.isnan(val):
            final_res=model.fit(maxiter=300, scale=val)
            return d2, final_res
        else:
            return d2,bres

###############################################################################
# run_gee_for_sentiment_measure_best_qic
###############################################################################
def run_gee_for_sentiment_measure_best_qic(df, sentiment, measure):
    """
    1) define _score_col for a given measure
    2) run try_all_families_and_scales => pick best QIC
    3) return a dict with (Sentiment,Measure,Best_Family,etc.)
    """
    d2=df.copy()
    if measure=="Quotation":
        pat=rf"^{re.escape(sentiment)}_\d+$"
        matched=[c for c in d2.columns if re.match(pat,c)]
        if not matched:
            return None
        d2["_score_col"]=d2[matched].clip(lower=0).mean(axis=1)
    elif measure=="Fulltext":
        fcol=f"{sentiment}_fulltext"
        if fcol not in d2.columns:
            return None
        d2["_score_col"]=d2[fcol].clip(lower=0)
    elif measure=="Fulltext_Intensity":
        fcol=f"{sentiment}_fulltext_intensity"
        if fcol not in d2.columns:
            return None
        d2["_score_col"]=d2[fcol].astype(float).clip(lower=0)
    elif measure=="Title_Intensity":
        fcol=f"{sentiment}_title_intensity"
        if fcol not in d2.columns:
            return None
        d2["_score_col"]=d2[fcol].astype(float).clip(lower=0)
    elif measure=="Quotation_Intensity":
        pat=rf"^{re.escape(sentiment)}_\d+_intensity$"
        matched=[c for c in d2.columns if re.match(pat,c)]
        if not matched:
            return None
        d2["_score_col"]=d2[matched].astype(float).clip(lower=0).mean(axis=1)
    else:
        return None

    needed=["_score_col","media_outlet_clean","media_category"]
    d2=d2.dropna(subset=needed)
    if len(d2)<2 or d2["media_category"].nunique()<2:
        return None

    best_tuple, combos=try_all_families_and_scales(d2,"_score_col ~ media_category","media_outlet_clean")
    if best_tuple is None:
        return None

    famName, corName, scOpt, bestQICVal=best_tuple
    combos_df=pd.DataFrame(combos, columns=["Family","CovStruct","Scale","QIC"])

    return {
        "Sentiment":sentiment,
        "Measure":measure,
        "Best_Family":famName,
        "Best_Structure":corName,
        "Best_Scale":scOpt,
        "Best_QIC_main":bestQICVal,
        "AllCombos": combos_df
    }

###############################################################################
# run_gee_analyses_best_qic
###############################################################################
def run_gee_analyses_best_qic(df):
    """
    For each sentiment & measure => run_gee_for_sentiment_measure_best_qic
    Collate the best QIC => df_best
    Collate combos => df_all
    """
    measure_list=[
        "Quotation",
        "Fulltext",
        "Fulltext_Intensity",
        "Title_Intensity",
        "Quotation_Intensity"
    ]
    best_list=[]
    combos_list=[]
    for s in CATEGORIES:
        for meas in measure_list:
            info=run_gee_for_sentiment_measure_best_qic(df,s,meas)
            if info is not None:
                best_list.append({
                    "Sentiment":info["Sentiment"],
                    "Measure":info["Measure"],
                    "Best_Family":info["Best_Family"],
                    "Best_Structure":info["Best_Structure"],
                    "Best_Scale":info["Best_Scale"],
                    "Best_QIC_main":info["Best_QIC_main"]
                })
                cdf=info["AllCombos"]
                cdf["Sentiment"]=s
                cdf["Measure"]=meas
                combos_list.append(cdf)

    df_best=pd.DataFrame(best_list)
    df_all=pd.concat(combos_list, ignore_index=True) if combos_list else pd.DataFrame()
    return df_best, df_all

###############################################################################
# exclude_small_clusters
###############################################################################
def exclude_small_clusters(df, cluster_col="media_outlet_clean", min_size=5, merge_into="OtherSmall"):
    """
    If cluster size < min_size => re-label cluster to 'OtherSmall'
    """
    df2=df.copy()
    sizes=df2[cluster_col].value_counts()
    small_clusters=sizes[sizes<min_size].index
    df2[cluster_col]=df2[cluster_col].apply(lambda c: merge_into if c in small_clusters else c)
    return df2

###############################################################################
# pairwise_and_diagnostics => fill COMPARE_IMBALANCE
###############################################################################
def pairwise_and_diagnostics(df, sentiment, measure, fam_name, struct, scale_name,
                             prefix="", isSens=False):
    from statsmodels.stats.multitest import multipletests
    global COMPARE_IMBALANCE

    d2, final_fit = refit_best_gee_with_scale(df, sentiment, measure, fam_name, struct, scale_name)
    if final_fit is None:
        return (None, None, None, None, None, None, None, None)

    summary_txt = final_fit.summary().as_text()
    diag_dict = check_residuals_and_correlation(final_fit)
    pseudo_dict = pseudo_likelihood_check(final_fit)
    cv_val = cross_validation_gee(d2,"_score_col ~ media_category","media_outlet_clean",
                                  final_fit.model.family, final_fit.model.cov_struct, n_folds=3)
    sens_df = sensitivity_analysis_correlation(d2,"_score_col ~ media_category","media_outlet_clean")

    # cluster bootstrap M&D => compare
    param_names = final_fit.params.index
    param_values= final_fit.params.values
    robust_cov = final_fit.cov_params()
    robust_se = np.sqrt(np.diag(robust_cov))

    # do the M&D cluster bootstrap
    boot_params_df, boot_se_df = bootstrap_gee_md(
        d2, "_score_col ~ media_category", "media_outlet_clean", B=200,
        family=final_fit.model.family, cov_struct=final_fit.model.cov_struct
    )
    param_stds = boot_params_df.std().round(4)

    # fill COMPARE_IMBALANCE
    for i, p in enumerate(param_names):
        sr_se=robust_se[i]
        sr_val=param_values[i]
        bs_std=param_stds.get(p, np.nan)
        ratio=np.nan
        if sr_se>0 and not np.isnan(sr_se):
            ratio = bs_std / sr_se
        COMPARE_IMBALANCE.append({
            "Sentiment":sentiment,
            "Measure":measure,
            "Param":p,
            "Prefix":prefix,
            "IsSens":isSens,
            "SingleRun_SE": sr_se,
            "MD_BootStd":   bs_std,
            "Ratio": ratio,
            "ParamEst": sr_val
        })

    param_mean_dict = boot_params_df.mean().round(4).to_dict()
    param_std_dict  = param_stds.to_dict()
    boot_info = {
        "ParamMean": param_mean_dict,
        "ParamStd":  param_std_dict
    }

    # pairwise differences
    params = final_fit.params
    cov = final_fit.cov_params()
    mdf = final_fit.model.data.frame
    cats = mdf["media_category"].unique()
    if pd.api.types.is_categorical_dtype(mdf["media_category"]):
        cats = mdf["media_category"].cat.categories

    ref = cats[0]
    idx_map={ref:0}
    for c in cats[1:]:
        nm=f"media_category[T.{c}]"
        if nm in final_fit.model.exog_names:
            idx_map[c]=final_fit.model.exog_names.index(nm)

    pair_list=[]
    for i2 in range(len(cats)):
        for j2 in range(i2+1,len(cats)):
            ca,cb = cats[i2], cats[j2]
            if ca not in idx_map and cb not in idx_map:
                continue
            con=np.zeros(len(params))
            if ca==ref and cb in idx_map:
                con[idx_map[cb]]=-1.0
            elif cb==ref and ca in idx_map:
                con[idx_map[ca]]=1.0
            else:
                if ca in idx_map: con[idx_map[ca]]=1.0
                if cb in idx_map: con[idx_map[cb]]=-1.0

            diff_est= con@params
            diff_var= con@cov@con
            if diff_var<=1e-12:
                diff_se=np.nan
                z=np.nan
                pval=np.nan
            else:
                diff_se=np.sqrt(diff_var)
                z=diff_est/diff_se
                pval=2*(1-norm.cdf(abs(z)))
            pair_list.append((ca,cb,diff_est,diff_se,z,pval))

    pair_df = pd.DataFrame(pair_list, columns=["CategoryA","CategoryB","Difference","SE","Z","p_value"])
    if not pair_df.empty:
        rej, p_adj,_,_= multipletests(pair_df["p_value"], method="fdr_bh")
        pair_df["p_value_adj"]=p_adj
        pair_df["reject_H0"]=rej
    else:
        pair_df["p_value_adj"]=np.nan
        pair_df["reject_H0"]=False

    diff_map={c:set() for c in cats}
    cat_list=list(cats)
    cat_index_map={cat_list[i]: i+1 for i in range(len(cat_list))}

    for i3, row3 in pair_df.iterrows():
        A=row3["CategoryA"]
        B=row3["CategoryB"]
        if row3["reject_H0"]:
            diff_map[A].add(cat_index_map[B])
            diff_map[B].add(cat_index_map[A])

    rows=[]
    for c in cat_list:
        diffs= sorted(list(diff_map[c]))
        diffs_str=",".join(str(x) for x in diffs)
        rows.append((c,diffs_str))
    diffIDs_df = pd.DataFrame(rows, columns=["Category","DiffIDs"])

    return summary_txt, pair_df, diffIDs_df, diag_dict, pseudo_dict, cv_val, sens_df, boot_info

###############################################################################
# compile_results_into_multiple_workbooks
###############################################################################
def compile_results_into_multiple_workbooks(
    aggregated_df, stats_df, raw_df,
    df_best_qic, df_all_combos,
    plots_dir,
    main_excel, raw_excel, gee_excel, plots_excel, combined_excel,
    df_full,
    validation_records=None
):
    ...
    # Full code from prior expansions

###############################################################################
# append_sens_to_excel
###############################################################################
def append_sens_to_excel(
    aggregated_df_sens, stats_df_sens, raw_df_sens,
    df_best_qic_sens, df_all_combos_sens,
    prefix,
    outdir,
    main_excel,
    raw_excel,
    gee_excel,
    combined_excel,
    df_full_sens
):
    ...
    # Full code from prior expansions

###############################################################################
# run_pipeline_for_df => merges small clusters => store SENS in same files
###############################################################################
def run_pipeline_for_df(
    df,
    prefix="",
    main_excel=BASE_MAIN_FILE,
    raw_excel=BASE_RAW_FILE,
    gee_excel=BASE_GEE_FILE,
    plots_excel=BASE_PLOTS_FILE,
    combined_excel=BASE_COMBINED_FILE,
    outdir="graphs_analysis",
    csvout="csv_raw_scores",
    min_size=5
):
    # define final outputs in subdir => rename with prefix
    main_excel    = rename_files_for_prefix(main_excel,    prefix)
    raw_excel     = rename_files_for_prefix(raw_excel,     prefix)
    gee_excel     = rename_files_for_prefix(gee_excel,     prefix)
    plots_excel   = rename_files_for_prefix(plots_excel,   prefix)
    combined_excel= rename_files_for_prefix(combined_excel,prefix)

    validation_records=[]

    # cluster checks, coverage
    c1=check_number_of_clusters(df,"media_outlet_clean",20)
    validation_records.append(c1)
    c2=check_cluster_balance(df,"media_outlet_clean",5.0)
    validation_records.append(c2)
    c_cov=coverage_simulation(n_sims=1000,n_clusters=10,cluster_size=10,true_beta=0.5)
    validation_records.append(c_cov)

    # if prefix indicates "Yes"
    if prefix=="Yes" or (prefix.endswith("_collapsed") and prefix.startswith("Yes_")):
        cat_codes, unique_cats = pd.factorize(df["media_category"])
        df["media_outlet_clean"] = cat_codes + 1
        logging.info(f"Collapsed each category => single cluster. #unique cats={len(unique_cats)}")

    chunk_and_save(df,20000,prefix=prefix)
    print_basic_stats(df,prefix=prefix)

    analyze_all_custom_correlations(df,prefix=prefix)

    agg_df=aggregate_sentiment_scores(df,CATEGORIES)
    agg_df=calculate_averages(agg_df)
    stats_df=calculate_mean_median(agg_df)
    save_aggregated_scores_to_csv(agg_df, csvout, prefix=prefix)
    plot_statistics(agg_df, outdir, prefix=prefix)

    df_best, df_allcombos=run_gee_analyses_best_qic(df)
    print(f"Best QIC => prefix={prefix}")
    print(df_best)

    # parent compile
    compile_results_into_multiple_workbooks(
        aggregated_df=agg_df,
        stats_df=stats_df,
        raw_df=df,
        df_best_qic=df_best,
        df_all_combos=df_allcombos,
        plots_dir=outdir,
        main_excel=main_excel,
        raw_excel=raw_excel,
        gee_excel=gee_excel,
        plots_excel=plots_excel,
        combined_excel=combined_excel,
        df_full=df,
        validation_records=validation_records
    )

    # SENS => merges small clusters
    df_sens=exclude_small_clusters(df,"media_outlet_clean", min_size=min_size, merge_into="OtherSmall")
    if len(df_sens)<2:
        print(f"[Sensitivity] => merging <{min_size} => not enough data => skip.")
        return

    print(f"[Sensitivity] => merging <{min_size}, prefix={prefix}, len={len(df_sens)} => SENS run")

    agg_sens=aggregate_sentiment_scores(df_sens,CATEGORIES)
    agg_sens=calculate_averages(agg_sens)
    stats_sens=calculate_mean_median(agg_sens)

    df_best_sens, df_allcombos_sens=run_gee_analyses_best_qic(df_sens)
    print(f"Best QIC => Sens run, prefix={prefix}")
    print(df_best_sens)

    # store aggregator CSV => e.g. "aggregated_sentiment_emotion_scores_Yes_Sens.csv"
    save_aggregated_scores_to_csv(agg_sens, csvout, prefix=f"{prefix}_Sens")

    # now append to same files
    append_sens_to_excel(
        aggregated_df_sens=agg_sens,
        stats_df_sens=stats_sens,
        raw_df_sens=df_sens,
        df_best_qic_sens=df_best_sens,
        df_all_combos_sens=df_allcombos_sens,
        prefix=prefix,
        outdir=outdir,
        main_excel=main_excel,
        raw_excel=raw_excel,
        gee_excel=gee_excel,
        combined_excel=combined_excel,
        df_full_sens=df_sens
    )

###############################################################################
# main => yes/all => yes_collapsed/all_collapsed
###############################################################################
def main():
    setup_logging()
    logging.info("Starting pipeline => yes/all => original cats, then yes/all => collapsed cats")

    df_raw=load_jsonl(INPUT_JSONL_FILE)
    print(f"Total articles loaded => {len(df_raw)}")

    if "high_rate_2" in df_raw.columns:
        df_raw["high_rate_2"]=df_raw["high_rate_2"].astype(str).str.strip().str.lower()

    # A) Original categories
    print("\n=== A) Original Categories (6-cat) ===")
    df_orig=map_media_outlet_to_category(df_raw, MEDIA_CATEGORIES_ORIG)

    if "high_rate_2" in df_orig.columns:
        df_yes_orig=df_orig[df_orig["high_rate_2"]=="yes"].copy()
        print(f"df_yes_orig => {len(df_yes_orig)} rows")
        if len(df_yes_orig)>0:
            print("**Pipeline => Yes** (orig cats)")
            run_pipeline_for_df(df_yes_orig, prefix="Yes")
        else:
            print("No rows => skip yes subset (orig cats)")
    else:
        print("No 'high_rate_2' => skip yes subset (orig cats)")

    print("**Pipeline => All** (orig cats)")
    run_pipeline_for_df(df_orig, prefix="All")
    del df_orig
    gc.collect()

    # B) Collapsed categories
    print("\n=== B) Collapsed Categories (4-cat) ===")
    df_coll=map_media_outlet_to_category(df_raw, MEDIA_CATEGORIES_COLLAPSED)

    if "high_rate_2" in df_coll.columns:
        df_yes_coll=df_coll[df_coll["high_rate_2"]=="yes"].copy()
        print(f"df_yes_collapsed => {len(df_yes_coll)} rows")
        if len(df_yes_coll)>0:
            print("**Pipeline => Yes_collapsed**")
            run_pipeline_for_df(df_yes_coll, prefix="Yes_collapsed")
        else:
            print("No rows => skip yes subset (collapsed cats)")
    else:
        print("No 'high_rate_2' => skip yes subset (collapsed cats)")

    print("**Pipeline => All_collapsed**")
    run_pipeline_for_df(df_coll, prefix="All_collapsed")
    del df_coll
    gc.collect()

    logging.info("All done => subset + full => original cats + collapsed cats.")


if __name__=="__main__":
    main()

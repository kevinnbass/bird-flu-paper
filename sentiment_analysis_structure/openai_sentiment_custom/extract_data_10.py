# If group has Quotation => forcibly blank Fulltext, Title_Intensity if Quotation=NaN
if "Quotation" in measure_group:
    for cat in cat_names:
        for a_id,qval in measure_data["Quotation"][cat].items():
            if pd.isna(qval):
                if "Fulltext" in measure_group:
                    measure_data["Fulltext"][cat][a_id] = np.nan
                if "Title_Intensity" in measure_group:
                    measure_data["Title_Intensity"][cat][a_id] = np.nan

# If group has Quotation_Intensity => forcibly blank Fulltext_Intensity, Title_Intensity if Quotation_Intensity=NaN
if "Quotation_Intensity" in measure_group:
    for cat in cat_names:
        for a_id,qi_val in measure_data["Quotation_Intensity"][cat].items():
            if pd.isna(qi_val):
                if "Fulltext_Intensity" in measure_group:
                    measure_data["Fulltext_Intensity"][cat][a_id] = np.nan
                if "Title_Intensity" in measure_group:
                    measure_data["Title_Intensity"][cat][a_id] = np.nan

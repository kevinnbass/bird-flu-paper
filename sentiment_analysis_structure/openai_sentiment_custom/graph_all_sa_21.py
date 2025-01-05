def main():
    # Declare globals at the very start of main
    global OUTPUT_EXCEL_MAIN, OUTPUT_EXCEL_RAW, OUTPUT_EXCEL_LMM, OUTPUT_EXCEL_PLOTS, OUTPUT_EXCEL_COMBINED

    setup_logging()
    print("Loading data from JSONL file...")
    try:
        df = load_jsonl(INPUT_JSONL_FILE)
        print(f"Total articles loaded: {len(df)}")
        logging.info(f"Total articles loaded: {len(df)}")
    except Exception as e:
        print(f"Error loading data: {e}")
        logging.error(f"Error loading data: {e}")
        return

    if df.empty:
        print("The input file is empty. Exiting.")
        logging.info("The input file is empty. Exiting.")
        return

    print("Mapping media outlets to categories...")
    try:
        df = map_media_outlet_to_category(df, MEDIA_CATEGORIES)
        print("Media outlets mapped to categories successfully.\n")
        logging.info("Media outlets mapped to categories successfully.")
    except Exception as e:
        print(f"Error mapping media outlets to categories: {e}")
        logging.error(f"Error mapping media outlets to categories: {e}")
        return

    print("Aggregating sentiment/emotion scores per media category...")
    try:
        aggregated_df = aggregate_sentiment_scores(df, CATEGORIES)
        aggregated_df = calculate_averages(aggregated_df)
        print("Aggregation of sentiment/emotion scores completed successfully.\n")
        logging.info("Aggregation of sentiment/emotion scores completed successfully.")
    except Exception as e:
        print(f"Error aggregating sentiment/emotion scores: {e}")
        logging.error(f"Error aggregating sentiment/emotion scores: {e}")
        return

    print("Calculating mean and median statistics...")
    try:
        stats_df = calculate_mean_median(aggregated_df)
        print("Mean and median statistics calculated successfully.\n")
        logging.info("Mean and median statistics calculated successfully.")
    except Exception as e:
        print(f"Error calculating statistics: {e}")
        logging.error(f"Error calculating statistics: {e}")
        return

    print("Saving aggregated scores to CSV files...")
    try:
        save_aggregated_scores_to_csv(aggregated_df, CSV_OUTPUT_DIR)
        print("Aggregated scores saved to CSV files successfully.\n")
        logging.info("Aggregated scores saved to CSV files successfully.")
    except Exception as e:
        print(f"Error saving aggregated scores to CSV: {e}")
        logging.error(f"Error saving aggregated scores to CSV: {e}")
        return

    print("Generating plots for statistics...")
    try:
        plot_statistics(aggregated_df, OUTPUT_DIR)
        print("Plots generated successfully.\n")
        logging.info("Plots generated successfully.")
    except Exception as e:
        print(f"Error generating plots: {e}")
        logging.error(f"Error generating plots: {e}")
        return

    print("Filtering outlets with insufficient observations...")
    outlet_counts = df['media_outlet'].value_counts()
    outlets_to_remove = outlet_counts[outlet_counts < 2].index.tolist()
    if outlets_to_remove:
        print("Filtering out outlets with fewer than 2 observations:")
        logging.info("Filtering out outlets with fewer than 2 observations:")
        for outlet in outlets_to_remove:
            obs_count = outlet_counts[outlet]
            print(f" - Removing {outlet}: {obs_count} observation(s)")
            logging.info(f" - Removing {outlet}: {obs_count} observation(s)")
        initial_count = len(df)
        df = df[~df['media_outlet'].isin(outlets_to_remove)]
        final_count = len(df)
        removed_count = initial_count - final_count
        print(f"Filtered out {removed_count} total observations from {len(outlets_to_remove)} outlet(s).")
        logging.info(f"Filtered out {removed_count} total observations from {len(outlets_to_remove)} outlet(s).")
    else:
        print("No outlets with fewer than 2 observations found.")
        logging.info("No outlets with fewer than 2 observations found.")

    predictor_columns = []
    check_collinearity(df, predictor_columns)

    print("Fitting linear mixed models and running post-hoc tests...")
    try:
        lmm_results = run_lmm_analyses(df)
        print("LMM analysis and post-hoc tests completed successfully.\n")
        logging.info("LMM analysis and post-hoc tests completed successfully.")
    except Exception as e:
        print(f"Error in LMM analysis: {e}")
        logging.error(f"Error in LMM analysis: {e}")
        return

    print("Compiling statistics, raw data, LMM results, and plots into multiple Excel files...")
    try:
        compile_results_into_multiple_workbooks(aggregated_df, stats_df, df, lmm_results, OUTPUT_DIR)
        logging.info("All results compiled into multiple Excel files successfully.")
    except Exception as e:
        print(f"Error compiling statistics and plots into separate Excel files: {e}")
        logging.error(f"Error compiling statistics and plots into separate Excel files: {e}")
        return

    print("Now compiling everything into a single combined Excel workbook as well...")
    try:
        compile_into_single_combined_workbook(aggregated_df, stats_df, df, lmm_results, OUTPUT_DIR)
        print(f"All results compiled into a single combined Excel file: {OUTPUT_EXCEL_COMBINED}")
        logging.info(f"All results compiled into a single combined Excel file: {OUTPUT_EXCEL_COMBINED}")
    except Exception as e:
        print(f"Error compiling single combined Excel file: {e}")
        logging.error(f"Error compiling single combined Excel file: {e}")

    print("Analysis completed successfully.")
    logging.info("Analysis completed successfully.")

    if problematic_outlets:
        print("\nOutlets identified with insufficient variation:", problematic_outlets)

        df_filtered = df[~df['media_outlet'].isin(problematic_outlets)].copy()

        print("\n--- Running analysis on filtered dataset ---")

        filtered_output_dir = 'graphs_analysis_filtered'
        os.makedirs(filtered_output_dir, exist_ok=True)

        filtered_csv_dir = 'csv_raw_scores_filtered'
        os.makedirs(filtered_csv_dir, exist_ok=True)

        # Temporarily store original file paths
        orig_main = OUTPUT_EXCEL_MAIN
        orig_raw = OUTPUT_EXCEL_RAW
        orig_lmm = OUTPUT_EXCEL_LMM
        orig_plots = OUTPUT_EXCEL_PLOTS
        orig_combined = OUTPUT_EXCEL_COMBINED

        # Assign filtered output names
        OUTPUT_EXCEL_MAIN_FILTERED = 'analysis_main_filtered.xlsx'
        OUTPUT_EXCEL_RAW_FILTERED = 'analysis_raw_filtered.xlsx'
        OUTPUT_EXCEL_LMM_FILTERED = 'analysis_lmm_filtered.xlsx'
        OUTPUT_EXCEL_PLOTS_FILTERED = 'analysis_plots_filtered.xlsx'
        OUTPUT_EXCEL_COMBINED_FILTERED = 'analysis_combined_filtered.xlsx'

        # Reassign to filtered variants
        OUTPUT_EXCEL_MAIN = OUTPUT_EXCEL_MAIN_FILTERED
        OUTPUT_EXCEL_RAW = OUTPUT_EXCEL_RAW_FILTERED
        OUTPUT_EXCEL_LMM = OUTPUT_EXCEL_LMM_FILTERED
        OUTPUT_EXCEL_PLOTS = OUTPUT_EXCEL_PLOTS_FILTERED
        OUTPUT_EXCEL_COMBINED = OUTPUT_EXCEL_COMBINED_FILTERED

        filtered_aggregated_df = aggregate_sentiment_scores(df_filtered, CATEGORIES)
        filtered_aggregated_df = calculate_averages(filtered_aggregated_df)
        filtered_stats_df = calculate_mean_median(filtered_aggregated_df)
        save_aggregated_scores_to_csv(filtered_aggregated_df, filtered_csv_dir)
        plot_statistics(filtered_aggregated_df, filtered_output_dir)
        filtered_lmm_results = run_lmm_analyses(df_filtered)
        compile_results_into_multiple_workbooks(filtered_aggregated_df, filtered_stats_df, df_filtered, filtered_lmm_results, filtered_output_dir)
        compile_into_single_combined_workbook(filtered_aggregated_df, filtered_stats_df, df_filtered, filtered_lmm_results, filtered_output_dir)

        print("Filtered analysis completed. Compare the filtered outputs with the original outputs to assess differences.")

        # Restore original names
        OUTPUT_EXCEL_MAIN = orig_main
        OUTPUT_EXCEL_RAW = orig_raw
        OUTPUT_EXCEL_LMM = orig_lmm
        OUTPUT_EXCEL_PLOTS = orig_plots
        OUTPUT_EXCEL_COMBINED = orig_combined

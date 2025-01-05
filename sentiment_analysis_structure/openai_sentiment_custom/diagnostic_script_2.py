def try_fit_model(df, score_col):
    # Simple LMM
    if 'media_category' not in df or 'media_outlet' not in df:
        print("Missing required columns for LMM.")
        return

    if df['media_category'].nunique() < 2:
        print("Not enough categories for LMM.")
        return

    formula = f"{score_col} ~ media_category"

    # Construct design matrices for debugging
    y, X = dmatrices(formula, data=df, return_type='dataframe')
    X_rank = np.linalg.matrix_rank(X)
    print(f"[DEBUG] X design matrix shape: {X.shape}")
    print(f"[DEBUG] y vector shape: {y.shape}")
    print(f"[DEBUG] Rank of X: {X_rank}")

    # Align df with the rows in X and y:
    df = df.iloc[X.index].copy()
    # Flatten y to a 1D array if needed
    y = y.values.ravel()

    try:
        # Now fit the model using the aligned df
        md = smf.mixedlm(formula, data=df, groups=df["media_outlet"])
        mdf = md.fit(reml=True, method='lbfgs')
        if not mdf.converged:
            print("Model did not converge.")
        else:
            print("Model converged successfully.")
    except np.linalg.LinAlgError as e:
        print(f"Model failed with a linear algebra error (likely singular): {e}")
    except Exception as e:
        print(f"Model failed: {e}")

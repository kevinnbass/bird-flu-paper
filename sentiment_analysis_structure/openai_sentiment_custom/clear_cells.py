import pandas as pd

def remove_blanks_in_column(df):
    """
    Given a DataFrame, remove blank (NaN) cells in each column 
    by shifting non-blank values up.
    """
    # Drop NaNs in each column, then reset index
    for column in df.columns:
        # Dropna in that column
        non_blank_series = df[column].dropna().reset_index(drop=True)
        # Create a new column with the correct length
        df[column] = non_blank_series.reindex(range(len(df)))
    return df

def main():
    input_file = 'pos-neg_sentiment.xlsx'
    output_file = 'pos-neg_sentiment_cleaned.xlsx'
    
    # Read the file to access its sheet names
    xls = pd.ExcelFile(input_file)
    sheet_names = xls.sheet_names
    
    # Dictionary to hold cleaned DataFrames by sheet
    cleaned_sheets = {}
    
    # Loop through all sheets
    for sheet in sheet_names:
        # Read the sheet into a DataFrame
        df = pd.read_excel(input_file, sheet_name=sheet)
        
        # Remove blank cells by shifting the values up in each column
        df_cleaned = remove_blanks_in_column(df)
        
        # Store the cleaned DataFrame
        cleaned_sheets[sheet] = df_cleaned
    
    # Write all cleaned sheets to a new Excel file
    with pd.ExcelWriter(output_file) as writer:
        for sheet, df_cleaned in cleaned_sheets.items():
            df_cleaned.to_excel(writer, sheet_name=sheet, index=False)

if __name__ == "__main__":
    main()

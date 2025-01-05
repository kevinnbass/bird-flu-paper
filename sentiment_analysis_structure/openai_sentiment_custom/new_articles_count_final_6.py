import json
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import warnings
import numpy as np

# Suppress FutureWarnings related to Pandas resampling
warnings.simplefilter(action='ignore', category=FutureWarning)

# Specify the path to your JSON file
json_file_path = 'final_processed_all_articles.json'  # Updated input file

# Load the JSON data
try:
    with open(json_file_path, 'r') as file:
        articles = json.load(file)
    print(f"Successfully loaded '{json_file_path}'.")
except FileNotFoundError:
    raise FileNotFoundError(f"The file '{json_file_path}' was not found.")
except json.JSONDecodeError:
    raise ValueError(f"The file '{json_file_path}' contains invalid JSON.")

# Convert the list of articles to a DataFrame
df = pd.DataFrame(articles)

# Check if required columns exist
required_columns = ['date', 'media_outlet']
missing_columns = [col for col in required_columns if col not in df.columns]
if missing_columns:
    raise ValueError(f"The following required fields are missing from the JSON data: {missing_columns}")
print("Data validation passed. All required fields are present.")

# Convert the 'date' column to datetime objects
df['date'] = pd.to_datetime(df['date'], errors='coerce')

# Drop rows with invalid dates
initial_count = len(df)
df = df.dropna(subset=['date'])
dropped_count = initial_count - len(df)
if dropped_count > 0:
    print(f"Dropped {dropped_count} articles due to invalid dates.")

# Ensure 'media_outlet' is a string and strip whitespace
df['media_outlet'] = df['media_outlet'].astype(str).str.strip()

# Define the start and end dates
start_date = '2020-01-01'
end_date = '2024-09-30'

# Generate a complete list of months from start_date to end_date
all_months = pd.date_range(start=start_date, end=end_date, freq='MS')  # 'MS' stands for Month Start
all_months_str = all_months.strftime('%b %Y')

# Create a 'month' column in 'Mon YYYY' format
df['month'] = df['date'].dt.strftime('%b %Y')  # e.g., 'Jan 2020'

# Aggregate total articles per month
monthly_counts = df.groupby('month').size().reset_index(name='counts')

# Create a DataFrame for all months
all_months_df = pd.DataFrame({'month': all_months_str})

# Merge the aggregated counts with the complete list of months
merged_df = pd.merge(all_months_df, monthly_counts, on='month', how='left')

# Replace NaN counts with zero for months with no articles
merged_df['counts'] = merged_df['counts'].fillna(0).astype(int)

# Debugging: Print the first few rows to verify
print("\nAggregated Data (First 5 Months):")
print(merged_df.head())

# Verify that merged_df has non-zero counts
print("\nTotal Articles Per Month (First 5 Months):")
print(merged_df['counts'].head())

# Check for any months with zero total articles
zero_article_months = merged_df['counts'] == 0
if zero_article_months.any():
    print("\nMonths with zero articles:")
    print(merged_df[zero_article_months]['month'].tolist())

# Define the list of months and counts
months = merged_df['month'].tolist()
counts = merged_df['counts'].tolist()

# Plotting the bar graph
plt.figure(figsize=(25, 12))  # Adjust figure size as needed

# Plot bars in pastel blue
pastel_blue = '#AEC6CF'  # HEX code for a pastel blue shade
plt.bar(months, counts, color=pastel_blue, edgecolor='white')

# Update the font sizes for the title
plt.title('Total number of articles published each month (Jan 2020 - Sept 2024)', fontsize=36)  # Increased by 50%

# Remove x-axis label by not setting it
# plt.xlabel('Month', fontsize=36)  # Removed

# Update the y-axis label with doubled font size
plt.ylabel('Number of Articles', fontsize=36)

# Define step for x-ticks to label every 3 months
step = 3  # Label every 3 months
ticks = np.arange(0, len(months), step)
tick_labels = [months[i] for i in ticks]

plt.xticks(ticks, tick_labels, rotation=90, fontsize=24)  # Adjusted fontsize

plt.yticks(fontsize=28)  # Adjusted fontsize

plt.tight_layout()

# Remove the legend since there's only one category
# No legend is added in this plot

# Add gridlines for better readability
plt.grid(axis='y', linestyle='--', linewidth=0.5, alpha=0.7)

# Save the plot as a PNG file
output_png = 'total_articles_per_month_pastel_blue.png'  # Output filename remains the same
plt.savefig(output_png, dpi=300, bbox_inches='tight')
print(f"Bar graph has been saved as '{output_png}'.")

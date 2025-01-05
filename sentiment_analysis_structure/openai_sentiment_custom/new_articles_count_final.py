import json
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import warnings
import numpy as np

# Suppress FutureWarnings related to Pandas resampling
warnings.simplefilter(action='ignore', category=FutureWarning)

# Specify the path to your JSON file
json_file_path = 'final_processed_all_articles.json'

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

# Filter the DataFrame to include only dates within the range
mask = (df['date'] >= start_date) & (df['date'] <= end_date)
df = df.loc[mask]
print(f"Filtered data to include articles from {start_date} to {end_date}. Total articles: {len(df)}.")

# Check if any articles remain after filtering
if len(df) == 0:
    print("No articles found within the specified date range.")
    exit()

# Print the range of dates in the data
print(f"Date range in data: {df['date'].min()} to {df['date'].max()}.")

# Create a 'month' column in 'Mon YYYY' format
df['month'] = df['date'].dt.strftime('%b %Y')  # Changed from '%Y-%m' to '%b %Y'

# Get list of unique media outlets
media_outlets = df['media_outlet'].unique()
print(f"Found {len(media_outlets)} unique media outlets: {media_outlets}")

# Group by 'month' and 'media_outlet', count the number of articles
monthly_media_counts = df.groupby(['month', 'media_outlet']).size().reset_index(name='counts')

# Pivot the table to have 'month' as index and 'media_outlet' as columns
pivot_table = monthly_media_counts.pivot(index='month', columns='media_outlet', values='counts').fillna(0)

# Sort the pivot_table by date
# To sort correctly, convert 'month' back to datetime temporarily
pivot_table.index = pd.to_datetime(pivot_table.index, format='%b %Y')
pivot_table = pivot_table.sort_index()
pivot_table.index = pivot_table.index.strftime('%b %Y')  # Convert back to 'Mon YYYY' format

# Debugging: Print the first few rows to verify
print("\nAggregated Data (First 5 Months):")
print(pivot_table.head())

# Verify that pivot_table has non-zero counts
print("\nTotal Articles Per Month (First 5 Months):")
print(pivot_table.sum(axis=1).head())

# Check for any months with zero total articles
zero_article_months = pivot_table.sum(axis=1) == 0
if zero_article_months.any():
    print("\nMonths with zero articles:")
    print(pivot_table[zero_article_months].index.tolist())

# Define the list of media outlets again to ensure order
media_outlets = pivot_table.columns.tolist()

# Assign distinct colors to each media outlet using a color palette
# You can customize the color palette as desired
colors = plt.cm.tab20.colors  # A colormap with 20 distinct colors
if len(media_outlets) > len(colors):
    # If there are more media outlets than colors, extend the color list
    colors = colors * (len(media_outlets) // len(colors) + 1)
colors = colors[:len(media_outlets)]

# Create a dictionary mapping media outlets to colors
color_dict = dict(zip(media_outlets, colors))

# Plotting the stacked bar graph
plt.figure(figsize=(25, 12))  # Adjust figure size as needed

# Create a list to accumulate bottom positions for stacking
bottom = np.zeros(len(pivot_table))

# Plot each media outlet's counts as a separate segment in the stacked bar
for outlet in media_outlets:
    counts = pivot_table[outlet].values
    plt.bar(pivot_table.index, counts, bottom=bottom, color=color_dict[outlet], edgecolor='white', label=outlet)
    bottom += counts

# Update the font sizes for the title and axis labels
plt.title('Number of Articles Produced Each Month by Media Outlet (Jan 2020 - Sep 2024)', fontsize=48)  # Doubled from 24 to 48
plt.xlabel('Month', fontsize=36)  # Doubled from 18 to 36
plt.ylabel('Number of Articles', fontsize=36)  # Doubled from 18 to 36

# Define step for x-ticks to label every 3 months
step = 3  # Changed from 6 to 3
ticks = np.arange(0, len(pivot_table.index), step)
tick_labels = pivot_table.index[::step]

plt.xticks(ticks, tick_labels, rotation=90, fontsize=24)  # Doubled fontsize from 12 to 24
plt.yticks(fontsize=28)  # Doubled fontsize from 14 to 28

plt.tight_layout()

# Add a legend on the right-hand side
plt.legend(title='Media Outlet', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=28, title_fontsize=32)  # Increased fontsize

# Add gridlines for better readability
plt.grid(axis='y', linestyle='--', linewidth=0.5, alpha=0.7)

# Save the plot as a PNG file
output_png = 'articles_per_month_stacked_bar.png'
plt.savefig(output_png, dpi=300, bbox_inches='tight')
print(f"Stacked bar graph has been saved as '{output_png}'.")

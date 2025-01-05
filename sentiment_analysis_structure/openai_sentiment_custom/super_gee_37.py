r_val, p_val = pearsonr(allc["Left_Average"], allc["Right_Average"])

plt.figure(figsize=(6,5))
# Turn off gridlines
sns.set_style("white")
ax = sns.regplot(
    x="Left_Average",
    y="Right_Average",
    data=allc,
    scatter_kws={"color": "black", "s": 40},  # black points
    line_kws={"color": "red"}                # trendline in red
)
plt.grid(False)  # explicitly ensure no grid
# Display R and p
plt.text(
    0.05, 0.95,
    f"R = {r_val:.2f}, p = {p_val:.3g}",
    ha='left',
    va='top',
    transform=ax.transAxes
)
plt.title(f"{correlation_title} scatter - prefix={prefix}")
scattername=f"scatter_{prefix}_{correlation_title.replace(' ','_')}.png"
scatterpath=os.path.join(GRAPH_OUTPUT_DIR, scattername.lower())
plt.tight_layout()
try:
    plt.savefig(scatterpath)
except:
    pass
plt.close()

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import warnings

# --- Configuration & Setup ---
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# Keep output clean
warnings.filterwarnings("ignore", message="Parsing dates in .* format when dayfirst=.*")
warnings.filterwarnings('ignore', category=UserWarning, module='pandas')
warnings.filterwarnings('ignore', category=FutureWarning, module='seaborn')


def load_and_clean_data(filepath):
    """
    Reads the CSV and applies rigorous cleaning to handle the 'messy' data.
    """
    df = pd.read_csv(filepath)

    # 1. Clean Typos
    severity_map = {
        'Bloker': 'Blocker', 'Blcokr': 'Blocker', 'Critcal': 'Critical',
        'Majro': 'Major', 'Minro': 'Minor', 'Critcial': 'Critical',
        'Reopend': 'Reopened', 'Closd': 'Closed', 'Oepn': 'Open',
        'Rejetced': 'Rejected', 'Reopened': 'Reopened', 'Reopend': 'Reopened'
    }
    df['severity'] = df['severity'].map(severity_map).fillna(df['severity'])

    status_map = {
        'Closd': 'Closed', 'Reopend': 'Reopened', 'Oepn': 'Open',
        'Rejetced': 'Rejected'
    }
    df['status'] = df['status'].replace(status_map)

    # 2. Clean Dates
    df['created_date'] = pd.to_datetime(df['created_date'], dayfirst=True, errors='coerce')
    df['closed_date'] = pd.to_datetime(df['closed_date'], dayfirst=True, errors='coerce')

    # 3. Calculate 'Days to Close' & Remove Negative Durations
    df['days_to_close'] = (df['closed_date'] - df['created_date']).dt.days
    df.loc[df['days_to_close'] < 0, 'days_to_close'] = np.nan

    return df


# --- Refactored Insight Functions ---

def get_risk_insight(df):
    """
    Insight 1: High Risk Components (>30% Critical/Blocker)

    create a pivot table that groups the data by component and severity and count them
    unstacking them which allow to do arithmatic calculations across the row
    calculate the ratio and compare with the base ratio 0.3 to determine the level of risk
    """
    comp_sev = df.groupby(['component', 'severity']).size().unstack(fill_value=0)

    if 'Critical' in comp_sev.columns and 'Blocker' in comp_sev.columns:
        comp_sev['high_sev_sum'] = comp_sev['Critical'] + comp_sev['Blocker']
        comp_sev['total'] = comp_sev.sum(axis=1)
        # Avoid division by zero check
        comp_sev = comp_sev[comp_sev['total'] > 0]
        comp_sev['risk_ratio'] = comp_sev['high_sev_sum'] / comp_sev['total']

        risky_comps = comp_sev[comp_sev['risk_ratio'] > 0.30].index.tolist()
        if risky_comps:
            return f"**Risk Alert:** The following components have >30% Critical/Blocker bugs: {', '.join(risky_comps)}"
    return None


def get_bottleneck_insight(df):
    """
    Insight 2: Throughput Bottlenecks

    Start from the list of bug and remove all the "Open" bug from the list as to ignore the unfixed bug for calculation
    Groups the remaining row by Severity and calculates the average of the Day to Close column
    Identify the highest number in that list
    """
    avg_time_sev = df[df['status'] == 'Closed'].groupby('severity')['days_to_close'].mean()
    if not avg_time_sev.empty:
        slowest_sev = avg_time_sev.idxmax()
        slowest_days = avg_time_sev.max()
        return f"**Bottleneck:** '{slowest_sev}' bugs take the longest to close (Avg: {slowest_days:.1f} days)."
    return None


def get_volume_insight(df):
    """
    Insight 3: Volume Leader

    create a list of bugs and which product they belong to
    count how many times each name appears
    look for the highest number in that list
    """
    if df.empty:
        return None
    top_product = df['product'].value_counts().idxmax()
    top_count = df['product'].value_counts().max()
    return f"**Volume:** Product '{top_product}' has the highest volume with {top_count} reported bugs."


def get_trend_insight(df):
    """
    Insight 4: Trend Analysis

    Iterates through all consecutive releases to identify the
    single largest reduction (improvement) in bug volume.
    """
    # 1. Get counts sorted by release version (R0.9 -> R1.0 -> R1.1...)
    release_counts = df['found_in_release'].value_counts().sort_index()

    if len(release_counts) < 2:
        return None

    max_reduction = 0
    best_pair = None

    # 2. Loop through the releases to find the biggest drop
    # Start from the second item (index 1) and compare it to the previous (i-1)
    for i in range(1, len(release_counts)):
        current_rel = release_counts.index[i]
        prev_rel = release_counts.index[i - 1]

        current_count = release_counts.iloc[i]
        prev_count = release_counts.iloc[i - 1]

        diff = current_count - prev_count
        # if bugs DECREASED (diff < 0)
        if diff < 0:
            reduction = abs(diff)
            # Check if this is the biggest reduction so far
            if reduction > max_reduction:
                max_reduction = reduction
                best_pair = (prev_rel, current_rel)

    if best_pair:
        prev, curr = best_pair
        return f"**Best Trend:** Largest bug reduction occurred from {prev} to {curr} (decreased by {max_reduction} bugs)."

    return "**Trend:** No bug reduction observed between any consecutive releases (volume consistently increased)."


def generate_insights(df):
    """Aggregator function that calls specific insight functions."""
    insights = []

    # List of functions to execute
    insight_generators = [
        get_risk_insight,
        get_bottleneck_insight,
        get_volume_insight,
        get_trend_insight
    ]

    for generator in insight_generators:
        result = generator(df)
        if result:
            insights.append(result)

    return insights


# --- Visualization ---

def plot_dashboard(df):
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # Plot 1: Open vs Closed (Volume) -> Top Left
    status_counts = df['status'].value_counts()
    sns.barplot(x=status_counts.index, y=status_counts.values, hue=status_counts.index, legend=False, ax=axes[0, 0],
                palette="viridis")
    axes[0, 0].set_title("Total Bugs by Status")
    axes[0, 0].set_ylabel("Count")

    # Plot 2: Bugs by Component & Severity -> Top Right
    high_sev = df[df['severity'].isin(['Critical', 'Blocker'])]
    if not high_sev.empty:
        sns.countplot(data=high_sev, x='component', hue='severity', ax=axes[0, 1], palette="magma")
        axes[0, 1].set_title("Critical & Blocker Bugs by Component")
        axes[0, 1].tick_params(axis='x', rotation=45)
    else:
        axes[0, 1].text(0.5, 0.5, "No Critical/Blocker Data", ha='center')

    # Plot 3: Avg Days to Close by Product -> Bottom Left
    closed_df = df[df['status'] == 'Closed']
    avg_days = closed_df.groupby('product')['days_to_close'].mean().reset_index()
    sns.barplot(data=avg_days, x='product', y='days_to_close', hue='product', legend=False, ax=axes[1, 0],
                palette="coolwarm")
    axes[1, 0].set_title("Avg Days to Close by Product")
    axes[1, 0].set_ylabel("Days")

    # Plot 4: Trend Analysis -> Bottom Right
    release_counts = df['found_in_release'].value_counts().sort_index()
    sns.lineplot(x=release_counts.index, y=release_counts.values, marker='o', ax=axes[1, 1], color='navy',
                 linewidth=2.5)
    axes[1, 1].set_title("Bug Trend by Release")
    axes[1, 1].set_ylabel("Total Bugs Found")
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


# --- Execution ---
if __name__ == "__main__":
    # Ensure 'bugs.csv' exists in your working directory
    try:
        df = load_and_clean_data('bugs.csv')

        total_open = df[df['status'] == 'Open'].shape[0]
        avg_close_time = df[df['status'] == 'Closed']['days_to_close'].mean()

        print(f"--- DASHBOARD METRICS ---")
        print(f"Total Open Bugs: {total_open}")
        print(f"Avg Time to Close: {avg_close_time:.1f} days")
        print("-" * 30)

        print("--- AUTOMATED INSIGHTS ---")
        insights = generate_insights(df)
        for i in insights:
            print(i)
        print("-" * 30)

        plot_dashboard(df)

    except FileNotFoundError:
        print("Error: 'bugs.csv' file not found. Please ensure the file is in the same directory.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
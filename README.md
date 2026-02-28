# Quality ML Developer Intern - Dashboard & Analysis

## Project Overview

This project is a solution to the "Quality ML Developer Intern" technical test (which I fail QAQ). The goal was to build a data dashboard using a provided bug tracking dataset (`bugs.csv`) to answer key questions regarding bug volume, severity, throughput, and component risk.

In addition to standard visualizations, this solution includes an **Automated Insight Extraction** engine and a robust **Data Cleaning Pipeline** to handle significant data quality issues found in the raw file.

## How to Run

### Prerequisites

  * Python 3.x
  * Libraries: `pandas`, `matplotlib`, `seaborn`, `numpy`

### Installation

```bash
pip install pandas matplotlib seaborn numpy
```

### Execution

Ensure `bugs.csv` is in the same directory as the script, then run:

```bash
python dashboard.py
```

*Note: The script will output key metrics and text insights to the console and launch a matplotlib window with the 4-panel dashboard.*

-----

## Dashboard Features (Part 1)

The solution visualizes the data via a 2x2 grid answering the core requirements:

1.  **Volume by Status:** A bar chart showing the breakdown of Open vs. Closed bugs.
2.  **Quality by Area:** A count plot highlighting components with **Critical** and **Blocker** bugs.
3.  **Throughput:** A bar chart showing the **Average Days to Close** by Product.
4.  **Trend Analysis:** A line chart visualizing bug volume changes across releases (`R0.9` $\to$ `R1.2`).

-----

## Automated Insights (Part 2)

To assist non-technical stakeholders, the script includes a rule-based engine that generates human-readable summaries:

  * **Risk Alert:** Flags components where the ratio of `Critical` + `Blocker` bugs exceeds **30%**.
  * **Throughput Bottleneck:** Identifies which severity level takes the longest on average to close.
  * **Volume Leader:** Identifies the Product with the highest total bug count.
  * **Best Trend:** (Stretch Goal) Iterates through consecutive releases to identify the specific transition (e.g., R1.0 $\to$ R1.1) that saw the largest reduction in bug volume.

-----

## Data Quality & Cleaning Report (Part 3)

A rigorous analysis of `bugs.csv` revealed three categories of data quality issues which were addressed in the `load_and_clean_data` function:

### 1\. Categorical Inconsistencies (Typos)

The raw data contained numerous typos in the `Severity` and `Status` columns.

  * **Issues:** `Bloker`, `Blcokr`, `Critcal`, `Majro`, `Minro`, `Closd`.
  * **Fix:** Implemented a dictionary mapping strategy to normalize these into standard categories (`Blocker`, `Critical`, etc.) before aggregation.

### 2\. Date Integrity

  * **Issues:** Mixed date formats (US vs. International) and garbage entries (e.g., `2025/13/40` in Row 47).
  * **Fix:** Used `pd.to_datetime` with `dayfirst=True` and `errors='coerce'` to standardize parsing and convert invalid dates to `NaT` (Not a Time).

### 3\. Logical "Time Travel" Anomalies

  * **Issue:** Approximately **30%** of the dataset contained bugs where the **Created Date was later than the Closed Date** (e.g., Created 2026, Closed 2020).
  * **Impact:** This results in negative duration values (e.g., -1,900 days).
  * **Fix:**
      * **For Volume:** These rows were retained to keep counts accurate.
      * **For Throughput:** Negative values were replaced with `NaN`. The reported "Average Time to Close" is calculated strictly from valid, positive durations.

-----

## Future ML Opportunities

If time permitted, I would implement a **Regression Model** (e.g., Random Forest) to predict **Time-to-Close** for new bugs.

  * **Features:** `Severity`, `Component`, `Reporter Team`, and `Found in Release`.

  * **Goal:** Provide accurate resolution estimates for Project Managers during the triage phase.

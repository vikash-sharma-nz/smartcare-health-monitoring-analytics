"""
==============================================================================
SmartCare Health Monitoring — Data Analysis Script
MBI806B: Business Data Analytics with Visualisation and Decision-Making
Case Study Report — Task 1

DESCRIPTION:
    This script performs the full data analysis pipeline for the SmartCare
    Health Monitoring dataset including:
      - Data loading and preprocessing
      - Exploratory data analysis (EDA)
      - Statistical correlation analysis
      - Risk classification
      - Generation of all 6 report figures

HOW TO RUN:
    1. Place this file in the same folder as smartwatch_health_data.csv
    2. Install dependencies:
       pip install pandas numpy matplotlib seaborn scipy
    3. Run:
       python smartcare_analysis.py
    4. Six PNG figures will be saved to the current directory.

LIBRARIES:
    - pandas     : data loading, preprocessing, groupby aggregation
    - numpy      : numerical operations, correlation, trend lines
    - matplotlib : static chart generation (all 6 report figures)
    - seaborn    : enhanced statistical heatmap visualisation
    - scipy      : Pearson r and p-value significance testing

OUTPUT FILES:
    fig1_correlation_heatmap.png
    fig2_health_score_bar.png
    fig3_steps_stress_line.png
    fig4_sleep_stress_scatter.png
    fig5_pie_charts.png
    fig6_dashboard.png
==============================================================================
"""

# ── Imports ───────────────────────────────────────────────────────────────────
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ── Chart Styling ─────────────────────────────────────────────────────────────
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['figure.dpi'] = 180

BLUE   = '#2E86AB'
RED    = '#E84855'
GREEN  = '#059669'
YELLOW = '#D97706'
PURPLE = '#7C3AED'
LIGHT  = '#F8FAFC'
CARD   = '#FFFFFF'
BORDER = '#E2E8F0'


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — DATA LOADING & PREPROCESSING
# ══════════════════════════════════════════════════════════════════════════════

def load_and_preprocess(filepath='smartwatch_health_data.csv'):
    """
    Load the SmartCare CSV dataset and apply preprocessing steps:
      - Age group binning (50-59, 60-69, 70-79)
      - Composite risk classification (High Risk / Low Risk)
      - Data quality checks (missing values, outliers)
    Returns a cleaned, enriched pandas DataFrame.
    """
    print("=" * 60)
    print("  SmartCare Health Analytics — Analysis Script")
    print("  MBI806B | Yoobee Colleges NZ")
    print("=" * 60)

    df = pd.read_csv(filepath)
    print(f"\n[1] Data loaded: {df.shape[0]} rows x {df.shape[1]} columns")

    # Missing value check
    missing = df.isnull().sum()
    if missing.sum() == 0:
        print("[2] Missing values: None detected — dataset complete")
    else:
        print(f"[2] Missing values detected:\n{missing[missing > 0]}")

    # Age group binning
    df['AgeGroup'] = pd.cut(
        df['Age'], bins=[49, 59, 69, 79],
        labels=['50-59', '60-69', '70-79']
    ).astype(str)
    print(f"[3] Age groups: {df['AgeGroup'].value_counts().to_dict()}")

    # Risk classification
    df['RiskLevel'] = np.where(
        (df['StressIndex'] >= 70) |
        (df['SleepQualityScore'] <= 55) |
        (df['HeartRate_Avg'] >= 90),
        'High Risk', 'Low Risk'
    )
    df['HighRisk'] = (df['RiskLevel'] == 'High Risk').astype(int)
    high_n = df['HighRisk'].sum()
    print(f"[4] Risk: {high_n} High Risk ({high_n/len(df)*100:.0f}%), "
          f"{len(df)-high_n} Low Risk")

    # Outlier check (IQR method)
    numeric_cols = ['DailyStepCount', 'HeartRate_Avg', 'StressIndex',
                    'SleepQualityScore', 'WeeklyHealthScore', 'CalorieBurn']
    outliers = {}
    for col in numeric_cols:
        Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        IQR = Q3 - Q1
        n_out = ((df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)).sum()
        if n_out > 0:
            outliers[col] = n_out
    if outliers:
        print(f"[5] Statistical outliers (natural variation retained): {outliers}")
    else:
        print("[5] No extreme outliers detected")

    return df


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — EXPLORATORY DATA ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

def run_eda(df):
    """
    Perform comprehensive EDA: descriptive stats, demographic breakdown,
    and Pearson correlation analysis with significance testing.
    """
    print("\n" + "-" * 60)
    print("  EXPLORATORY DATA ANALYSIS")
    print("-" * 60)

    # Descriptive statistics
    print("\n[DESCRIPTIVE STATISTICS]")
    cols = ['Age', 'DailyStepCount', 'HeartRate_Avg', 'SleepDuration_Hours',
            'SleepQualityScore', 'StressIndex', 'CalorieBurn', 'WeeklyHealthScore']
    print(df[cols].describe().round(2).to_string())

    # Gender breakdown
    print("\n[GENDER BREAKDOWN]")
    for gender in df['Gender'].unique():
        s = df[df['Gender'] == gender]
        print(f"  {gender}: n={len(s)}, "
              f"Avg Health={s['WeeklyHealthScore'].mean():.1f}, "
              f"Avg Stress={s['StressIndex'].mean():.1f}, "
              f"Avg Steps={s['DailyStepCount'].mean():,.0f}")

    # Age group breakdown
    print("\n[AGE GROUP BREAKDOWN]")
    for ag in ['50-59', '60-69', '70-79']:
        s = df[df['AgeGroup'] == ag]
        print(f"  {ag}: n={len(s)}, "
              f"Avg Health={s['WeeklyHealthScore'].mean():.1f}, "
              f"Avg Steps={s['DailyStepCount'].mean():,.0f}, "
              f"High Risk={s['HighRisk'].sum()}/{len(s)}")

    # Pearson correlations with Weekly Health Score
    print("\n[PEARSON CORRELATIONS vs WeeklyHealthScore]")
    predictors = ['Age', 'DailyStepCount', 'HeartRate_Avg', 'SleepDuration_Hours',
                  'SleepQualityScore', 'StressIndex', 'CalorieBurn']
    for col in predictors:
        r, p = stats.pearsonr(df[col], df['WeeklyHealthScore'])
        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
        print(f"  {col:<28} r = {r:+.3f}   p = {p:.4f}  {sig}")

    r_ss, p_ss = stats.pearsonr(df['SleepQualityScore'], df['StressIndex'])
    print(f"\n  SleepQuality vs StressIndex: r = {r_ss:+.3f}, p = {p_ss:.4f}")

    # Fall alerts
    fall_n = df['FallAlerts'].sum()
    print(f"\n[FALL ALERTS] {fall_n} patients ({fall_n/len(df)*100:.0f}%)")

    print("\n" + "-" * 60)
    print("  EDA complete.")
    print("-" * 60 + "\n")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — VISUALIZATIONS
# ══════════════════════════════════════════════════════════════════════════════

def create_visualizations(df, output_dir='.'):
    """
    Generate all 6 report figures as high-resolution PNG files.
    """

    def save(name):
        path = f"{output_dir}/{name}"
        plt.savefig(path, bbox_inches='tight', facecolor=CARD, dpi=180)
        plt.close()
        print(f"  Saved: {name}")

    print("[GENERATING FIGURES]")

    # Figure 1: Pearson Correlation Heatmap
    fig, ax = plt.subplots(figsize=(11, 9), facecolor=CARD)
    ax.set_facecolor(CARD)
    cols = ['Age', 'DailyStepCount', 'HeartRate_Min', 'HeartRate_Max', 'HeartRate_Avg',
            'SleepDuration_Hours', 'SleepQualityScore', 'StressIndex',
            'CalorieBurn', 'WeeklyHealthScore']
    labels = ['Age', 'Steps', 'HR Min', 'HR Max', 'HR Avg',
              'Sleep Hrs', 'Sleep Qual', 'Stress', 'Calories', 'Health']
    corr = df[cols].corr()
    corr.index = labels
    corr.columns = labels
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                linewidths=0.8, linecolor=BORDER, ax=ax,
                annot_kws={'size': 10, 'weight': 'bold'},
                cbar_kws={'shrink': 0.8})
    ax.set_title('Figure 1: Pearson Correlation Heatmap — SmartCare Health Metrics',
                 fontsize=14, fontweight='bold', pad=18, color='#1E293B')
    ax.tick_params(colors='#475569', labelsize=10)
    plt.tight_layout()
    save('fig1_correlation_heatmap.png')

    # Figure 2: Health Score by Age Group & Gender
    fig, ax = plt.subplots(figsize=(10, 6), facecolor=CARD)
    ax.set_facecolor(LIGHT)
    grouped = df.groupby(['AgeGroup', 'Gender'], observed=True)['WeeklyHealthScore'] \
                .mean().unstack()
    x = np.arange(len(grouped))
    w = 0.35
    bars1 = ax.bar(x - w/2, grouped.get('Female', [0]*3), w,
                   color=BLUE, label='Female', alpha=0.9, zorder=3)
    bars2 = ax.bar(x + w/2, grouped.get('Male', [0]*3), w,
                   color=RED, label='Male', alpha=0.9, zorder=3)
    for bar in list(bars1) + list(bars2):
        col = BLUE if bar in bars1 else RED
        ax.annotate(f'{bar.get_height():.1f}',
                    (bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5),
                    ha='center', va='bottom', fontsize=10, fontweight='bold', color=col)
    ax.set_xticks(x)
    ax.set_xticklabels(['50-59', '60-69', '70-79'], fontsize=11)
    ax.set_xlabel('Age Group', fontsize=12, labelpad=8)
    ax.set_ylabel('Avg Weekly Health Score', fontsize=12)
    ax.set_ylim(0, 110)
    ax.set_title('Figure 2: Average Weekly Health Score by Age Group & Gender',
                 fontsize=14, fontweight='bold', pad=15, color='#1E293B')
    ax.legend(fontsize=11, framealpha=0.9)
    ax.grid(axis='y', color=BORDER, linewidth=0.8, zorder=0)
    ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    save('fig2_health_score_bar.png')

    # Figure 3: Dual-axis Line — Steps & Stress by Age Group
    fig, ax = plt.subplots(figsize=(10, 6), facecolor=CARD)
    ax.set_facecolor(LIGHT)
    steps_age = df.groupby('AgeGroup', observed=True)[['DailyStepCount', 'StressIndex']].mean()
    ax2 = ax.twinx()
    ax.fill_between(range(len(steps_age)), steps_age['DailyStepCount'], alpha=0.15, color=BLUE)
    ax.plot(range(len(steps_age)), steps_age['DailyStepCount'],
            marker='o', color=BLUE, linewidth=2.5, markersize=10, label='Daily Steps', zorder=5)
    ax2.plot(range(len(steps_age)), steps_age['StressIndex'],
             marker='s', color=RED, linewidth=2.5, markersize=10,
             linestyle='--', label='Stress Index', zorder=5)
    for i, (xv, yv) in enumerate(zip(range(len(steps_age)), steps_age['DailyStepCount'])):
        ax.annotate(f'{yv:,.0f}', (xv, yv + 120),
                    ha='center', fontsize=10, fontweight='bold', color=BLUE)
    for i, (xv, yv) in enumerate(zip(range(len(steps_age)), steps_age['StressIndex'])):
        ax2.annotate(f'{yv:.1f}', (xv, yv + 0.8),
                     ha='center', fontsize=10, fontweight='bold', color=RED)
    ax.set_xticks(range(len(steps_age)))
    ax.set_xticklabels(['50-59', '60-69', '70-79'], fontsize=11)
    ax.set_xlabel('Age Group', fontsize=12, labelpad=8)
    ax.set_ylabel('Avg Daily Step Count', color=BLUE, fontsize=12)
    ax2.set_ylabel('Avg Stress Index', color=RED, fontsize=12)
    ax.tick_params(axis='y', colors=BLUE)
    ax2.tick_params(axis='y', colors=RED)
    lines1, labs1 = ax.get_legend_handles_labels()
    lines2, labs2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labs1 + labs2, loc='upper right', fontsize=11, framealpha=0.9)
    ax.set_title('Figure 3: Avg Daily Step Count & Stress Index Across Age Groups',
                 fontsize=14, fontweight='bold', pad=15, color='#1E293B')
    ax.spines[['top']].set_visible(False)
    ax2.spines[['top']].set_visible(False)
    ax.grid(axis='y', color=BORDER, linewidth=0.8)
    plt.tight_layout()
    save('fig3_steps_stress_line.png')

    # Figure 4: Sleep Quality vs Stress Index scatter
    fig, ax = plt.subplots(figsize=(10, 6), facecolor=CARD)
    ax.set_facecolor(LIGHT)
    for gender, colour, marker in [('Female', BLUE, 'o'), ('Male', RED, 's')]:
        sub = df[df['Gender'] == gender]
        ax.scatter(sub['SleepQualityScore'], sub['StressIndex'],
                   c=colour, label=gender, alpha=0.8, s=90, marker=marker,
                   edgecolors='white', linewidths=0.8, zorder=5)
    m, b = np.polyfit(df['SleepQualityScore'], df['StressIndex'], 1)
    r, p = stats.pearsonr(df['SleepQualityScore'], df['StressIndex'])
    xl = np.linspace(df['SleepQualityScore'].min(), df['SleepQualityScore'].max(), 100)
    ax.plot(xl, m * xl + b, color='#94A3B8', linestyle='--',
            linewidth=2, label=f'Trend (r = {r:.3f}, p = {p:.3f})')
    ax.set_xlabel('Sleep Quality Score', fontsize=12, labelpad=8)
    ax.set_ylabel('Stress Index', fontsize=12)
    ax.set_title('Figure 4: Sleep Quality Score vs. Stress Index by Gender',
                 fontsize=14, fontweight='bold', pad=15, color='#1E293B')
    ax.legend(fontsize=11, framealpha=0.9)
    ax.grid(color=BORDER, linewidth=0.8, zorder=0)
    ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    save('fig4_sleep_stress_scatter.png')

    # Figure 5: Pie charts — Fall Alerts & Gender
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), facecolor=CARD)
    fig.patch.set_facecolor(CARD)
    fall = df['FallAlerts'].value_counts().sort_index()
    axes[0].pie(fall, labels=['No Alert', 'Fall Alert'],
                autopct='%1.1f%%', colors=[GREEN, YELLOW], startangle=90,
                wedgeprops={'edgecolor': 'white', 'linewidth': 2.5},
                textprops={'fontsize': 12, 'fontweight': 'bold'})
    axes[0].set_title('Figure 5a: Fall Alert Distribution',
                      fontsize=13, fontweight='bold', color='#1E293B', pad=12)
    gender_counts = df['Gender'].value_counts()
    axes[1].pie(gender_counts, labels=gender_counts.index,
                autopct='%1.1f%%', colors=[BLUE, RED], startangle=90,
                wedgeprops={'edgecolor': 'white', 'linewidth': 2.5},
                textprops={'fontsize': 12, 'fontweight': 'bold'})
    axes[1].set_title('Figure 5b: Patient Gender Distribution',
                      fontsize=13, fontweight='bold', color='#1E293B', pad=12)
    plt.tight_layout(pad=3)
    save('fig5_pie_charts.png')

    # Figure 6: Composite Analytics Dashboard
    fig = plt.figure(figsize=(18, 11), facecolor=LIGHT)
    fig.suptitle('SmartCare Health Monitoring — Analytics Dashboard',
                 fontsize=17, fontweight='bold', y=0.98, color='#1E293B')

    def style_ax(ax):
        ax.set_facecolor(CARD)
        for spine in ax.spines.values():
            spine.set_edgecolor(BORDER)
        ax.grid(color=BORDER, linewidth=0.7, zorder=0)
        ax.tick_params(colors='#64748B', labelsize=9)

    # Panel 1: Heart Rate Boxplot
    ax1 = fig.add_subplot(2, 3, 1)
    df_melt = df.melt(id_vars=['AgeGroup'],
                      value_vars=['HeartRate_Min', 'HeartRate_Avg', 'HeartRate_Max'],
                      var_name='Metric', value_name='BPM')
    sns.boxplot(data=df_melt, x='AgeGroup', y='BPM', hue='Metric',
                palette={'HeartRate_Min': GREEN, 'HeartRate_Avg': YELLOW, 'HeartRate_Max': RED},
                ax=ax1, width=0.6, linewidth=1.2)
    ax1.set_title('Heart Rate by Age Group', fontsize=11, fontweight='bold', color='#1E293B')
    ax1.set_xlabel('Age Group', fontsize=9)
    ax1.set_ylabel('BPM', fontsize=9)
    ax1.legend(fontsize=7, title='', labels=['Min', 'Avg', 'Max'])
    style_ax(ax1)

    # Panel 2: Sleep Duration by Age & Gender
    ax2 = fig.add_subplot(2, 3, 2)
    sleep_grp = df.groupby(['AgeGroup', 'Gender'], observed=True)['SleepDuration_Hours'] \
                  .mean().unstack()
    x2 = np.arange(len(sleep_grp))
    ax2.bar(x2 - 0.175, sleep_grp.get('Female', [0]*3), 0.35, color=BLUE, alpha=0.9, label='Female')
    ax2.bar(x2 + 0.175, sleep_grp.get('Male', [0]*3),   0.35, color=RED,  alpha=0.9, label='Male')
    ax2.set_xticks(x2)
    ax2.set_xticklabels(['50-59', '60-69', '70-79'], fontsize=9)
    ax2.set_title('Avg Sleep Duration (hrs)', fontsize=11, fontweight='bold', color='#1E293B')
    ax2.set_ylabel('Hours', fontsize=9)
    ax2.legend(fontsize=8)
    style_ax(ax2)

    # Panel 3: Steps vs Health Score (coloured by Stress)
    ax3 = fig.add_subplot(2, 3, 3)
    sc = ax3.scatter(df['DailyStepCount'], df['WeeklyHealthScore'],
                     c=df['StressIndex'], cmap='RdYlGn_r', s=55,
                     alpha=0.85, edgecolors='white', linewidths=0.5, zorder=5)
    plt.colorbar(sc, ax=ax3, label='Stress', shrink=0.8)
    m3, b3 = np.polyfit(df['DailyStepCount'], df['WeeklyHealthScore'], 1)
    xl3 = np.linspace(df['DailyStepCount'].min(), df['DailyStepCount'].max(), 100)
    ax3.plot(xl3, m3 * xl3 + b3, '--', color='#94A3B8', linewidth=1.5)
    ax3.set_title('Steps vs Health Score', fontsize=11, fontweight='bold', color='#1E293B')
    ax3.set_xlabel('Daily Steps', fontsize=9)
    ax3.set_ylabel('Health Score', fontsize=9)
    style_ax(ax3)

    # Panel 4: Risk Distribution by Age Group
    ax4 = fig.add_subplot(2, 3, 4)
    risk_age = df.groupby('AgeGroup', observed=True)['HighRisk'] \
                 .value_counts().unstack(fill_value=0)
    risk_age.columns = ['Low Risk', 'High Risk']
    risk_age.plot(kind='bar', stacked=True, ax=ax4,
                  color=[GREEN, RED], edgecolor='white', width=0.6)
    ax4.set_title('Risk Distribution by Age Group', fontsize=11, fontweight='bold', color='#1E293B')
    ax4.set_xlabel('Age Group', fontsize=9)
    ax4.set_ylabel('Patients', fontsize=9)
    ax4.set_xticklabels(['50-59', '60-69', '70-79'], rotation=0, fontsize=9)
    ax4.legend(fontsize=8)
    style_ax(ax4)

    # Panel 5: Calorie Burn Line Chart
    ax5 = fig.add_subplot(2, 3, 5)
    cal = df.groupby('AgeGroup', observed=True)['CalorieBurn'].mean()
    ax5.fill_between(range(len(cal)), cal.values, alpha=0.15, color=BLUE)
    ax5.plot(range(len(cal)), cal.values, marker='o', color=BLUE, linewidth=2.5, markersize=10, zorder=5)
    for i, (xv, yv) in enumerate(zip(range(len(cal)), cal.values)):
        ax5.annotate(f'{yv:,.0f}', (xv, yv + 40), ha='center', fontsize=10, fontweight='bold', color=BLUE)
    ax5.set_xticks(range(len(cal)))
    ax5.set_xticklabels(['50-59', '60-69', '70-79'], fontsize=9)
    ax5.set_title('Avg Calorie Burn by Age Group', fontsize=11, fontweight='bold', color='#1E293B')
    ax5.set_ylabel('Calories', fontsize=9)
    style_ax(ax5)

    # Panel 6: KPI Summary
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.axis('off')
    ax6.add_patch(mpatches.FancyBboxPatch(
        (0.02, 0.02), 0.96, 0.96,
        boxstyle='round,pad=0.02', facecolor='#EFF6FF', edgecolor='#BFDBFE',
        linewidth=1.5, transform=ax6.transAxes, zorder=-1
    ))
    kpis = [
        ('Total Patients',     str(len(df)),                               BLUE),
        ('Avg Age',            f'{df["Age"].mean():.1f} yrs',              '#475569'),
        ('Avg Health Score',   f'{df["WeeklyHealthScore"].mean():.1f}/100', GREEN),
        ('Avg Daily Steps',    f'{df["DailyStepCount"].mean():,.0f}',      BLUE),
        ('Avg Stress Index',   f'{df["StressIndex"].mean():.1f}/100',      YELLOW),
        ('Fall Alert Rate',    f'{df["FallAlerts"].mean()*100:.1f}%',      '#D97706'),
        ('High-Risk Patients', f'{df["HighRisk"].sum()} ({df["HighRisk"].mean()*100:.0f}%)', RED),
        ('Avg Sleep Duration', f'{df["SleepDuration_Hours"].mean():.1f} hrs', PURPLE),
    ]
    for i, (lbl, val, col) in enumerate(kpis):
        y = 0.90 - i * 0.115
        ax6.text(0.06, y, lbl + ':', transform=ax6.transAxes,
                 fontsize=10, fontweight='bold', color='#374151')
        ax6.text(0.65, y, val, transform=ax6.transAxes,
                 fontsize=10, fontweight='bold', color=col)
    ax6.set_title('Key Performance Indicators', fontsize=11, fontweight='bold', color='#1E293B')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    save('fig6_dashboard.png')

    print("\n  All 6 figures generated successfully.")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    df = load_and_preprocess('smartwatch_health_data.csv')
    run_eda(df)
    create_visualizations(df, output_dir='.')
    print("\n" + "=" * 60)
    print("  Analysis complete. All figures saved.")
    print("=" * 60)

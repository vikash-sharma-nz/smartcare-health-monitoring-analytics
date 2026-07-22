"""
==============================================================================
SmartCare Health Analytics — Interactive Dashboard
MBI806B: Business Data Analytics with Visualisation and Decision-Making

HOW TO RUN:
    1. pip install dash plotly pandas numpy
    2. Place this file in the same folder as smartwatch_health_data.csv
    3. python smartcare_dashboard.py
    4. Open browser → http://127.0.0.1:8050
==============================================================================
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go

import dash
from dash import dcc, html, dash_table, Input, Output

# ── Colours ───────────────────────────────────────────────────────────────────
C_BLUE   = '#2E86AB'
C_RED    = '#E84855'
C_GREEN  = '#059669'
C_YELLOW = '#D97706'
C_BG     = '#F1F5F9'
C_CARD   = '#FFFFFF'
C_BORDER = '#E2E8F0'
C_TEXT   = '#1E293B'
C_MUTED  = '#64748B'

# ── Load & Preprocess ─────────────────────────────────────────────────────────
def load_data():
    df = pd.read_csv('smartwatch_health_data.csv')
    df['AgeGroup'] = pd.cut(df['Age'], bins=[49, 59, 69, 79],
                             labels=['50-59', '60-69', '70-79']).astype(str)
    df['RiskLevel'] = np.where(
        (df['StressIndex'] >= 70) | (df['SleepQualityScore'] <= 55) | (df['HeartRate_Avg'] >= 90),
        'High Risk', 'Low Risk'
    )
    df['HighRisk'] = (df['RiskLevel'] == 'High Risk').astype(int)
    return df

try:
    df_full = load_data()
    print(f"Data loaded successfully: {len(df_full)} patients")
except FileNotFoundError:
    print("ERROR: smartwatch_health_data.csv not found.")
    print("Place the CSV in the same folder as this script and try again.")
    raise

# ── Layout Helper ─────────────────────────────────────────────────────────────
def base_layout(title='', height=320):
    return dict(
        title=dict(text=f'<b>{title}</b>',
                   font=dict(size=13, color=C_TEXT), x=0.01),
        paper_bgcolor=C_CARD,
        plot_bgcolor=C_CARD,
        font=dict(family='Arial', color=C_MUTED, size=11),
        height=height,
        margin=dict(t=48, b=55, l=55, r=25),
        legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(color=C_TEXT, size=11)),
        xaxis=dict(gridcolor=C_BORDER, linecolor=C_BORDER,
                   zerolinecolor=C_BORDER, tickfont=dict(color=C_MUTED)),
        yaxis=dict(gridcolor=C_BORDER, linecolor=C_BORDER,
                   zerolinecolor=C_BORDER, tickfont=dict(color=C_MUTED)),
    )

def empty_chart(msg='No data for current filters'):
    fig = go.Figure()
    fig.update_layout(**base_layout(msg))
    fig.add_annotation(text=msg, xref='paper', yref='paper',
                       x=0.5, y=0.5, showarrow=False,
                       font=dict(color=C_MUTED, size=13))
    return fig

# ── Chart Functions ───────────────────────────────────────────────────────────

def chart_bar(df):
    agg = df.groupby(['AgeGroup', 'Gender'], observed=True)['WeeklyHealthScore'].mean().reset_index()
    fig = go.Figure()
    colour_map = {'Female': C_BLUE, 'Male': C_RED}
    for gender in ['Female', 'Male']:
        sub = agg[agg['Gender'] == gender]
        if sub.empty:
            continue
        fig.add_trace(go.Bar(
            x=sub['AgeGroup'], y=sub['WeeklyHealthScore'].round(1),
            name=gender,
            marker_color=colour_map.get(gender, C_YELLOW),
            text=sub['WeeklyHealthScore'].round(1),
            textposition='outside',
            textfont=dict(color=C_TEXT, size=11),
        ))
    fig.update_layout(**base_layout('Avg Weekly Health Score by Age Group & Gender'),
                      barmode='group', yaxis_range=[0, 115])
    return fig


def chart_heatmap(df):
    cols = ['Age', 'DailyStepCount', 'HeartRate_Min', 'HeartRate_Max', 'HeartRate_Avg',
            'SleepDuration_Hours', 'SleepQualityScore', 'StressIndex', 'CalorieBurn', 'WeeklyHealthScore']
    labels = ['Age', 'Steps', 'HR Min', 'HR Max', 'HR Avg',
              'Sleep Hrs', 'Sleep Qual', 'Stress', 'Calories', 'Health']
    corr = df[cols].corr().round(2).values
    fig = go.Figure(go.Heatmap(
        z=corr, x=labels, y=labels,
        colorscale=[[0, C_RED], [0.5, '#E2E8F0'], [1, C_BLUE]],
        zmid=0, zmin=-1, zmax=1,
        text=[[f'{v:.2f}' for v in row] for row in corr],
        texttemplate='%{text}',
        textfont=dict(size=8, color=C_TEXT),
        showscale=True,
        colorbar=dict(thickness=10, tickfont=dict(color=C_MUTED, size=9))
    ))
    layout = base_layout('Pearson Correlation Heatmap', height=370)
    layout['margin'] = dict(t=48, b=95, l=95, r=25)
    layout['xaxis'] = dict(tickangle=-40, tickfont=dict(color=C_MUTED, size=10),
                           gridcolor=C_BORDER, linecolor=C_BORDER)
    layout['yaxis'] = dict(tickfont=dict(color=C_MUTED, size=10),
                           gridcolor=C_BORDER, linecolor=C_BORDER)
    fig.update_layout(**layout)
    return fig


def chart_scatter_steps(df):
    colour_map = {'Female': C_BLUE, 'Male': C_RED}
    fig = go.Figure()
    for gender in ['Female', 'Male']:
        sub = df[df['Gender'] == gender]
        if sub.empty:
            continue
        fig.add_trace(go.Scatter(
            x=sub['DailyStepCount'],
            y=sub['WeeklyHealthScore'],
            mode='markers',
            name=gender,
            marker=dict(
                color=sub['StressIndex'].tolist(),
                colorscale=[[0, C_GREEN], [0.5, C_YELLOW], [1, C_RED]],
                size=10,
                line=dict(width=1, color='rgba(255,255,255,0.2)'),
                showscale=(gender == 'Female'),
                colorbar=dict(title='Stress', thickness=10,
                              tickfont=dict(color=C_MUTED, size=9)) if gender == 'Female' else None,
            ),
            customdata=list(zip(sub['PatientID'], sub['StressIndex'], sub['HeartRate_Avg'])),
            hovertemplate='ID: %{customdata[0]}<br>Steps: %{x:,}<br>'
                          'Health: %{y}<br>Stress: %{customdata[1]}<br>'
                          'HR Avg: %{customdata[2]}<extra></extra>',
        ))
    # Trend line
    if len(df) >= 2:
        m, b = np.polyfit(df['DailyStepCount'], df['WeeklyHealthScore'], 1)
        x0, x1 = float(df['DailyStepCount'].min()), float(df['DailyStepCount'].max())
        fig.add_trace(go.Scatter(
            x=[x0, x1], y=[m*x0+b, m*x1+b],
            mode='lines', name='Trend',
            line=dict(color='rgba(100,116,139,0.5)', width=2, dash='dot'),
            hoverinfo='skip'
        ))
    fig.update_layout(**base_layout('Daily Steps vs Weekly Health Score  (colour = Stress Index)'))
    return fig


def chart_donut(df):
    counts = df['RiskLevel'].value_counts()
    labels = counts.index.tolist()
    values = counts.values.tolist()
    colours = [C_RED if l == 'High Risk' else C_GREEN for l in labels]
    fig = go.Figure(go.Pie(
        labels=labels,
        values=values,
        hole=0.60,
        marker=dict(
            colors=colours,
            line=dict(color='#F1F5F9', width=2)
        ),
        textfont=dict(color='#1E293B', size=12),
        textinfo='percent+label',
        hovertemplate='%{label}: %{value} patients (%{percent})<extra></extra>',
    ))
    fig.update_layout(
        title=dict(text='<b>Patient Risk Distribution</b>',
                   font=dict(size=13, color=C_TEXT), x=0.01),
        paper_bgcolor=C_CARD,
        font=dict(family='Arial', color=C_MUTED, size=11),
        height=320,
        margin=dict(t=48, b=55, l=25, r=25),
        annotations=[dict(
            text=f'{len(df)}<br>patients',
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=14, color='#1E293B', family='Arial'),
            align='center',
        )],
        legend=dict(
            orientation='h', yanchor='bottom', y=-0.15,
            xanchor='center', x=0.5,
            font=dict(color='#1E293B')
        )
    )
    return fig


def chart_sleep_stress(df):
    colour_map = {'Female': C_BLUE, 'Male': C_RED}
    fig = go.Figure()
    for gender in ['Female', 'Male']:
        sub = df[df['Gender'] == gender]
        if sub.empty:
            continue
        fig.add_trace(go.Scatter(
            x=sub['SleepQualityScore'], y=sub['StressIndex'],
            mode='markers', name=gender,
            marker=dict(color=colour_map.get(gender, C_YELLOW), size=9,
                        line=dict(width=1, color='rgba(255,255,255,0.15)')),
            customdata=list(zip(sub['PatientID'], sub['Age'])),
            hovertemplate='ID: %{customdata[0]}  Age: %{customdata[1]}<br>'
                          'Sleep Q: %{x}  Stress: %{y}<extra></extra>',
        ))
    if len(df) >= 2:
        m, b = np.polyfit(df['SleepQualityScore'], df['StressIndex'], 1)
        r = np.corrcoef(df['SleepQualityScore'], df['StressIndex'])[0, 1]
        x0, x1 = float(df['SleepQualityScore'].min()), float(df['SleepQualityScore'].max())
        fig.add_trace(go.Scatter(
            x=[x0, x1], y=[m*x0+b, m*x1+b],
            mode='lines', name=f'Trend (r={r:.2f})',
            line=dict(color='rgba(100,116,139,0.5)', width=2, dash='dot'),
            hoverinfo='skip'
        ))
    layout = base_layout('Sleep Quality Score vs Stress Index')
    layout['xaxis']['title'] = 'Sleep Quality Score'
    layout['yaxis']['title'] = 'Stress Index'
    fig.update_layout(**layout)
    return fig


def chart_box(df):
    fig = go.Figure()
    metrics = [
        ('HeartRate_Min', 'HR Min', C_GREEN,  'rgba(5,150,105,0.15)'),
        ('HeartRate_Avg', 'HR Avg', C_YELLOW, 'rgba(217,119,6,0.15)'),
        ('HeartRate_Max', 'HR Max', C_RED,    'rgba(232,72,85,0.15)'),
    ]
    age_groups = [g for g in ['50-59', '60-69', '70-79'] if g in df['AgeGroup'].values]
    for col, label, colour, fill in metrics:
        for i, ag in enumerate(age_groups):
            sub = df[df['AgeGroup'] == ag][col].tolist()
            fig.add_trace(go.Box(
                y=sub, name=label,
                x=[ag] * len(sub),
                marker_color=colour, line_color=colour,
                fillcolor=fill,
                boxmean=True,
                legendgroup=label,
                showlegend=(i == 0),
            ))
    fig.update_layout(**base_layout('Heart Rate Distribution by Age Group'), boxmode='group')
    return fig


def chart_line_calories(df):
    agg = (df.groupby('AgeGroup', observed=True)['CalorieBurn']
             .mean().reset_index()
             .rename(columns={'CalorieBurn': 'Avg'}))
    agg = agg[agg['AgeGroup'].isin(['50-59', '60-69', '70-79'])]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=agg['AgeGroup'].tolist(),
        y=agg['Avg'].round(0).tolist(),
        mode='lines+markers+text',
        line=dict(color=C_BLUE, width=3, shape='spline'),
        marker=dict(color=C_BLUE, size=12, line=dict(color=C_BG, width=2)),
        fill='tozeroy', fillcolor='rgba(46,134,171,0.12)',
        text=[f"{v:,.0f}" for v in agg['Avg']],
        textposition='top center',
        textfont=dict(color=C_BLUE, size=12),
        hovertemplate='%{x}: %{y:,.0f} kcal<extra></extra>',
        name='Avg Calories',
    ))
    layout = base_layout('Avg Calorie Burn by Age Group')
    layout['yaxis']['title'] = 'Avg Calories Burned'
    fig.update_layout(**layout)
    return fig


def chart_pie_fall(df):
    no_fall  = int((df['FallAlerts'] == 0).sum())
    has_fall = int((df['FallAlerts'] == 1).sum())
    fig = go.Figure(go.Pie(
        labels=['No Alert', 'Fall Alert'],
        values=[no_fall, has_fall],
        marker=dict(colors=[C_GREEN, C_YELLOW],
                    line=dict(color='#F1F5F9', width=2)),
        textfont=dict(color='#1E293B', size=12),
        textinfo='percent+label',
        pull=[0, 0.06],
        hovertemplate='%{label}: %{value} patients (%{percent})<extra></extra>',
    ))
    fig.update_layout(
        title=dict(text='<b>Fall Alert Distribution</b>',
                   font=dict(size=13, color=C_TEXT), x=0.01),
        paper_bgcolor=C_CARD,
        font=dict(family='Arial', color=C_MUTED, size=11),
        height=320,
        margin=dict(t=48, b=55, l=25, r=25),
        legend=dict(
            orientation='h', yanchor='bottom', y=-0.15,
            xanchor='center', x=0.5,
            font=dict(color='#1E293B')
        )
    )
    return fig


# ── KPI Card ─────────────────────────────────────────────────────────────────
def kpi_card(title, value, subtitle, colour, icon):
    return html.Div([
        html.Div(icon, style={'fontSize': '22px', 'marginBottom': '8px'}),
        html.Div(value, style={'fontSize': '28px', 'fontWeight': '800',
                               'color': colour, 'lineHeight': '1'}),
        html.Div(title, style={'fontSize': '11px', 'color': C_MUTED,
                               'marginTop': '6px', 'textTransform': 'uppercase',
                               'letterSpacing': '0.5px'}),
        html.Div(subtitle, style={'fontSize': '11px', 'color': C_MUTED, 'marginTop': '3px'}),
    ], style={
        'background': C_CARD, 'borderRadius': '12px', 'padding': '20px 22px',
        'flex': '1', 'minWidth': '180px',
        'borderTop': f'3px solid {colour}',
        'border': f'1px solid {C_BORDER}',
    })


# ── App ───────────────────────────────────────────────────────────────────────
app = dash.Dash(__name__, title='SmartCare Health Analytics')

CARD_STYLE = {'background': C_CARD, 'borderRadius': '12px',
              'border': f'1px solid {C_BORDER}', 'padding': '6px'}

DD = {'backgroundColor': C_CARD, 'color': C_TEXT,
      'border': f'1px solid {C_BORDER}', 'borderRadius': '8px'}

app.layout = html.Div(
    style={'backgroundColor': C_BG, 'minHeight': '100vh',
           'fontFamily': 'Arial, sans-serif', 'color': C_TEXT},
    children=[

    # Header
    html.Div(style={'background': '#1F4E79', 'borderBottom': f'1px solid {C_BORDER}',
                    'padding': '0 32px'}, children=[
        html.Div(style={'display': 'flex', 'alignItems': 'center',
                        'justifyContent': 'space-between', 'height': '64px'}, children=[
            html.Div(style={'display': 'flex', 'alignItems': 'center', 'gap': '14px'}, children=[
                html.Span('🫀', style={'fontSize': '26px'}),
                html.Div([
                    html.Div('SmartCare Health Analytics',
                             style={'fontWeight': '700', 'fontSize': '18px', 'color': '#FFFFFF'}),
                    html.Div('MBI806B · Business Data Analytics · Yoobee Colleges NZ',
                             style={'fontSize': '11px', 'color': '#CBD5E1'}),
                ]),
            ]),
            html.Div(style={'display': 'flex', 'gap': '10px'}, children=[
                html.Span('Python · Dash · Plotly', style={
                    'background': 'rgba(46,134,171,0.15)', 'color': C_BLUE,
                    'border': '1px solid rgba(46,134,171,0.3)',
                    'padding': '4px 12px', 'borderRadius': '20px', 'fontSize': '11px'}),
                html.Span('50 Patients · Ages 50–79', style={
                    'background': 'rgba(6,214,160,0.1)', 'color': C_GREEN,
                    'border': '1px solid rgba(6,214,160,0.25)',
                    'padding': '4px 12px', 'borderRadius': '20px', 'fontSize': '11px'}),
            ]),
        ])
    ]),

    # Main
    html.Div(style={'padding': '24px 32px'}, children=[

        # Filters
        html.Div(style={
            'background': C_CARD, 'border': f'1px solid {C_BORDER}', 'borderRadius': '12px',
            'padding': '16px 24px', 'display': 'flex', 'gap': '24px',
            'alignItems': 'flex-end', 'flexWrap': 'wrap', 'marginBottom': '20px',
        }, children=[
            html.Span('FILTERS', style={'fontSize': '11px', 'color': C_MUTED,
                                        'letterSpacing': '1px', 'fontWeight': '600',
                                        'alignSelf': 'center'}),
            html.Div([
                html.Label('Age Group', style={'fontSize': '11px', 'color': C_MUTED,
                                               'display': 'block', 'marginBottom': '4px'}),
                dcc.Dropdown(id='dd-age', clearable=False, value='All', style={'width':'180px', **DD},
                    options=[{'label': 'All Age Groups', 'value': 'All'},
                             {'label': '50–59', 'value': '50-59'},
                             {'label': '60–69', 'value': '60-69'},
                             {'label': '70–79', 'value': '70-79'}])
            ]),
            html.Div([
                html.Label('Gender', style={'fontSize': '11px', 'color': C_MUTED,
                                            'display': 'block', 'marginBottom': '4px'}),
                dcc.Dropdown(id='dd-gender', clearable=False, value='All', style={'width':'180px', **DD},
                    options=[{'label': 'All Genders', 'value': 'All'},
                             {'label': 'Female', 'value': 'Female'},
                             {'label': 'Male',   'value': 'Male'}])
            ]),
            html.Div([
                html.Label('Risk Level', style={'fontSize': '11px', 'color': C_MUTED,
                                                'display': 'block', 'marginBottom': '4px'}),
                dcc.Dropdown(id='dd-risk', clearable=False, value='All', style={'width':'180px', **DD},
                    options=[{'label': 'All Risk Levels', 'value': 'All'},
                             {'label': 'High Risk', 'value': 'High Risk'},
                             {'label': 'Low Risk',  'value': 'Low Risk'}])
            ]),
            html.Div(id='pat-count', style={'marginLeft': 'auto', 'color': C_MUTED,
                                            'fontSize': '13px', 'alignSelf': 'center'}),
        ]),

        # KPIs
        html.Div(id='kpi-row', style={'display': 'flex', 'gap': '16px',
                                       'marginBottom': '20px', 'flexWrap': 'wrap'}),

        # Row 1: Bar + Heatmap
        html.Div(style={'display': 'grid', 'gridTemplateColumns': '1fr 1fr',
                        'gap': '16px', 'marginBottom': '16px'}, children=[
            html.Div(style=CARD_STYLE, children=[dcc.Graph(id='g-bar',     config={'displayModeBar': False})]),
            html.Div(style=CARD_STYLE, children=[dcc.Graph(id='g-heatmap', config={'displayModeBar': False})]),
        ]),

        # Row 2: Wide scatter + Donut
        html.Div(style={'display': 'grid', 'gridTemplateColumns': '2fr 1fr',
                        'gap': '16px', 'marginBottom': '16px'}, children=[
            html.Div(style=CARD_STYLE, children=[dcc.Graph(id='g-scatter', config={'displayModeBar': False})]),
            html.Div(style=CARD_STYLE, children=[dcc.Graph(id='g-donut',   config={'displayModeBar': False})]),
        ]),

        # Row 3: 4 charts
        html.Div(style={'display': 'grid', 'gridTemplateColumns': '1fr 1fr 1fr 1fr',
                        'gap': '16px', 'marginBottom': '16px'}, children=[
            html.Div(style=CARD_STYLE, children=[dcc.Graph(id='g-sleep',   config={'displayModeBar': False})]),
            html.Div(style=CARD_STYLE, children=[dcc.Graph(id='g-box',     config={'displayModeBar': False})]),
            html.Div(style=CARD_STYLE, children=[dcc.Graph(id='g-line',    config={'displayModeBar': False})]),
            html.Div(style=CARD_STYLE, children=[dcc.Graph(id='g-pie',     config={'displayModeBar': False})]),
        ]),

        # Table
        html.Div(style={**CARD_STYLE, 'padding': '16px', 'marginBottom': '20px'}, children=[
            html.H3('Patient Data Table', style={'color': C_TEXT, 'fontSize': '14px',
                                                  'fontWeight': '600', 'marginBottom': '12px'}),
            dash_table.DataTable(
                id='tbl',
                columns=[
                    {'name': 'Patient ID',    'id': 'PatientID'},
                    {'name': 'Age',           'id': 'Age'},
                    {'name': 'Gender',        'id': 'Gender'},
                    {'name': 'Age Group',     'id': 'AgeGroup'},
                    {'name': 'Daily Steps',   'id': 'DailyStepCount'},
                    {'name': 'HR Avg (BPM)',  'id': 'HeartRate_Avg'},
                    {'name': 'Sleep (hrs)',   'id': 'SleepDuration_Hours'},
                    {'name': 'Sleep Quality', 'id': 'SleepQualityScore'},
                    {'name': 'Stress Index',  'id': 'StressIndex'},
                    {'name': 'Health Score',  'id': 'WeeklyHealthScore'},
                    {'name': 'Fall Alert',    'id': 'FallAlerts'},
                    {'name': 'Risk Level',    'id': 'RiskLevel'},
                ],
                style_table={'overflowX': 'auto'},
                style_header={'backgroundColor': '#1F4E79', 'color': 'white',
                              'fontWeight': '600', 'fontSize': '12px',
                              'border': f'1px solid {C_BORDER}'},
                style_cell={'backgroundColor': C_CARD, 'color': C_TEXT,
                            'border': f'1px solid {C_BORDER}',
                            'fontSize': '12px', 'padding': '9px 12px'},
                style_data_conditional=[
                    {'if': {'row_index': 'odd'}, 'backgroundColor': '#F8FAFC'},
                    {'if': {'filter_query': '{RiskLevel} = "High Risk"', 'column_id': 'RiskLevel'},
                     'color': C_RED, 'fontWeight': '700'},
                    {'if': {'filter_query': '{RiskLevel} = "Low Risk"', 'column_id': 'RiskLevel'},
                     'color': C_GREEN, 'fontWeight': '700'},
                    {'if': {'filter_query': '{FallAlerts} = 1', 'column_id': 'FallAlerts'},
                     'color': C_YELLOW, 'fontWeight': '700'},
                    {'if': {'filter_query': '{WeeklyHealthScore} >= 80', 'column_id': 'WeeklyHealthScore'},
                     'color': C_GREEN, 'fontWeight': '700'},
                    {'if': {'filter_query': '{WeeklyHealthScore} < 65', 'column_id': 'WeeklyHealthScore'},
                     'color': C_RED, 'fontWeight': '700'},
                    {'if': {'filter_query': '{StressIndex} >= 70', 'column_id': 'StressIndex'},
                     'color': C_RED},
                ],
                sort_action='native', filter_action='native', page_size=15,
            )
        ]),
    ]),

    # Footer
    html.Div('SmartCare Health Analytics · MBI806B · Yoobee Colleges NZ · 2024/2025',
             style={'textAlign': 'center', 'padding': '16px', 'fontSize': '11px',
                    'color': C_MUTED, 'borderTop': f'1px solid {C_BORDER}',
                    'background': C_CARD}),
])


# ── Callback ──────────────────────────────────────────────────────────────────
@app.callback(
    Output('pat-count', 'children'),
    Output('kpi-row',   'children'),
    Output('g-bar',     'figure'),
    Output('g-heatmap', 'figure'),
    Output('g-scatter', 'figure'),
    Output('g-donut',   'figure'),
    Output('g-sleep',   'figure'),
    Output('g-box',     'figure'),
    Output('g-line',    'figure'),
    Output('g-pie',     'figure'),
    Output('tbl',       'data'),
    Input('dd-age',    'value'),
    Input('dd-gender', 'value'),
    Input('dd-risk',   'value'),
)
def update(age, gender, risk):
    df = df_full.copy()
    if age    != 'All': df = df[df['AgeGroup']  == age]
    if gender != 'All': df = df[df['Gender']    == gender]
    if risk   != 'All': df = df[df['RiskLevel'] == risk]

    n = len(df)
    count_label = f"Showing {n} of {len(df_full)} patients"

    if n == 0:
        blank = empty_chart()
        kpis  = html.Div(kpi_card('No Data', '—', 'Adjust your filters', C_MUTED, '⚠️'))
        return count_label, kpis, blank, blank, blank, blank, blank, blank, blank, blank, []

    # KPIs
    avg_steps  = f"{df['DailyStepCount'].mean():,.0f}"
    high_n     = int(df['HighRisk'].sum())
    risk_pct   = f"{high_n / n * 100:.0f}%"
    avg_sleep  = f"{df['SleepDuration_Hours'].mean():.1f} h"
    avg_health = f"{df['WeeklyHealthScore'].mean():.1f}"
    fall_n     = int(df['FallAlerts'].sum())

    kpis = html.Div(style={'display': 'flex', 'gap': '16px',
                            'width': '100%', 'flexWrap': 'wrap'}, children=[
        kpi_card('Avg Daily Steps',    avg_steps,  'Goal: 10,000 steps/day',          C_BLUE,   '🏃'),
        kpi_card('High-Risk Patients', risk_pct,   f'{high_n} of {n} patients',       C_RED,    '⚠️'),
        kpi_card('Avg Sleep Duration', avg_sleep,  'Recommended: 7–9 hrs/night',      C_YELLOW, '💤'),
        kpi_card('Avg Health Score',   avg_health, f'Fall alerts: {fall_n} patients', C_GREEN,  '💚'),
    ])

    tbl_data = df[['PatientID','Age','Gender','AgeGroup','DailyStepCount',
                   'HeartRate_Avg','SleepDuration_Hours','SleepQualityScore',
                   'StressIndex','WeeklyHealthScore','FallAlerts','RiskLevel']
                 ].to_dict('records')

    def safe(fn):
        try:
            return fn(df)
        except Exception as e:
            print(f"  ⚠ Chart error [{fn.__name__}]: {e}")
            return empty_chart(f'Error in {fn.__name__}')

    return (
        count_label, kpis,
        safe(chart_bar),
        safe(chart_heatmap),
        safe(chart_scatter_steps),
        safe(chart_donut),
        safe(chart_sleep_stress),
        safe(chart_box),
        safe(chart_line_calories),
        safe(chart_pie_fall),
        tbl_data,
    )


# ── Entry Point ───────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("\n" + "="*60)
    print("  SmartCare Health Analytics Dashboard")
    print("  MBI806B — Business Data Analytics")
    print("="*60)
    print("  Open browser → http://127.0.0.1:8050")
    print("  Press Ctrl+C to stop.\n")
    app.run(debug=False, host='127.0.0.1', port=8050)

# pip install dash
# pip install dash-bootstrap-components
# pip install plotly
# pip install pandas
# pip install numpy

#run by clicking the http://..... link in terminal


import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
from collections import deque

# Initialize Dash app with dark theme
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])

# Store for real-time data
anomaly_data = deque(maxlen=100)
alerts_data = deque(maxlen=20)

# Layout
app.layout = dbc.Container([
    # Header
    dbc.Row([
        dbc.Col([
            html.H1("🛡️ Smart Grid Security Monitor", className="text-center mb-4"),
            html.H4("AI-Powered Threat Detection System", className="text-center text-muted")
        ])
    ]),
    
    # Control Panel
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H5("System Controls"),
                    dbc.Label("Threat Simulation Level"),
                    dcc.Slider(
                        id='threat-slider',
                        min=0,
                        max=3,
                        marks={0: 'Normal', 1: 'Elevated', 2: 'High', 3: 'Critical'},
                        value=0,
                        className="mb-3"
                    ),
                    dbc.Label("AI Confidence Threshold"),
                    dcc.Slider(
                        id='confidence-slider',
                        min=50,
                        max=95,
                        step=5,
                        value=75,
                        marks={i: f'{i}%' for i in range(50, 100, 10)},
                        className="mb-3"
                    ),
                ])
            ], className="mb-4")
        ], width=12)
    ]),
    
    # Main Dashboard
    dbc.Row([
        # Left Column - Alerts
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.H5("🚨 Active Alerts")),
                dbc.CardBody([
                    html.Div(id='alerts-container', style={'height': '400px', 'overflowY': 'auto'})
                ])
            ])
        ], width=3),
        
        # Middle Column - Grid Map
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.H5("🗺️ Grid Infrastructure Status")),
                dbc.CardBody([
                    dcc.Graph(id='grid-map', style={'height': '400px'})
                ])
            ])
        ], width=6),
        
        # Right Column - Metrics
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.H5("📊 System Metrics")),
                dbc.CardBody([
                    html.Div(id='metrics-container')
                ])
            ])
        ], width=3),
    ], className="mb-4"),
    
    # Anomaly Detection Chart
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.H5("📈 Real-Time Anomaly Detection")),
                dbc.CardBody([
                    dcc.Graph(id='anomaly-chart', style={'height': '350px'})
                ])
            ])
        ], width=8),
        
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.H5("🤖 AI Insights")),
                dbc.CardBody([
                    html.Div(id='ai-insights', style={'height': '350px'})
                ])
            ])
        ], width=4),
    ], className="mb-4"),
    
    # Protocol Analysis
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.H5("🔍 Protocol Traffic Analysis")),
                dbc.CardBody([
                    dcc.Graph(id='protocol-chart', style={'height': '300px'})
                ])
            ])
        ], width=6),
        
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.H5("⚡ Threat Timeline")),
                dbc.CardBody([
                    dcc.Graph(id='threat-timeline', style={'height': '300px'})
                ])
            ])
        ], width=6),
    ]),
    
    # Auto-refresh interval
    dcc.Interval(
        id='interval-component',
        interval=2000,  # Update every 2 seconds
        n_intervals=0
    )
], fluid=True, className="p-4")

# Helper functions
def generate_alert(threat_level, confidence_threshold):
    """Generate realistic alerts based on threat level"""
    alert_configs = {
        0: {'critical': 0.05, 'warning': 0.1, 'info': 0.2},  # Normal
        1: {'critical': 0.15, 'warning': 0.3, 'info': 0.3},  # Elevated
        2: {'critical': 0.3, 'warning': 0.4, 'info': 0.2},   # High
        3: {'critical': 0.5, 'warning': 0.3, 'info': 0.1}    # Critical
    }
    
    alert_messages = {
        'critical': [
            "Unauthorized DNP3 write command detected to critical PLC",
            "Multiple authentication failures from external IP range",
            "Abnormal frequency deviation detected - possible grid manipulation",
            "IEC 61850 GOOSE message replay attack in progress",
            "Malware signature detected on HMI workstation"
        ],
        'warning': [
            "Unusual Modbus traffic pattern detected on TCP/502",
            "Configuration change on RTU without authorization",
            "Voltage readings 15% above normal at Substation Delta",
            "Increased latency on SCADA network segment",
            "New device detected on OT network - verification required"
        ],
        'info': [
            "Scheduled maintenance window activated",
            "Backup communication channel test successful",
            "Firmware update available for protection relays"
        ]
    }
    
    probabilities = alert_configs[threat_level]
    
    for alert_type, prob in probabilities.items():
        if random.random() < prob:
            return {
                'id': datetime.now().timestamp(),
                'type': alert_type,
                'message': random.choice(alert_messages[alert_type]),
                'confidence': random.randint(confidence_threshold, 95),
                'timestamp': datetime.now(),
                'protocol': random.choice(['DNP3', 'Modbus', 'IEC 61850', 'IEC 60870-5-104'])
            }
    return None

def create_alert_card(alert):
    """Create alert card with appropriate styling"""
    color_map = {
        'critical': 'danger',
        'warning': 'warning',
        'info': 'info'
    }
    
    icon_map = {
        'critical': '🔴',
        'warning': '🟡',
        'info': '🔵'
    }
    
    return dbc.Alert([
        html.H6([icon_map[alert['type']], f" {alert['type'].upper()}"], className="mb-1"),
        html.P(alert['message'], className="mb-1"),
        html.Small([
            f"Protocol: {alert['protocol']} | ",
            f"AI Confidence: {alert['confidence']}% | ",
            f"{alert['timestamp'].strftime('%H:%M:%S')}"
        ])
    ], color=color_map[alert['type']], className="mb-2")

# Callbacks
@app.callback(
    [Output('alerts-container', 'children'),
     Output('grid-map', 'figure'),
     Output('anomaly-chart', 'figure'),
     Output('metrics-container', 'children'),
     Output('ai-insights', 'children'),
     Output('protocol-chart', 'figure'),
     Output('threat-timeline', 'figure')],
    [Input('interval-component', 'n_intervals')],
    [State('threat-slider', 'value'),
     State('confidence-slider', 'value')]
)
def update_dashboard(n, threat_level, confidence_threshold):
    # Generate new alert
    new_alert = generate_alert(threat_level, confidence_threshold)
    if new_alert:
        alerts_data.append(new_alert)
    
    # Update alerts display
    alerts_display = [create_alert_card(alert) for alert in reversed(list(alerts_data)[-5:])]
    
    # Create grid map
    substations = pd.DataFrame({
        'name': ['North Station', 'East Station', 'South Station', 'West Station', 'Central Hub'],
        'lat': [40.7128, 40.7580, 40.6892, 40.7489, 40.7282],
        'lon': [-74.0060, -73.9855, -74.0445, -73.9680, -73.9942],
        'status': ['Normal', 'Warning', 'Critical', 'Normal', 'Warning'] if threat_level > 1 
                  else ['Normal', 'Normal', 'Warning', 'Normal', 'Normal'],
        'load': [75 + random.randint(-10, 10) for _ in range(5)]
    })
    
    fig_map = px.scatter_mapbox(
        substations,
        lat='lat',
        lon='lon',
        color='status',
        size='load',
        color_discrete_map={'Normal': '#00ff00', 'Warning': '#ffaa00', 'Critical': '#ff0000'},
        hover_data=['name', 'load'],
        zoom=10,
        height=350
    )
    
    fig_map.update_layout(
        mapbox_style="carto-darkmatter",
        margin={"r":0,"t":0,"l":0,"b":0},
        showlegend=True,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    # Generate anomaly data
    current_time = datetime.now()
    time_points = [current_time - timedelta(minutes=i) for i in range(100, 0, -1)]
    
    # Normal baseline
    normal_values = 50 + 10 * np.sin(np.linspace(0, 4*np.pi, 100)) + np.random.normal(0, 2, 100)
    
    # Actual values with anomalies
    actual_values = normal_values.copy()
    if threat_level >= 2:
        anomaly_indices = np.random.choice(range(70, 100), size=5, replace=False)
        for idx in anomaly_indices:
            actual_values[idx] += np.random.uniform(15, 30) * np.random.choice([-1, 1])
    
    # Anomaly scores
    anomaly_scores = np.abs(actual_values - normal_values) * 5
    
    # Create anomaly chart
    fig_anomaly = go.Figure()
    
    fig_anomaly.add_trace(go.Scatter(
        x=time_points,
        y=normal_values,
        mode='lines',
        name='Expected Baseline',
        line=dict(color='green', width=2)
    ))
    
    fig_anomaly.add_trace(go.Scatter(
        x=time_points,
        y=actual_values,
        mode='lines',
        name='Actual Values',
        line=dict(color='cyan', width=2)
    ))
    
    fig_anomaly.add_trace(go.Scatter(
        x=time_points,
        y=anomaly_scores,
        mode='lines',
        name='Anomaly Score',
        line=dict(color='red', width=1, dash='dash'),
        yaxis='y2'
    ))
    
    fig_anomaly.update_layout(
        template='plotly_dark',
        yaxis=dict(title='Power (MW)'),
        yaxis2=dict(title='Anomaly Score', overlaying='y', side='right'),
        hovermode='x unified',
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    # Metrics
    metrics = dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4(f"{random.randint(75, 95)}%", className="text-center"),
                    html.P("Grid Load", className="text-center text-muted")
                ])
            ], color="primary", outline=True)
        ], width=6),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4(f"{len([a for a in alerts_data if a['type'] == 'critical'])}", 
                           className="text-center text-danger"),
                    html.P("Active Threats", className="text-center text-muted")
                ])
            ], color="danger", outline=True)
        ], width=6),
    ], className="mb-3")
    
    metrics2 = dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4(f"{random.randint(85, 98)}%", className="text-center"),
                    html.P("AI Confidence", className="text-center text-muted")
                ])
            ], color="success", outline=True)
        ], width=6),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4(f"{random.randint(50, 200)}ms", className="text-center"),
                    html.P("Response Time", className="text-center text-muted")
                ])
            ], color="info", outline=True)
        ], width=6),
    ])
    
    metrics_display = html.Div([metrics, metrics2])
    
    # AI Insights
    insights = [
        "🔍 Pattern analysis detected coordinated scanning activity across multiple substations",
        "⚡ ML model identified unusual power flow patterns consistent with false data injection",
        "🎯 Predictive analytics indicate 78% probability of attack escalation in next 2 hours",
        "🛡️ Recommended action: Isolate East Station and enable backup communication channels"
    ]
    
    ai_insights_display = [
        dbc.Alert(insight, color="info", className="mb-2") 
        for insight in insights[:3]
    ]
    
    # Protocol chart
    protocols = ['DNP3', 'Modbus', 'IEC 61850', 'IEC 60870-5-104']
    protocol_data = pd.DataFrame({
        'Protocol': protocols,
        'Normal Traffic': [random.randint(1000, 5000) for _ in protocols],
        'Anomalous Traffic': [random.randint(0, 500) * (threat_level + 1) for _ in protocols]
    })
    
    fig_protocol = px.bar(
        protocol_data,
        x='Protocol',
        y=['Normal Traffic', 'Anomalous Traffic'],
        title='Protocol Traffic Distribution',
        template='plotly_dark',
        color_discrete_map={'Normal Traffic': '#00ff00', 'Anomalous Traffic': '#ff0000'}
    )


    
    fig_protocol.update_layout(
        margin=dict(l=0, r=0, t=30, b=0),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    # Threat timeline
    threat_history = pd.DataFrame({
        'Time': [current_time - timedelta(hours=i) for i in range(24, 0, -1)],
        'Threat Level': [random.randint(20 + threat_level*20, 40 + threat_level*20) for _ in range(24)]
    })
    
    fig_threat = px.area(
        threat_history,
        x='Time',
        y='Threat Level',
        title='24-Hour Threat Level Trend',
        template='plotly_dark',
        color_discrete_sequence=['red']
    )
    
    fig_threat.add_hline(y=confidence_threshold, line_dash="dash", 
                         annotation_text="Alert Threshold")
    
    fig_threat.update_layout(
        margin=dict(l=0, r=0, t=30, b=0),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    return (alerts_display, fig_map, fig_anomaly, metrics_display, 
            ai_insights_display, fig_protocol, fig_threat)

# Run the app 
if __name__ == '__main__':

    app.run(debug=True, port=8050)  # NOT app.run_server

# pip install dash dash-bootstrap-components plotly pandas numpy

import dash
from dash import dcc, html, Input, Output, State, dash_table, callback_context
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
from collections import deque
import json

# Initialize Dash app with dark theme
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])
app.config.suppress_callback_exceptions = True

# Store for real-time data
anomaly_data = deque(maxlen=100)
alerts_data = deque(maxlen=20)
sensor_logs = deque(maxlen=200)
protocol_events = deque(maxlen=500)

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
    
    # Sensor & Log Correlation Timeline
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader([
                    html.H5("🔄 Sensor & Log Correlation Timeline"),
                    dbc.Badge("NEW", color="success", className="ms-2")
                ]),
                dbc.CardBody([
                    dcc.Graph(id='correlation-timeline', style={'height': '400px'}),
                    html.Hr(),
                    html.H6("Event Details", className="mb-3"),
                    html.Div(id='event-details', style={'height': '200px', 'overflowY': 'auto'})
                ])
            ])
        ], width=12),
    ], className="mb-4"),
    
    # Anomaly Detection Chart and AI Insights
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
                    html.Div(id='ai-insights', style={'height': '350px', 'overflowY': 'auto'})
                ])
            ])
        ], width=4),
    ], className="mb-4"),
    
    # Protocol-Aware Traffic Analysis
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader([
                    html.H5("🔍 Protocol-Aware Traffic Analysis"),
                    dbc.Badge("ENHANCED", color="info", className="ms-2")
                ]),
                dbc.CardBody([
                    dbc.Tabs([
                        dbc.Tab(label="Protocol Overview", tab_id="protocol-overview"),
                        dbc.Tab(label="Command Analysis", tab_id="command-analysis"),
                        dbc.Tab(label="Anomalous Patterns", tab_id="anomalous-patterns"),
                    ], id="protocol-tabs", active_tab="protocol-overview"),
                    html.Div(id='protocol-content', className="mt-3")
                ])
            ])
        ], width=12),
    ], className="mb-4"),
    
    # Threat Timeline
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.H5("⚡ Threat Timeline")),
                dbc.CardBody([
                    dcc.Graph(id='threat-timeline', style={'height': '300px'})
                ])
            ])
        ], width=12),
    ]),
    
    # Auto-refresh interval
    dcc.Interval(
        id='interval-component',
        interval=2000,  # Update every 2 seconds
        n_intervals=0
    ),
    
    # Hidden div to store intermediate values
    html.Div(id='intermediate-value', style={'display': 'none'})
], fluid=True, className="p-4")

# Helper functions
def generate_alert(threat_level, confidence_threshold):
    """Generate realistic alerts based on threat level"""
    alert_configs = {
        0: {'critical': 0.05, 'warning': 0.1, 'info': 0.2},
        1: {'critical': 0.15, 'warning': 0.3, 'info': 0.3},
        2: {'critical': 0.3, 'warning': 0.4, 'info': 0.2},
        3: {'critical': 0.5, 'warning': 0.3, 'info': 0.1}
    }
    
    alert_messages = {
        'critical': [
            "Unauthorized DNP3 write command detected to critical PLC",
            "Multiple authentication failures from external IP range",
            "Abnormal frequency deviation detected - possible grid manipulation",
            "IEC 61850 GOOSE message replay attack in progress",
            "Repeated breaker flip commands detected - possible sabotage"
        ],
        'warning': [
            "Unusual Modbus traffic pattern detected on TCP/502",
            "Configuration change on RTU without authorization",
            "Voltage readings 15% above normal at Substation Delta",
            "DNP3 cold restart command from unknown source"
        ],
        'info': [
            "Scheduled maintenance window activated",
            "Backup communication channel test successful"
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

def generate_sensor_logs(threat_level):
    """Generate correlated sensor and SCADA logs"""
    current_time = datetime.now()
    events = []
    
    # Simple event generation
    for i in range(3):
        timestamp = current_time - timedelta(seconds=i*10)
        
        # Sensor event
        if threat_level > 1 and random.random() > 0.5:
            event_type = 'voltage_spike'
            status = 'critical'
            value = 145.8
        else:
            event_type = 'voltage_reading'
            status = 'normal'
            value = 120.5
            
        events.append({
            'timestamp': timestamp,
            'source': 'sensor',
            'event': {
                'type': event_type,
                'value': value,
                'unit': 'V',
                'status': status
            },
            'correlation_id': f"CID_{random.randint(1000, 9999)}"
        })
    
    return events

def generate_protocol_events(threat_level):
    """Generate protocol-specific events"""
    events = []
    current_time = datetime.now()
    
    protocols = ['DNP3', 'Modbus', 'IEC 61850']
    commands = {
        'DNP3': ['READ', 'WRITE', 'SELECT', 'OPERATE'],
        'Modbus': ['READ_COILS', 'WRITE_COILS', 'READ_REGISTERS'],
        'IEC 61850': ['GetDataValues', 'SetDataValues', 'Report']
    }
    
    for i in range(5):
        protocol = random.choice(protocols)
        command = random.choice(commands[protocol])
        
        is_anomaly = threat_level > 1 and random.random() > 0.6
        
        events.append({
            'timestamp': current_time - timedelta(seconds=random.randint(0, 60)),
            'protocol': protocol,
            'command': command,
            'source_ip': f"192.168.1.{random.randint(1, 254)}",
            'destination': f"RTU_{random.randint(1, 10)}",
            'is_anomaly': is_anomaly,
            'anomaly_type': 'Suspicious Command' if is_anomaly else None,
            'response_time': random.randint(50, 500)
        })
    
    return sorted(events, key=lambda x: x['timestamp'], reverse=True)

# Main callback
@app.callback(
    [Output('alerts-container', 'children'),
     Output('grid-map', 'figure'),
     Output('anomaly-chart', 'figure'),
     Output('metrics-container', 'children'),
     Output('ai-insights', 'children'),
     Output('correlation-timeline', 'figure'),
     Output('event-details', 'children'),
     Output('threat-timeline', 'figure')],
    [Input('interval-component', 'n_intervals')],
    [State('threat-slider', 'value'),
     State('confidence-slider', 'value')]
)
def update_main_dashboard(n, threat_level, confidence_threshold):
    # Initialize threat_level and confidence_threshold if None
    if threat_level is None:
        threat_level = 0
    if confidence_threshold is None:
        confidence_threshold = 75
        
    # Generate new data
    new_alert = generate_alert(threat_level, confidence_threshold)
    if new_alert:
        alerts_data.append(new_alert)
    
    new_sensor_logs = generate_sensor_logs(threat_level)
    sensor_logs.extend(new_sensor_logs)
    
    new_protocol_events = generate_protocol_events(threat_level)
    protocol_events.extend(new_protocol_events)
    
    # Create visualizations
    # 1. Alerts
    alerts_display = []
    if alerts_data:
        alerts_display = [create_alert_card(alert) for alert in reversed(list(alerts_data)[-5:])]
    else:
        alerts_display = [html.P("No active alerts", className="text-muted")]
    
    # 2. Grid Map
    substations = pd.DataFrame({
        'name': ['North Station', 'East Station', 'South Station', 'West Station', 'Central Hub'],
        'lat': [40.7128, 40.7580, 40.6892, 40.7489, 40.7282],
        'lon': [-74.0060, -73.9855, -74.0445, -73.9680, -73.9942],
        'status': ['Normal', 'Warning', 'Critical', 'Normal', 'Warning'] if threat_level > 1 
                  else ['Normal', 'Normal', 'Normal', 'Normal', 'Normal'],
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
        zoom=10
    )
    
    fig_map.update_layout(
        mapbox_style="carto-darkmatter",
        margin={"r":0,"t":0,"l":0,"b":0},
        height=350
    )
    
    # 3. Anomaly Chart
    current_time = datetime.now()
    time_points = [current_time - timedelta(minutes=i) for i in range(60, 0, -1)]
    normal_values = 50 + 10 * np.sin(np.linspace(0, 4*np.pi, 60))
    actual_values = normal_values + np.random.normal(0, 2, 60)
    
    if threat_level >= 2:
        # Add anomalies
        for i in random.sample(range(40, 60), 5):
            actual_values[i] += random.uniform(10, 20) * random.choice([-1, 1])
    
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
    
    fig_anomaly.update_layout(
        template='plotly_dark',
        yaxis=dict(title='Power (MW)'),
        hovermode='x unified',
        margin=dict(l=0, r=0, t=0, b=0),
        height=300
    )
    
    # 4. Metrics
    metrics_display = html.Div([
        dbc.Row([
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
        ], className="mb-3"),
        dbc.Row([
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
    ])
    
    # 5. AI Insights
    insights = [
        "🔍 Pattern analysis detected coordinated scanning activity across multiple substations",
        "⚡ ML model identified unusual power flow patterns consistent with false data injection",
        "🎯 Predictive analytics indicate 78% probability of attack escalation in next 2 hours",
        "📊 Correlation detected between DNP3 anomalies and voltage spikes"
    ]
    
    ai_insights_display = [
        dbc.Alert(insight, color="info", className="mb-2") 
        for insight in insights[:3]
    ]
    
    # 6. Sensor & Log Correlation Timeline
    fig_correlation = go.Figure()
    
    # Get recent sensor logs
    recent_logs = list(sensor_logs)[-20:] if sensor_logs else []
    
    if recent_logs:
        # Group by source
        for i, source in enumerate(['sensor', 'scada', 'network']):
            source_events = [log for log in recent_logs if log.get('source') == source]
            
            if source_events:
                timestamps = [event['timestamp'] for event in source_events]
                y_position = 3 - i  # sensor=3, scada=2, network=1
                
                # Determine colors
                colors = []
                texts = []
                for event in source_events:
                    event_data = event.get('event', {})
                    status = event_data.get('status', 'normal')
                    
                    if status == 'critical':
                        colors.append('red')
                    elif status == 'warning':
                        colors.append('orange')
                    else:
                        colors.append('green')
                    
                    # Create hover text
                    if 'type' in event_data:
                        texts.append(f"{event_data['type']}: {event_data.get('value', 'N/A')}")
                    else:
                        texts.append(source)
                
                fig_correlation.add_trace(go.Scatter(
                    x=timestamps,
                    y=[y_position] * len(timestamps),
                    mode='markers',
                    name=source.capitalize(),
                    marker=dict(size=12, color=colors),
                    text=texts,
                    hovertemplate='%{text}<br>Time: %{x}<extra></extra>'
                ))
    
    fig_correlation.update_layout(
        title="Event Correlation Timeline",
        yaxis=dict(
            tickmode='array',
            tickvals=[1, 2, 3],
            ticktext=['Network', 'SCADA', 'Sensor'],
            range=[0.5, 3.5]
        ),
        xaxis=dict(title='Time'),
        template='plotly_dark',
        height=350,
        showlegend=True,
        margin=dict(l=0, r=0, t=30, b=0)
    )
    
    # 7. Event Details - FIXED VERSION
    if recent_logs:
        recent_event = recent_logs[-1]
        # Using a simple div with styled content instead of dbc.Table
        event_details = html.Div([
            html.Table([
                html.Tbody([
                    html.Tr([
                        html.Td("Timestamp", style={'fontWeight': 'bold', 'padding': '8px'}),
                        html.Td(recent_event['timestamp'].strftime('%H:%M:%S'), style={'padding': '8px'})
                    ]),
                    html.Tr([
                        html.Td("Source", style={'fontWeight': 'bold', 'padding': '8px'}),
                        html.Td(recent_event.get('source', 'Unknown').upper(), style={'padding': '8px'})
                    ]),
                    html.Tr([
                        html.Td("Event Type", style={'fontWeight': 'bold', 'padding': '8px'}),
                        html.Td(str(recent_event.get('event', {}).get('type', 'N/A')), style={'padding': '8px'})
                    ]),
                    html.Tr([
                        html.Td("Status", style={'fontWeight': 'bold', 'padding': '8px'}),
                        html.Td(str(recent_event.get('event', {}).get('status', 'N/A')), style={'padding': '8px'})
                    ])
                ])
            ], style={'width': '100%', 'borderCollapse': 'collapse', 'border': '1px solid #444'})
        ], style={'backgroundColor': '#1a1a1a', 'padding': '10px', 'borderRadius': '5px'})
    else:
        event_details = html.P("No events to display", className="text-muted")
    
    # 8. Threat Timeline
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
        height=250
    )
    
    return (alerts_display, fig_map, fig_anomaly, metrics_display, 
            ai_insights_display, fig_correlation, event_details, fig_threat)

# Separate callback for protocol content
@app.callback(
    Output('protocol-content', 'children'),
    [Input('protocol-tabs', 'active_tab'),
     Input('interval-component', 'n_intervals')]
)
def update_protocol_content(active_tab, n):
    if not active_tab:
        active_tab = "protocol-overview"
    
    recent_events = list(protocol_events)[-50:] if protocol_events else []
    
    if active_tab == "protocol-overview":
        # Protocol distribution
        if recent_events:
            protocol_counts = {}
            for event in recent_events:
                protocol = event.get('protocol', 'Unknown')
                if protocol not in protocol_counts:
                    protocol_counts[protocol] = {'normal': 0, 'anomalous': 0}
                
                if event.get('is_anomaly', False):
                    protocol_counts[protocol]['anomalous'] += 1
                else:
                    protocol_counts[protocol]['normal'] += 1
            
            protocol_df = pd.DataFrame([
                {'Protocol': p, 'Normal Traffic': counts['normal'], 'Anomalous Traffic': counts['anomalous']}
                for p, counts in protocol_counts.items()
            ])
            
            if not protocol_df.empty:
                fig_protocol = px.bar(
                    protocol_df,
                    x='Protocol',
                    y=['Normal Traffic', 'Anomalous Traffic'],
                    title='Protocol Traffic Distribution',
                    template='plotly_dark',
                    color_discrete_map={'Normal Traffic': '#00ff00', 'Anomalous Traffic': '#ff0000'}
                )
                
                return dcc.Graph(figure=fig_protocol, style={'height': '400px'})
        
        return html.P("No protocol data available", className="text-muted")
    
    elif active_tab == "command-analysis":
        # Command table
        if recent_events:
            table_data = []
            for event in recent_events[:20]:  # Show last 20
                table_data.append({
                    'Time': event['timestamp'].strftime('%H:%M:%S'),
                    'Protocol': event.get('protocol', 'Unknown'),
                    'Command': event.get('command', 'Unknown'),
                    'Source': event.get('source_ip', 'Unknown'),
                    'Destination': event.get('destination', 'Unknown'),
                    'Response Time': f"{event.get('response_time', 0)}ms",
                    'Status': '🔴 Anomaly' if event.get('is_anomaly', False) else '🟢 Normal'
                })
            
            return html.Div([
                html.H6("Real-time Protocol Command Analysis"),
                html.P("Detecting unusual commands like repeated breaker flips or abnormal setpoints"),
                dash_table.DataTable(
                    data=table_data,
                    columns=[
                        {'name': col, 'id': col} for col in 
                        ['Time', 'Protocol', 'Command', 'Source', 'Destination', 'Response Time', 'Status']
                    ],
                    style_cell={'textAlign': 'left', 'backgroundColor': 'rgb(50, 50, 50)', 'color': 'white'},
                    style_data_conditional=[
                        {
                            'if': {'column_id': 'Status', 'filter_query': '{Status} contains "Anomaly"'},
                            'backgroundColor': 'rgb(100, 0, 0)',
                        }
                    ],
                    style_header={'backgroundColor': 'rgb(30, 30, 30)', 'fontWeight': 'bold'},
                    page_size=10
                )
            ])
        
        return html.P("No command data available", className="text-muted")
    
    else:  # anomalous-patterns
        # Anomaly patterns
        if recent_events:
            anomaly_events = [e for e in recent_events if e.get('is_anomaly', False)]
            
            if anomaly_events:
                return html.Div([
                    html.H6("Detected Anomalous Patterns"),
                    html.Div([
                        dbc.Alert([
                            html.Strong(f"{event.get('protocol', 'Unknown')} - {event.get('command', 'Unknown')}"),
                            html.Br(),
                            html.Small(f"Type: {event.get('anomaly_type', 'Unknown')} | "
                                     f"Source: {event.get('source_ip', 'Unknown')} | "
                                     f"Time: {event['timestamp'].strftime('%H:%M:%S')}")
                        ], color="danger", className="mb-2")
                        for event in anomaly_events[:5]  # Show last 5
                    ])
                ])
        
        return html.P("No anomalous patterns detected", className="text-muted")

# Run the app 
if __name__ == '__main__':
    app.run(debug=True, port=8050)

import dash
from dash import dcc, html, Input, Output, State, dash_table
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
from collections import deque
from tcp_data_reader import TCPDataReader

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])
app.config.suppress_callback_exceptions = True

data_reader = TCPDataReader('tcp_scaled_15102025.csv')

alerts_data = deque(maxlen=20)
protocol_events = deque(maxlen=500)
response_time_data = deque(maxlen=60)
sensor_logs = deque(maxlen=200)

def generate_sensor_logs(threat_level):
    current_time = datetime.now()
    events = []
    num_event_sets = 1 + threat_level
    
    for i in range(num_event_sets):
        correlation_id = f"CID_{random.randint(1000, 9999)}"
        base_time = current_time - timedelta(seconds=random.randint(i * 10, (i+1) * 10))
        is_anomaly = random.random() < (0.1 + threat_level * 0.25)

        events.append({
            'timestamp': base_time, 'source': 'network',
            'event': {'status': 'suspicious' if is_anomaly else 'ok'},
            'correlation_id': correlation_id
        })
        events.append({
            'timestamp': base_time + timedelta(seconds=random.uniform(1, 2)), 'source': 'scada',
            'event': {'status': 'critical' if is_anomaly else 'normal'},
            'correlation_id': correlation_id
        })
        events.append({
            'timestamp': base_time + timedelta(seconds=random.uniform(2, 3)), 'source': 'sensor',
            'event': {'status': 'critical' if is_anomaly else 'normal'},
            'correlation_id': correlation_id
        })
    return events

def create_alert_from_row(row):
    if row['is_anomaly'] == 1:
        return {
            'id': row['Time'],
            'type': 'critical',
            'message': f"Anomalous TCP traffic detected from {row['Source']} to {row['Destination']}",
            'confidence': row['confidence_score'],
            'timestamp': datetime.now(),
            'protocol': row['Protocol']
        }
    return None

def create_alert_card(alert):
    color_map = {'critical': 'danger', 'warning': 'warning', 'info': 'info'}
    icon_map = {'critical': '🔴', 'warning': '🟡', 'info': '🔵'}
    
    return dbc.Alert([
        html.H6([icon_map[alert['type']], f" {alert['type'].upper()}"], className="mb-1"),
        html.P(alert['message'], className="mb-1"),
        html.Small([
            f"Protocol: {alert['protocol']} | ",
            f"AI Confidence: {alert['confidence']}% | ",
            f"{alert['timestamp'].strftime('%H:%M:%S')}"
        ])
    ], color=color_map[alert['type']], className="mb-2")

app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H1("🛡️ Smart Grid Security Monitor", className="text-center mb-4"),
            html.H4("AI Powered Threat Detection System", className="text-center text-muted")
        ])
    ]),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H5("System Controls"),
                    dbc.Label("Threat Simulation Level (Now controls anomaly frequency)"),
                    dcc.Slider(
                        id='threat-slider', min=0, max=3,
                        marks={0: 'Normal', 1: 'Elevated', 2: 'High', 3: 'Critical'},
                        value=0, className="mb-3"
                    ),
                    dbc.Label("AI Confidence Threshold"),
                    dcc.Slider(
                        id='confidence-slider', min=50, max=95, step=5, value=75,
                        marks={i: f'{i}%' for i in range(50, 100, 10)}, className="mb-3"
                    ),
                ])
            ], className="mb-4")
        ], width=12)
    ]),
    
    dbc.Row([
        dbc.Col(dbc.Card([
            dbc.CardHeader(html.H5("🚨 Active Alerts")),
            dbc.CardBody(html.Div(id='alerts-container', style={'height': '400px', 'overflowY': 'auto'}))
        ]), width=3),
        dbc.Col(dbc.Card([
            dbc.CardHeader(html.H5("🗺️ Grid Infrastructure Status")),
            dbc.CardBody(dcc.Graph(id='grid-map', style={'height': '400px'}))
        ]), width=6),
        dbc.Col(dbc.Card([
            dbc.CardHeader(html.H5("📊 System Metrics")),
            dbc.CardBody(html.Div(id='metrics-container'))
        ]), width=3),
    ], className="mb-4"),
    
    dbc.Row([
        dbc.Col(dbc.Card([
            dbc.CardHeader(html.H5("🔄 Sensor & Log Correlation Timeline")),
            dbc.CardBody([
                dcc.Graph(id='correlation-timeline', style={'height': '400px'}),
            ])
        ]), width=12),
    ], className="mb-4"),
    
    dbc.Row([
        dbc.Col(dbc.Card([
            dbc.CardHeader(html.H5("📈 Real Time Anomaly Detection")),
            dbc.CardBody(dcc.Graph(id='anomaly-chart', style={'height': '350px'}))
        ]), width=8),
        dbc.Col(dbc.Card([
            dbc.CardHeader(html.H5("🤖 AI Insights")),
            dbc.CardBody(html.Div(id='ai-insights', style={'height': '350px', 'overflowY': 'auto'}))
        ]), width=4),
    ], className="mb-4"),
    
    dbc.Row([
        dbc.Col(dbc.Card([
            dbc.CardHeader(html.H5("🔍 Protocol-Aware Traffic Analysis (from CSV)")),
            dbc.CardBody([
                dbc.Tabs([
                    dbc.Tab(label="Protocol Overview", tab_id="protocol-overview"),
                    dbc.Tab(label="Command Analysis", tab_id="command-analysis"),
                    dbc.Tab(label="Anomalous Patterns", tab_id="anomalous-patterns"),
                ], id="protocol-tabs", active_tab="protocol-overview"),
                html.Div(id='protocol-content', className="mt-3")
            ])
        ]), width=12),
    ], className="mb-4"),
    
    dbc.Row([
        dbc.Col(dbc.Card([
            dbc.CardHeader(html.H5("⚡ Threat Timeline")),
            dbc.CardBody(dcc.Graph(id='threat-timeline', style={'height': '300px'}))
        ]), width=12),
    ]),
    
    dcc.Interval(id='interval-component', interval=2000, n_intervals=0),
], fluid=True, className="p-4")

@app.callback(
    [Output('alerts-container', 'children'),
     Output('grid-map', 'figure'),
     Output('anomaly-chart', 'figure'),
     Output('metrics-container', 'children'),
     Output('ai-insights', 'children'),
     Output('threat-timeline', 'figure'),
     Output('correlation-timeline', 'figure')],
    [Input('interval-component', 'n_intervals')],
    [State('threat-slider', 'value'),
     State('confidence-slider', 'value')]
)
def update_main_dashboard(n, threat_level, confidence_threshold):
    packet = data_reader.get_next_packet()
    if packet is None:
        return [html.P("Error: Could not read TCP data file.")] * 7

    new_alert = create_alert_from_row(packet)
    if new_alert and new_alert['confidence'] >= confidence_threshold:
        alerts_data.append(new_alert)
    
    alerts_display = [create_alert_card(alert) for alert in reversed(list(alerts_data))]
    if not alerts_display:
        alerts_display = [html.P("No active alerts", className="text-muted")]

    protocol_event = {
        'timestamp': datetime.now(),
        'protocol': packet['Protocol'],
        'command': packet['Flags'],
        'source_ip': packet['Source'],
        'destination': packet['Destination'],
        'is_anomaly': packet['is_anomaly'] == 1,
        'anomaly_type': 'Suspicious Flag' if packet['is_anomaly'] == 1 else None,
        'response_time': packet.get('response_time', 0) * 1000
    }
    protocol_events.append(protocol_event)

    response_time_data.append(packet['response_time'])
    
    time_points = list(range(len(response_time_data)))
    actual_values = list(response_time_data)
    
    baseline_values = pd.Series(actual_values).rolling(window=10, min_periods=1).mean()

    fig_anomaly = go.Figure()
    fig_anomaly.add_trace(go.Scatter(
        x=time_points, y=baseline_values, mode='lines',
        name='Expected Baseline (Moving Avg)', line=dict(color='green', width=2)
    ))
    fig_anomaly.add_trace(go.Scatter(
        x=time_points, y=actual_values, mode='lines',
        name='Actual Response Time', line=dict(color='cyan', width=2)
    ))
    fig_anomaly.update_layout(
        template='plotly_dark', yaxis=dict(title='Response Time (s)'),
        hovermode='x unified', margin=dict(l=0, r=0, t=40, b=0), height=300,
        title="Real-Time Network Response Time"
    )

    substations = pd.DataFrame({
        'name': ['North', 'East', 'South', 'West', 'Central'],
        'lat': [40.7128, 40.7580, 40.6892, 40.7489, 40.7282],
        'lon': [-74.0060, -73.9855, -74.0445, -73.9680, -73.9942],
        'status': ['Normal', 'Warning', 'Critical', 'Normal', 'Warning'] if len(alerts_data) > 2 else ['Normal'] * 5,
        'load': [75 + random.randint(-10, 10) for _ in range(5)]
    })
    fig_map = px.scatter_mapbox(
        substations, lat='lat', lon='lon', color='status', size='load',
        color_discrete_map={'Normal': '#00ff00', 'Warning': '#ffaa00', 'Critical': '#ff0000'},
        hover_data=['name', 'load'], zoom=10
    )
    fig_map.update_layout(mapbox_style="carto-darkmatter", margin={"r":0,"t":0,"l":0,"b":0})

    metrics_display = html.Div([
        dbc.Row([
            dbc.Col(dbc.Card(dbc.CardBody([
                html.H4(f"{random.randint(75, 95)}%", className="text-center"),
                html.P("Grid Load", className="text-center text-muted")])), width=6),
            dbc.Col(dbc.Card(dbc.CardBody([
                html.H4(f"{len(alerts_data)}", className="text-center text-danger"),
                html.P("Active Threats", className="text-center text-muted")])), width=6),
        ], className="mb-3"),
        dbc.Row([
            dbc.Col(dbc.Card(dbc.CardBody([
                html.H4(f"{confidence_threshold}%", className="text-center"),
                html.P("AI Threshold", className="text-center text-muted")])), width=6),
            dbc.Col(dbc.Card(dbc.CardBody([
                html.H4(f"{int(packet['response_time']*1000)}ms", className="text-center"),
                html.P("Last Ping", className="text-center text-muted")])), width=6),
        ])
    ])

    insights = [dbc.Alert(f"Insight: Monitoring {packet['Protocol']} traffic.", color="info", className="mb-2")]
    if alerts_data:
        insights.append(dbc.Alert("Action: Critical anomalies detected. Review alerts.", color="danger"))

    threat_history = pd.DataFrame({
        'Time': pd.to_datetime(pd.Series([datetime.now() - timedelta(hours=i) for i in range(24, 0, -1)])),
        'Threat Level': [random.randint(10, 30) + (len(alerts_data) * 10) for _ in range(24)]
    })
    fig_threat = px.area(threat_history, x='Time', y='Threat Level', template='plotly_dark', color_discrete_sequence=['red'])
    fig_threat.add_hline(y=confidence_threshold, line_dash="dash", annotation_text="Alert Threshold")

    new_sensor_logs = generate_sensor_logs(threat_level)
    sensor_logs.extend(new_sensor_logs)

    fig_correlation = go.Figure()
    recent_logs = sorted(list(sensor_logs), key=lambda x: x['timestamp'])

    if recent_logs:
        source_map = {'network': 1, 'scada': 2, 'sensor': 3}
        for source_name, y_val in source_map.items():
            source_events = [log for log in recent_logs if log['source'] == source_name]
            if source_events:
                fig_correlation.add_trace(go.Scatter(
                    x=[e['timestamp'] for e in source_events],
                    y=[y_val] * len(source_events),
                    mode='markers', name=source_name.capitalize(),
                    marker=dict(size=12, color=['red' if e['event']['status'] in ['critical', 'suspicious'] else 'green' for e in source_events])
                ))

        correlations = {}
        for log in recent_logs:
            cid = log.get('correlation_id')
            if cid not in correlations: correlations[cid] = []
            correlations[cid].append(log)
        
        for cid, events in correlations.items():
            if len(events) > 1:
                events.sort(key=lambda x: x['timestamp'])
                for i in range(len(events) - 1):
                    y1 = source_map.get(events[i]['source'])
                    y2 = source_map.get(events[i+1]['source'])
                    is_anomaly = events[i]['event']['status'] in ['critical', 'suspicious']
                    fig_correlation.add_trace(go.Scatter(
                        x=[events[i]['timestamp'], events[i+1]['timestamp']],
                        y=[y1, y2], mode='lines', showlegend=False,
                        line=dict(color='yellow' if is_anomaly else 'gray', width=1, dash='dot')
                    ))

    fig_correlation.update_layout(
        title="Event Correlation Timeline (Simulated)",
        yaxis=dict(tickmode='array', tickvals=[1, 2, 3], ticktext=['Network', 'SCADA', 'Sensor'], range=[0.5, 3.5]),
        template='plotly_dark', showlegend=True
    )

    return (alerts_display, fig_map, fig_anomaly, metrics_display, insights, fig_threat, fig_correlation)

@app.callback(
    Output('protocol-content', 'children'),
    [Input('protocol-tabs', 'active_tab'),
     Input('interval-component', 'n_intervals')]
)
def update_protocol_content(active_tab, n):
    if not active_tab:
        active_tab = "protocol-overview"
    
    recent_events = list(protocol_events)
    
    if active_tab == "protocol-overview":
        if recent_events:
            df = pd.DataFrame(recent_events)
            protocol_counts = df.groupby('protocol')['is_anomaly'].value_counts().unstack(fill_value=0)
            protocol_counts.rename(columns={False: 'Normal Traffic', True: 'Anomalous Traffic'}, inplace=True)
            
            if 'Normal Traffic' not in protocol_counts.columns:
                protocol_counts['Normal Traffic'] = 0
            if 'Anomalous Traffic' not in protocol_counts.columns:
                protocol_counts['Anomalous Traffic'] = 0

            protocol_counts = protocol_counts.reset_index()

            protocol_df_long = protocol_counts.melt(
                id_vars='protocol', 
                value_vars=['Normal Traffic', 'Anomalous Traffic'],
                var_name='Traffic Type',
                value_name='Count'
            )
            
            fig_protocol = px.bar(
                protocol_df_long,
                x='protocol',
                y='Count',
                color='Traffic Type',
                title='Protocol Traffic Distribution (from CSV)',
                template='plotly_dark',
                color_discrete_map={'Normal Traffic': '#00ff00', 'Anomalous Traffic': '#ff0000'},
                barmode='group'
            )
            return dcc.Graph(figure=fig_protocol)
        return html.P("No protocol data available", className="text-muted")
    
    elif active_tab == "command-analysis":
        if recent_events:
            table_data = [{
                'Time': e['timestamp'].strftime('%H:%M:%S'),
                'Protocol': e.get('protocol', 'N/A'),
                'Command': e.get('command', 'N/A'),
                'Source': e.get('source_ip', 'N/A'),
                'Destination': e.get('destination', 'N/A'),
                'Response Time': f"{e.get('response_time', 0):.0f}ms",
                'Status': '🔴 Anomaly' if e.get('is_anomaly') else '🟢 Normal'
            } for e in reversed(recent_events[-20:])]
            
            
            table_columns = [
                {'name': 'Time', 'id': 'Time'},
                {'name': 'Protocol', 'id': 'Protocol'},
                {'name': 'Command', 'id': 'Command'},
                {'name': 'Source', 'id': 'Source'},
                {'name': 'Destination', 'id': 'Destination'},
                {'name': 'Response Time', 'id': 'Response Time'},
                {'name': 'Status', 'id': 'Status'},
            ]
            
            return dash_table.DataTable(
                data=table_data,
                columns=table_columns,
                style_cell={'textAlign': 'left', 'backgroundColor': 'rgb(50, 50, 50)', 'color': 'white'},
                style_data_conditional=[{
                    'if': {'filter_query': '{Status} contains "Anomaly"'},
                    'backgroundColor': 'rgb(100, 0, 0)',
                }],
                style_header={'backgroundColor': 'rgb(30, 30, 30)', 'fontWeight': 'bold'},
                page_size=10
            )
        return html.P("No command data available", className="text-muted")
        
    elif active_tab == "anomalous-patterns":
        if recent_events:
            anomaly_events = [e for e in recent_events if e.get('is_anomaly')]
            if anomaly_events:
                return html.Div([
                    dbc.Alert([
                        html.Strong(f"{event.get('protocol', 'N/A')} - {event.get('command', 'N/A')}"),
                        html.Br(),
                        html.Small(f"Source: {event.get('source_ip', 'N/A')} | Time: {event['timestamp'].strftime('%H:%M:%S')}")
                    ], color="danger", className="mb-2")
                    for event in reversed(anomaly_events[-5:])
                ])
            return html.P("No anomalous patterns detected", className="text-muted")
    return html.P("Select a tab", className="text-muted")

if __name__ == '__main__':
    app.run(debug=True, port=8050)


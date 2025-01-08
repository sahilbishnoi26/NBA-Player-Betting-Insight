import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
from datetime import datetime
from bs4 import BeautifulSoup 
import numpy as np
from appfunctions import * 
from AnalyticsFiles.moneylinefunctions import *

# Read the CSV file containing player stats
df = pd.read_csv('appdata/filtered_last_10_games.csv')
df_long = pd.read_csv("appdata/game_logs.csv")
processed_df = pd.read_csv("appdata/ProcessedDFDemo.csv")

# Scrape player odds data
scraped_data = scrape_line()

# Get the list of players and available stats
players = df['PLAYER_NAME'].unique().tolist()
stats = df.columns[1:].tolist()  # Exclude the first column ('PLAYER_NAME')

# Initialize the Dash app
app = dash.Dash(__name__)

font_style={'fontFamily': 'Arial, sans-serif'}

# Define the app layout
app.layout = html.Div([
    html.H1("Real-Time Betting Analysis Dashboard for NBA Players", style={'fontFamily': 'Arial, sans-serif'}),
    html.H2("CSE6242 Project", style={'fontFamily': 'Arial, sans-serif'}), 
    html.Div("Group 42: Sahil Bishnoi, Josh Garretson, Avery Girsky, Oliver Hewett, Hardik Patel, Atticus Rex", style={'fontFamily': 'Arial, sans-serif'}),
    html.Br(style={'height':'50px'}), 
    dcc.Dropdown(
        id="player-dropdown",
        options=[{'label': player, 'value': player} for player in players],
        value='LeBron James',  # Default value is the first player
        clearable=False,
        style={'width': '50%'}
    ),
    dcc.Dropdown(
        id="stat-dropdown",
        options=[{'label': stat, 'value': stat} for stat in stats],
        value=stats[0],  # Default value is the first stat
        clearable=False,
        style={'width': '50%'}
    ),
    html.Br(style={'height':'50px'}),
    html.Div(id='selected-odds'),
    dcc.Graph(id='player-stats-graph'), 
    html.Br(),
    html.Div(id='model-output', style = {'fontFamily':'Arial, sans-serif', 'fontWeight':'bold'}),
    dcc.Graph(id='probability-distribution'),
    html.Div(id='slider-output', style = font_style),
    html.Br(style={'height':'50px'}),
    dcc.Slider(
        id='points-slider',
        min=0,
        max=50,
        step=1,
        value=5,
        marks={i: str(i) for i in range(0, 51, 5)}  # Mark every 5 points on the slider for readability
    ),
    html.Br(style = {'height':'50px'}),
    html.Div(id='expected-over-return', style = font_style ),
    html.Div(id='expected-under-return', style = font_style )
])

@app.callback(
        [Output('points-slider', 'max'),
         Output('points-slider', 'value')],
        [Input('player-dropdown', 'value'),
         Input('stat-dropdown', 'value')]
)

def update_slider(selected_player, stat_selection):
    if stat_selection == "PTS":
        max_val = 50
    elif stat_selection == "AST":
        max_val = 20
    elif stat_selection == "REB":
        max_val = 20 
    elif stat_selection == "STL":
        max_val = 10 
    else:
        max_val = 10 
    try:
        return (max_val,np.ceil(float(scraped_data[stat_selection][selected_player]['over']['line'])) )
    except:
        return (max_val, 1)

    

# Callback to update the over/under odds based on selected player and stat
@app.callback(
    Output('selected-odds', 'children'),
    [Input('player-dropdown', 'value'),
     Input('stat-dropdown', 'value')]
)
def update_odds(selected_player, selected_stat):
    if selected_player in scraped_data[selected_stat]:
        player_odds = scraped_data[selected_stat][selected_player]
        over_line = selected_stat + " Over: \nLine: " + player_odds['over']['line'] + " Odds: " + player_odds['over']['odds'] + "\n\n"
        under_line = selected_stat + " Under: \nLine: " + player_odds['under']['line'] + " Odds: " + player_odds['under']['odds'] + "\n"
    

        return [html.Div(f"Betting Lines for {selected_player}:", style={'fontFamily': 'Arial, sans-serif', 'fontWeight':'bold'}),
                html.Div(over_line, style = font_style), 
                html.Div(under_line, style = font_style)]
    else:
        return html.Div("No odds available for selected player and stat.", style={'fontFamily': 'Arial, sans-serif'})

# Callback to update the graph based on selected player and stat
@app.callback(
    Output('player-stats-graph', 'figure'),
    [Input('player-dropdown', 'value'),
     Input('stat-dropdown', 'value')]
)

def update_graph(selected_player, selected_stat):
    # Filter data for the selected player and stat
    player_data = df[df['PLAYER_NAME'] == selected_player]

    # Create scatter plot
    fig = px.scatter(player_data, x='GAME_DATE', y=selected_stat, title=f'{selected_stat} for {selected_player} over last 10 games')
    fig.update_xaxes(title='Game Date')
    fig.update_yaxes(title=selected_stat)
    fig.update_traces(marker=dict(size=15))  # Adjust the size value as needed

    # Add today's point as a scatter point
    today = datetime.now().strftime('%Y-%m-%d')
    today_point = player_data[player_data['GAME_DATE'] == today][selected_stat]

    # Add a scatter point for today's over line
    if selected_player in scraped_data[selected_stat]:
        over_line = scraped_data[selected_stat][selected_player]['over']['line']
        if over_line is not None:
            fig.add_scatter(x=[today], y=[float(over_line)], mode='markers', marker=dict(color='green', size=15), name=f'Today\'s Over Line')

    fig.add_scatter(x=[today], y=today_point, mode='markers', marker=dict(color='red'), name=f'Today\'s {selected_stat}')

    return fig

# Callback to update the graph based on selected player and stat
@app.callback(
    [Output('model-output', 'children'),
     Output('probability-distribution', 'figure'),
     Output('slider-output', 'children'),
     Output('expected-over-return', 'children'), 
     Output('expected-under-return', 'children')],
    [Input('player-dropdown', 'value'),
     Input('stat-dropdown', 'value'),
     Input('points-slider', 'value')]
)

def update_graph(selected_player, selected_stat, selected_points):
    # Get the prediction 
    prediction = get_prediction(selected_player, selected_stat)
    # Model text 
    model_text = html.Div(f'Lasso Regression Model Prediction: {prediction:.2f} ' + selected_stat)


    hist, bins = get_distribution(processed_df, selected_stat, prediction)
    
    # Determine the color of each bar based on its position relative to the slider value
    colors = ['red' if int(point) <= selected_points else 'green' for point in bins[1:]]
    
    fig = go.Figure(data=[
        go.Bar(x=bins, y=hist, marker_color=colors)
    ])
    
    fig.update_layout(title_text=f'Probability Distribution for {selected_player}', xaxis_title="Points", yaxis_title="Probability")
    
    # Calculate the sum of probabilities for points above the selected threshold
    probabilities_sum = hist[bins[1:] > selected_points].sum()
    output_text = f'Probability of scoring {selected_points} ' + selected_stat + f' or more: {probabilities_sum:.4f}'
    
    # Calculate the expected return
    try:
        proportion = odds_to_proportion(scraped_data[selected_stat][selected_player]['over']['odds'])
        expected_over_return = probabilities_sum*proportion - (1-probabilities_sum)
        proportion = odds_to_proportion(scraped_data[selected_stat][selected_player]['under']['odds'])
        expected_under_return = (1-probabilities_sum)*proportion-probabilities_sum
        expected_over_return_output = f"Expected Over Return (For a $100 Bet): ${100*expected_over_return:.2f}"
        expected_under_return_output = f"Expected Under Return (For a $100 Bet): ${100*expected_under_return:.2f}"
    except:
        expected_over_return_output = ""
        expected_under_return_output = "No Line for this Player!"
    return model_text, fig, output_text, expected_over_return_output, expected_under_return_output

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)


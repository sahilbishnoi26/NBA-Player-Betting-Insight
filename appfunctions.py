import requests
from bs4 import BeautifulSoup 
import pandas as pd
from pprint import pprint 
import pandas as pd  
from tqdm.notebook import tqdm
from AnalyticsFiles.moneylinefunctions import *
tqdm.pandas()
import pickle as p 


headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
}

categories = ['player-points&subcategory=points', 'player-rebounds&subcategory=rebounds', 'player-assists&subcategory=assists', 'player-defense&subcategory=blocks', 'player-defense&subcategory=steals']


def scrape_line():
    sportsbook = []

    for cat in categories:
        url = f'https://sportsbook.draftkings.com/leagues/basketball/nba?category={cat}'
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.text, 'html.parser')

        players_data = []

        for player_wrapper in soup.select('.sportsbook-row-name__wrapper'):
            player_name = player_wrapper.find('span', class_='sportsbook-row-name').text.strip()
            player_data = {
                'player_name': player_name,
                'odds': []
            }

            outcome_cells = player_wrapper.find_parent('th').find_next_siblings('td')
            for cell in outcome_cells:
                line = cell.find('span', class_='sportsbook-outcome-cell__line').text.strip()
                odds = cell.find('span', class_='sportsbook-odds american default-color').text.strip()

                player_data['odds'].append({
                    'line': line,
                    'odds': odds
                })

            players_data.append(player_data)
        sportsbook.append(players_data)

    table = sportsbook

    table_dict = {
        'PTS':{},
        'REB':{},
        'AST':{},
        'BLK':{},
        'STL':{}
    }

    for (i, category) in enumerate(table_dict.keys()):
        for player_dict in table[i]:
            table_dict[category][player_dict['player_name']] = {
                'over':player_dict['odds'][0],
                'under':player_dict['odds'][1]
            }


    return table_dict

def get_prediction(player_name, stat_of_interest):
    df = pd.read_csv("appdata/game_logs.csv")
    df.drop(columns=['Unnamed: 0'], inplace=True)

    player_df = df.query('PLAYER_NAME == "' + player_name + '"')

    # Filter only regular season games
    player_df = player_df.loc[player_df['SEASON_ID'].astype(str).str[0:1] == "2"].reset_index(drop = True)

    # Create binary home variable 
    player_df['Home'] = 0 
    player_df.loc[player_df['MATCHUP'].str.contains('@'), 'Home'] = 1

    # Make WL a numeric variable
    player_df.loc[player_df['WL'] == 'W', 'WL'] = 1
    player_df.loc[player_df['WL'] == 'L', 'WL'] = 0
    player_df['WL'] = player_df['WL'].astype(int)

    # Drop unnecessary columns 
    cols_to_drop = [
        'PLAYER_NAME', 'SEASON',
        'VIDEO_AVAILABLE'
    ]
    player_df = player_df.drop(columns=cols_to_drop)

    # Specify the quantitative columns 
    quant_cols = ['WL', 'MIN', 'FGM', 'FGA', 'FG_PCT', 'FG3M', 'FG3A', 'FG3_PCT', 'FTM',
        'FTA', 'FT_PCT', 'OREB', 'DREB', 'REB', 'AST', 'STL', 'BLK', 'TOV',
        'PF', 'PTS', 'PLUS_MINUS']
    
    lookback = 5

    # Define funcitonality of seasonal processing 
    def helper_func(season):
        # Sort values in chronological order 
        season["GAME_DATE"] = pd.to_datetime(season["GAME_DATE"])
        season = season.sort_values("GAME_DATE")
        # Create an objective column 
        season["OBJ"] = season[stat_of_interest].copy()

        # Creating a column for the number of games
        season['Count'] = season['GAME_DATE'].expanding(1).count().copy()
        
        # Computing moving averages
        season[quant_cols] = season[quant_cols].expanding(1).sum().copy()
        for col in quant_cols:
            season[col] = season[col] / season['Count']
            season[col + "_STD"] = season[col].expanding(1).std().copy()

        season.drop(columns='Count', inplace=True)

        #print(season.head())

        return season 
        


    processed_df = player_df.groupby(by = ["Player_ID", "SEASON_ID"]).progress_apply(helper_func).dropna().reset_index(drop = True)

    with open("appdata/Scaler_" + stat_of_interest + ".pkl", "rb") as infile:
        scaler = p.load(infile)

    X = processed_df.drop(columns = ["SEASON_ID", "Game_ID", "GAME_DATE", "MATCHUP", "OBJ"]).copy()

    X = scaler.transform(X)

    with open("appdata/Lasso_" + stat_of_interest + ".pkl", "rb") as infile:
        model = p.load(infile)

    Yhat = model.predict(X)

    return Yhat.ravel()[-1]

def get_distribution(processed_df, stat_of_interest, pred):
    # Round the prediction
    processed_df = processed_df.copy()
    pred = np.round(pred)

    with open("appdata/Scaler_" + stat_of_interest + ".pkl", "rb") as infile:
        scaler = p.load(infile)

    stats_to_remove = set(["OBJ_PTS", "OBJ_REB", "OBJ_AST", "OBJ_BLK", "OBJ_STL"])
    stats_to_remove.remove("OBJ_" + stat_of_interest)

    processed_df.drop(columns = list(stats_to_remove), inplace=True)

    Y = processed_df["OBJ_" + stat_of_interest].copy().astype("float64")
    X = processed_df.drop(columns = ["SEASON_ID", "Game_ID", "GAME_DATE", "MATCHUP", "OBJ_" + stat_of_interest]).copy()

    X = scaler.transform(X)

    with open("appdata/Lasso_" + stat_of_interest + ".pkl", "rb") as infile:
        model = p.load(infile)

    Yhat = model.predict(X)

    Y_spec = Y[np.round(Yhat) == pred]

    bins = np.arange(np.min(Y_spec), np.max(Y_spec))

    hist, bins = np.histogram(Y_spec, bins = bins, density=True)

    return hist, bins

    
def get_master_df():
    df = pd.read_csv("appdata/game_logs.csv")
    df.drop(columns=['Unnamed: 0'], inplace=True)

    # Query a specific player
    specific_player = False 

    player_df = df.copy()

    # Filter only regular season games
    player_df = player_df.loc[player_df['SEASON_ID'].astype(str).str[0:1] == "2"].reset_index(drop = True)

    # Create binary home variable 
    player_df['Home'] = 0 
    player_df.loc[player_df['MATCHUP'].str.contains('@'), 'Home'] = 1

    # Make WL a numeric variable
    player_df.loc[player_df['WL'] == 'W', 'WL'] = 1
    player_df.loc[player_df['WL'] == 'L', 'WL'] = 0
    player_df['WL'] = player_df['WL'].astype(int)

    # Drop unnecessary columns 
    cols_to_drop = [
        'PLAYER_NAME', 'SEASON',
        'VIDEO_AVAILABLE'
    ]
    player_df = player_df.drop(columns=cols_to_drop)

    stats_of_interest = set(["PTS", "REB", "AST", "BLK", "STL"])
     # How many games to look back for the autoregressive calculations


    # Specify the quantitative columns 
    quant_cols = ['WL', 'MIN', 'FGM', 'FGA', 'FG_PCT', 'FG3M', 'FG3A', 'FG3_PCT', 'FTM',
        'FTA', 'FT_PCT', 'OREB', 'DREB', 'REB', 'AST', 'STL', 'BLK', 'TOV',
        'PF', 'PTS', 'PLUS_MINUS']

    # Define funcitonality of seasonal processing 
    def helper_func(season):
        # Sort values in chronological order 
        season["GAME_DATE"] = pd.to_datetime(season["GAME_DATE"])
        season = season.sort_values("GAME_DATE")
        # Create an objective column 
        for stat in stats_of_interest:
            season["OBJ_" + stat] = season[stat].copy()

        # Creating a column for the number of games
        season['Count'] = season['GAME_DATE'].expanding(1).count().shift(1).copy()
        
        # Computing moving averages
        season[quant_cols] = season[quant_cols].expanding(1).sum().shift(1).copy()
        for col in quant_cols:
            season[col] = season[col] / season['Count']
            season[col + "_STD"] = season[col].expanding(1).std().shift(1).copy()

        season.drop(columns='Count', inplace=True)

        #print(season.head())

        return season 

    processed_df = player_df.groupby(by = ["Player_ID", "SEASON_ID"]).progress_apply(helper_func).dropna().reset_index(drop = True)
    processed_df.to_csv("ProcessedDF.csv", index=False)
    return processed_df

def odds_to_proportion(odds):
    if odds[0] == "−":
        odds = float(odds[1:])
        return 100 / odds
    elif odds[0] == "+":
        odds = float(odds[1:])
        return odds / 100
    else:
        odds = float(odds)
        return odds / 100



import pandas as pd
from nba_api.stats.static import teams
from nba_api.stats.endpoints import leaguegamefinder
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegressionCV, LinearRegression
from sklearn.neighbors import KNeighborsClassifier
import seaborn as sns
import requests, json
from bs4 import BeautifulSoup

# Function for loading historical NBA Game Data starting from 1994-1995 Season
def load_data(load_new_games=True, start_date='10/01/1994'):
    print("Loading Games...")
    # Checks if it should load new games from NBA Api or from file
    if load_new_games:
        # Gets the dataframe of games
        df = get_games(start_date = start_date)    
        # Saves it to a raw CSV file for easy access if you want to run multiple times    
        df.to_csv("data/RawDF.csv", index=False)
        # Loads from the CSV file for caching reasons it's way faster
        df = pd.read_csv("data/RawDF.csv")
    else:
        df = pd.read_csv("data/RawDF.csv")
    
    print("Loaded Games!\n")

    return df

# Getting games from NBA API
def get_games(start_date = '10/01/1994'):
    # Create empty dataframe
    games_df = pd.DataFrame()

    # Get a list of NBA Teams
    nba_teams = teams.get_teams()

    # Iterate through the list of teams
    for team in tqdm(nba_teams):
        team_id = team['id']
        # Query for games with this specific team
        gamefinder = leaguegamefinder.LeagueGameFinder(date_from_nullable=start_date,team_id_nullable=team_id, timeout=60)
        # The first DataFrame of those returned is what we want.
        games = gamefinder.get_data_frames()[0]
        
        # Concatenate the games to the games_df object
        games_df = pd.concat((games_df, games))
    
    # Return & Reset index
    return games_df.reset_index(drop=True)

# Function for scraping the Odds from Draftkings Website
def get_todays_odds():
    # Web Address for NBA Lines
    url = "https://sportsbook.draftkings.com/leagues/basketball/nba"
    # Get the response 
    response = requests.get(url)

    # Check if response was received
    if response.status_code == 200:
        # Parse HTML
        soup = BeautifulSoup(response.text, 'html.parser')
        # Find the table body elements
        tbody = soup.find('tbody')
        # Initialize data dictionary
        odds_data = {}

        # Load the dictionary that has the name to abbreviation matching file
        with open("data/DraftkingsNameMatcher.json","r") as infile:
            names_matcher = json.load(infile)

        # Assuming each row is within a <tr> tag and each cell within a <td> tag
        for (name, data) in zip(tbody.find_all('th'), tbody.find_all('tr')):
            # Extract the text from each cell in the row and add to the row_data list
            name = [cell.text.strip() for cell in name.find_all('div')][-1]
            
            # Convert the name to a 3-letter abbreviation (e.g. Phoenix Suns to PHX)
            name = names_matcher[name]
            
            # Get the odds 
            moneyline_odds = [cell.text.strip() for cell in data.find_all('td')][-1]
            # Replace the weird unicode - glitch for negative odds 
            odds_data[name] = moneyline_odds.replace("−","-")

        # Create new data dictionary        
        full_odds_data = {}
        # Iterate through each team and odds to append more data
        for (i, (team, odds)) in enumerate(odds_data.items()):
            # If it's the second team, it's the home team 
            if (i+1)%2 == 0:
                full_odds_data[team] = {
                    "odds":odds,
                    "home_team":team,
                    "away_team":list(odds_data.keys())[i-1], # Gets the away team 
                    "profit":odds_to_profit(odds) # Converts odds to profit proportion if won
                }
            else:
                full_odds_data[team] = {
                    "odds":odds,
                    "home_team":list(odds_data.keys())[i+1],
                    "away_team":team,
                    "profit":odds_to_profit(odds)
                }
        
        return full_odds_data
    else:
        raise ValueError("No Response Received from Draftkings!")

# Function to convert the odds to an actual profit proportion
# e.g. if odds for a team are +180, then profit is 1.8 
def odds_to_profit(odds):
    if odds[0] == "+":
        odds = float(odds[1:])
        return odds / 100
    else:
        odds = float(odds[1:])
        return 100 / odds

# Function to produce a calibration plot of the outputted probabilities of a 
# Machine Learning model compared with the true probabilities         
def prob_plot(Y_true, Y_pred, bins=25):
    # Create empty array of probabilities
    probs = np.zeros((bins))
    # Create empty array of number of samples within each bin
    samples = np.zeros_like(probs)
    # Create array of standard errors 
    errors = np.zeros_like(probs)
    # Create a vector of bounds for each bin 
    bound_vec = np.linspace(0, 1, bins + 1)
    bounds = zip(bound_vec[0:-1], bound_vec[1:])
    
    # Iterate through the bounds to focus on each bin 
    for (i, (lower, upper)) in enumerate(bounds):
        # Count the number of predictions fall within this probability bin
        total = len((Y_pred[(Y_pred >= lower) & (Y_pred < upper)]))
        # Count the number of correct predictions 
        correct = np.sum(Y_true[(Y_pred >= lower) & (Y_pred < upper)])
        if total > 0:
            # Compute the proportion the model got correct 
            probs[i] = correct / total
            # Compute the proportion of the total samples this bin gets
            samples[i] = total / len(Y_pred)
            # Compute the margin of error (98.5%)
            errors[i] = 2.5 * np.sqrt(probs[i]*(1 - probs[i]) / total)
        else:
            # If there are no observations within the bin, then 
            # It's just set to the average of the bounds 
            probs[i] = 0.5*(upper + lower)
            # Computing the sample proportion
            samples[i] = total / len(Y_pred)
            # Just setting the error to +- 0.5 
            errors[i] = 0.5
    
    return (bound_vec, probs, samples, errors)

# Filtering the dataset to have regular season data
# rows, etc. 
def filter_dataset(df):
    # Creating a column to indicate if the home team is playing
    df['Home'] = 1
    # Creating a list of indices to include 
    inds_to_include = []
    # Iterating through the rows
    for (index, row) in tqdm(df.iterrows()):
        # Getting rid of the first character on the SEASON_ID
        row.SEASON_ID = str(row.SEASON_ID)[1:]
        df.loc[index, 'SEASON_ID'] = row.SEASON_ID
        # Setting home games
        if '@' in row.MATCHUP:
            df.loc[index, 'Home'] = 0
        
        # Setting the WL column to be binary instead of W or L
        if row.WL == 'W':
            df.loc[index, 'WL'] = 1
        else:
            df.loc[index, 'WL'] = 0

        # Only including the in-season months
        month = int(row.GAME_DATE.split('-')[1])
        day = int(row.GAME_DATE.split('-')[2])
        if month < 7 or month > 9:
            if month == 10 and day > 20:
                inds_to_include.append(index)
            elif month > 10 or month < 7:
                inds_to_include.append(index)

    # Dropping necessary columns and resetting indices
    df = df.loc[inds_to_include, :]
    df.drop(columns = ['TEAM_NAME', 'MATCHUP'], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

quant_cols = ['MIN', 'PTS', 'FGM', 'FGA', 'FG_PCT', 'FG3M', 'FG3A', 'FG3_PCT', 'FTM', 'FTA', 'FT_PCT', 'OREB', 'DREB', 'REB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PLUS_MINUS']

# Function to calculate win and loss streaks 
# for the group function in the running average
def calculate_streaks(series):
        # Initialize streak counters
        WStreak, LStreak = 0, 0
        streaks = []
        for result in series:
            if result == 1:
                WStreak += 1
                LStreak = 0
            elif result == 0:
                LStreak += 1
                WStreak = 0
            else:
                WStreak, LStreak = 0, 0  # Reset streaks for non-W/L results
            streaks.append((WStreak, LStreak))
        return streaks

# Grouping the dataframe by teams and seasons and 
# computing running averages 
def groupby_team_season(df):
    def helper_func(group):
        # Sort in ascending order by date
        group['GAME_DATE'] = pd.to_datetime(group['GAME_DATE'])
        group.sort_values(by='GAME_DATE', ascending=True, inplace=True)
        
        # Replace any columns that have NA's with 0 
        group.fillna(0, inplace=True)
        
        # Create running averages for quantitative columns
        group[quant_cols] = group[quant_cols].expanding(1).sum().shift(1).copy()
        group['Count'] = group['GAME_DATE'].expanding(1).count().shift(1).copy()
        for col in quant_cols:
            group[col] = group[col] / group['Count']
        
        # Convert WL column into a win percentage
        group['WIN_PCT'] = group['WL'].expanding(1).sum().shift(1).copy() / group['Count']
        
        # Calculate win/loss streaks
        streaks = calculate_streaks(group['WL'])
        group['WStreak'], group['LStreak'] = zip(*streaks)
        group[['WStreak', 'LStreak']] = group[['WStreak', 'LStreak']].shift(1)
        group[['WStreak', 'LStreak']].fillna(0, inplace=True)

        # Compute Home Win Percentage
        home_games_mask = group['Home'] == 1
        home_wins = group['WL'][home_games_mask].expanding().apply(lambda x: (x == 1).sum())
        total_home_games = home_games_mask.expanding().sum()
        group['HomeWinPct'] = home_wins / total_home_games
        group['HomeWinPct'].fillna(method='ffill', inplace=True)  # Set away game values correctly
        group['HomeWinPct'] = group['HomeWinPct'].shift(1).fillna(0)

        # Remove the Count column
        group.drop(columns='Count', inplace=True)

        return group

    # Sort by unique SEASON_ID and TEAM_ID, apply the helper_func(), drop all the NA's and reset the indices
    running_totals = df.groupby(['SEASON_ID', 'TEAM_ID']).apply(helper_func).dropna().reset_index(drop=True)
    return running_totals

def match_opponents_optimized(running_totals):
    # Drop unnecessary columns just once
    reduced_totals = running_totals.drop(columns=['SEASON_ID', 'TEAM_ABBREVIATION', 'GAME_DATE'])
    
    # Create two DataFrames, one for each team in a game, and shift column names for the second team
    team_1 = reduced_totals.copy()
    team_2 = reduced_totals.add_suffix('_y').drop(columns=['WL_y', 'Home_y'])

    # Merge these DataFrames based on the GAME_ID, ensuring different teams are matched
    merged_df = pd.merge(team_1, team_2, left_on=['GAME_ID'], right_on=['GAME_ID_y'])
    
    # Filter out rows where the team IDs are the same, as we only want matchups
    match_df = merged_df[merged_df['TEAM_ID'] != merged_df['TEAM_ID_y']]

    return match_df

# Preprocessing the training data from the dataframe of matchups 
def preprocess_training(match_df, test_size=0.20, random_state = 420):
    # drop the unnecessary identifying information
    X = match_df.drop(columns=['TEAM_ID', 'TEAM_ID_y', 'GAME_ID', 'GAME_ID_y']).dropna().reset_index(drop=True).apply(pd.to_numeric)
    # Produce the outputs 
    Y = X['WL'].copy()
    # Drop the Win/Loss column from the input data 
    X.drop(columns=['WL'], inplace=True)

    # Create training and validation datasets 
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state = random_state)

    # Scale the input/output data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return [X_train, X_test, Y_train, Y_test, list(X.columns), scaler]

# Training an ensemble of machine learning models 
def train_models(X_train, Y_train, mlp=True, logit=True, knn=True, rf=True, gb = True):
    models = []
    # Define models 
    if mlp:
        mlp = MLPClassifier((30, 15, 15, 15), activation='tanh',solver='sgd', max_iter=750, warm_start=True, alpha=5e-1, verbose=True, tol=1e-8)
        models.append(mlp)
    if logit:
        logit = LogisticRegressionCV()
        models.append(logit)
    if knn:
        knn = KNeighborsClassifier(n_neighbors=150)
        models.append(knn)
    if rf:
        rf = RandomForestClassifier(max_depth=5, n_estimators=100, n_jobs=-1)
        models.append(rf)
    if gb:
        gb = GradientBoostingClassifier()
        models.append(gb)

    # Fit each models to the training data
    for (i, model) in tqdm(enumerate(models)):
        #model = CalibratedClassifierCV(model, method = "isotonic")
        model.fit(X_train, Y_train)
        models[i] = model

    # Assign weights based on ROC-AUC Score or Accuracy
    weights = []
    [weights.append(roc_auc_score(Y_train, model.predict(X_train))) for model in models]

    # Normalize the weights
    weights = [weight / sum(weights) for weight in weights]

    return [models, weights]

# A class of ensembles of machine learning models 
class Ensemble:
    # Initialize the model with the list of models and weighing of each model
    def __init__(self, models, weights):
        self.models = models
        self.weights = weights 
    
    # Function for predicting outcomes based on some input 
    def predict(self, X):
        return np.round(np.array(sum([weight * model.predict(X) for (weight, model) in zip(self.weights, self.models)])))

    # Function for predicting probabilities given an input X
    def predict_proba(self, X):
        return np.array(sum([weight * model.predict_proba(X) for (weight, model) in zip(self.weights, self.models)]))
    
    # Predicting a confidence interval based on some input X
    def predict_CI(self, X, axis=1):
        preds = np.zeros((X.shape[0], len(self.models)))
        for j in range(len(self.models)):
            preds[:,j] = self.models[j].predict_proba(X)[:,axis]
        
        means = np.mean(preds, axis=1)
        maxs = np.max(preds, axis=1)
        mins = np.min(preds, axis=1)

        return maxs, mins, means


# Function for evaluating a machine learning model with training/validation data 
def evaluate_model(model, X_train, X_test, Y_train, Y_test):
    # Print the statistics
    print("Train Accuracy: %.3f %%" % (100 * accuracy_score(Y_train, model.predict(X_train))))
    print("Test  Accuracy: %.3f %%" % (100 * accuracy_score(Y_test, model.predict(X_test))))
    print(confusion_matrix(Y_test, model.predict(X_test)))
    print("Train ROC Score: %.4f" % (roc_auc_score(Y_train, model.predict_proba(X_train)[:,1])))
    print("Test ROC Score: %.4f" % (roc_auc_score(Y_test, model.predict_proba(X_test)[:,1]))) if len(Y_test) > 50 else print("")

# Function to create a coefficient plot from a 
# logistic regression to evaluate feature importances 
def coefficient_plot(logit, feature_names ):
    # Get the importance scores from the coefficients 
    importance_scores = logit.coef_[0]

    # Create a dataframe from the feature names and importance scores
    feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importance_scores}).head(45)

    # Sort features by importance in descending order
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

    # Create a bar plot from seaborn
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=feature_importance_df)
    plt.xlabel('Feature Coefficient')
    plt.ylabel('Features')
    plt.title('Stat Contribution to Probability of Team Winning')
    plt.show()

# Function to create a calibration plot given the 
# test/validation data and the model which corrects the probabilities 
# outputted by the model 
def calibration_plot(model, X_train, Y_train, X_test, Y_test):
    # Indicate a specific number of bins 
    n_bins = 18
    # Get the data from the probability plot function 
    (bounds, probs, samples, errors) = prob_plot(Y_train, model.predict_proba(X_train)[:,1], bins=n_bins)

    # Compute the bounds for the X-Axis 
    x_bounds = 0.5*(bounds[1:]+bounds[0:-1]).copy()
    y_probs = probs.copy()

    # Create a polynomial features object from sklearn and transform the input data
    poly = PolynomialFeatures(9)
    x_bounds_poly = poly.fit_transform(x_bounds.reshape(-1,1))

    # Fit a linear regression to the polynomial features 
    linear_model = LinearRegression()
    linear_model.fit(x_bounds_poly, y_probs)

    # Create a conversion function to convert raw probabilities to calibrated 
    # probabilities using the polynomial regression 
    def conversion_func(x, linear_model, poly):
        x = poly.transform(x.reshape(-1,1))
        return linear_model.predict(x)

    # Reproducing the plots for the test data 
    x_test = np.linspace(0,1,100)
    x_test_poly = poly.transform(x_test.reshape(-1,1))
    y_test = linear_model.predict(x_test_poly)

    plt.figure(figsize=(10,5))
    plt.scatter(x_bounds, y_probs, label = "Raw Probabilities")
    plt.plot(x_test, y_test, label = "Polynomial Approximation")
    plt.scatter(0.5*(bounds[1:]+bounds[0:-1]), samples / np.max(samples), label = "Sample Size Proportions")
    plt.errorbar(0.5*(bounds[1:]+bounds[0:-1]), probs, yerr=errors, ecolor='black', capsize=3)
    plt.plot([0,1], [0,1], label = "Ideal Relationship")
    plt.grid()
    plt.legend()
    plt.xlabel("Probability Predicted Won")
    plt.ylabel("Actual Probability Won")
    plt.title("Calibration Curve with 98.75% Confidence Intervals")

    plt.show()

    plt.figure(figsize=(10,5))
    (bounds, probs, samples, errors) = prob_plot(Y_test, conversion_func(model.predict_proba(X_test)[:,1], linear_model, poly), bins=n_bins)
    plt.scatter(x_bounds, probs, c='red', label = "Corrected Points")
    plt.errorbar(x_bounds, probs, yerr=errors, ecolor='black', capsize=3)
    (bounds, probs, samples, errors) = prob_plot(Y_test, model.predict_proba(X_test)[:,1], bins=n_bins)
    plt.scatter(x_bounds, probs, c='green', label="Uncorrected Points")
    plt.errorbar(x_bounds, probs, yerr=errors, ecolor='blue', capsize=3)
    plt.plot([0,1], [0,1])
    plt.grid()
    plt.xlabel("Probability Predicted Won")
    plt.ylabel("Actual Probability Won")
    plt.title("Calibration Curve with Calibrated Probabilities (98.75 Confidence Intervals)")
    plt.legend()

    plt.show()

    # Returns the conversion function, linear_model and the polynomial features object 
    return [conversion_func, linear_model, poly]

# Gets the test dataframe for live predictions (doesn't shift down data)
def get_test_df(df):
    def helper_func_test(group):
        # Sort in ascending order by date
        group['GAME_DATE'] = pd.to_datetime(group['GAME_DATE'])
        group.sort_values(by='GAME_DATE', ascending=True, inplace=True)
        
        # Replace any columns that have NA's with 0 
        group.fillna(0, inplace=True)
        
        # Create running averages for quantitative columns
        group[quant_cols] = group[quant_cols].expanding(1).sum().copy()
        group['Count'] = group['GAME_DATE'].expanding(1).count().copy()
        for col in quant_cols:
            group[col] = group[col] / group['Count']
        
        # Convert WL column into a win percentage
        group['WIN_PCT'] = group['WL'].expanding(1).sum().copy() / group['Count']
        
        # Calculate win/loss streaks
        streaks = calculate_streaks(group['WL'])
        group['WStreak'], group['LStreak'] = zip(*streaks)
        #group[['WStreak', 'LStreak']] = group[['WStreak', 'LStreak']].shift(1)
        #group[['WStreak', 'LStreak']].fillna(0, inplace=True)

        # Compute Home Win Percentage
        home_games_mask = group['Home'] == 1
        home_wins = group['WL'][home_games_mask].expanding().apply(lambda x: (x == 1).sum())
        total_home_games = home_games_mask.expanding().sum()
        group['HomeWinPct'] = home_wins / total_home_games
        group['HomeWinPct'].fillna(method='ffill', inplace=True)  # Set away game values correctly
        #group['HomeWinPct'] = group['HomeWinPct'].shift(1).fillna(0)

        # Remove the Count column
        group.drop(columns='Count', inplace=True)



        return group

    # Sort by unique SEASON_ID and TEAM_ID, apply the helper_func(), drop all the NA's and reset the indices
    test_df = df.groupby(['SEASON_ID', 'TEAM_ID']).apply(helper_func_test).dropna().reset_index(drop=True)
    return test_df[test_df['SEASON_ID'] == '2023'].copy()

# Makes a prediction for a specific matchup 
def make_prediction(home_team, away_team, test_df, scaler, model, conversion_func, linear_model, poly, home_odds, away_odds, ensemble=False):
    team1 = test_df[test_df['TEAM_ABBREVIATION'] == home_team].tail(1).copy() # Home Team
    team2 = test_df[test_df['TEAM_ABBREVIATION'] == away_team].tail(1).copy()

    for col in team1.columns:
        if '_y' in col:
            team1.drop(columns = [col], inplace=True)
            team2.drop(columns = [col], inplace=True)

    # Set team1 as the home team 
    team1["Home"] = 1
    pd.concat((team1, team2)).head()

    final_df = pd.DataFrame()
    # Drop the unnecessary/redundant columns
    team2.drop(columns=['SEASON_ID', 'TEAM_ID', 'TEAM_ABBREVIATION', 'GAME_ID', 'GAME_DATE', 'WL', 'Home'], inplace=True)
    # Reset the index
    team1.reset_index(inplace=True, drop=True)
    team2.reset_index(inplace=True, drop=True)
    # Create new columns with suffix _y for the other team
    new_cols = [col + "_y" for col in list(team2.columns)]
    # Save new columns as the other team's columns
    team2.columns = new_cols
    # Combine this team and other team data
    for col in list(team2.columns):
        team1[col] = team2[col]
    # Concatenate the new dataframe with this one 
    final_df = pd.concat((final_df, team1))

    # Drop unnecessary columns and scale data 
    X_test_final = final_df.drop(columns=['SEASON_ID', 'TEAM_ID', 'TEAM_ABBREVIATION', 'GAME_ID', 'GAME_DATE', 'WL']).dropna().reset_index(drop=True).apply(pd.to_numeric)
    X_test_final = scaler.transform(X_test_final)

    # Get the probability of winning for each team 
    p1 = conversion_func(model.predict_proba(X_test_final)[0][1], linear_model, poly)
    p2 = conversion_func(model.predict_proba(X_test_final)[0][0], linear_model, poly)
    lb_prop = 0.05 # Lower bound for probability prediction e.g. worst case scenario

    # Saves the data in a dictionary and returns the dictionary 
    df_dict = {
        "Home Team":home_team,
        "Away Team":away_team,
        "Home Odds":int(home_odds),
        "Away Odds":int(away_odds),
        "Home Prob":p1,
        "Away Prob":p2,
        "Home LB Return":100*odds_to_profit(home_odds)*(p1-lb_prop) - 100 * (1 - p1+lb_prop),
        "Away LB Return":100 * odds_to_profit(away_odds) * (p2-lb_prop) - 100 * (1 - p1+lb_prop)
    }

    return(pd.DataFrame(df_dict))

    
    


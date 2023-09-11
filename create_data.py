import pandas as pd
import numpy as np
import glob
import os

def load_data():

    # Novice
    path_central_novice = './CentralR-N'
    all_files_central_novice = sorted(glob.glob(os.path.join(path_central_novice, "*.json")))

    central = []
    for f in all_files_central_novice:
        df = pd.read_json(f)
        df['logDate'] = f[25:35]
        df['logDate'] = pd.to_datetime(df['logDate'], format='%Y-%m-%d')
        central.append(df)
        
    central_novice = pd.concat(central, ignore_index=True)
    central_novice['region'] = 'Central'

    path_east_novice = './EastR-N'
    all_files_east_novice = sorted(glob.glob(os.path.join(path_east_novice, "*.json")))

    east = []
    for f in all_files_east_novice:
        df = pd.read_json(f)
        df['logDate'] = f[22:32]
        df['logDate'] = pd.to_datetime(df['logDate'], format='%Y-%m-%d')
        east.append(df)
        
    east_novice = pd.concat(east, ignore_index=True)
    east_novice['region'] = 'East'

    novice = pd.concat(([central_novice, east_novice]), ignore_index=True)
    novice['rankedMode'] = 'Novice'

    # Expert
    path_central_expert = './CentralR-E'
    all_files_central_expert = sorted(glob.glob(os.path.join(path_central_expert, "*.json")))

    central = []
    for f in all_files_central_expert:
        df = pd.read_json(f)
        df['logDate'] = f[25:35]
        df['logDate'] = pd.to_datetime(df['logDate'], format='%Y-%m-%d')
        central.append(df)
        
    central_expert = pd.concat(central, ignore_index=True)
    central_expert['region'] = 'Central'

    path_east_expert = './EastR-E'
    all_files_east_expert = sorted(glob.glob(os.path.join(path_east_expert, "*.json")))

    east = []
    for f in all_files_east_expert:
        df = pd.read_json(f)
        df['logDate'] = f[22:32]
        df['logDate'] = pd.to_datetime(df['logDate'], format='%Y-%m-%d')
        east.append(df)
        
    east_expert = pd.concat(east, ignore_index=True)
    east_expert['region'] = 'East'

    expert = pd.concat(([central_expert, east_expert]), ignore_index=True)
    expert['rankedMode'] = 'Expert'

    expert = expert.replace('Unrated', 0)
    expert = expert.astype({'difficulty': 'float'})

    X = pd.concat(([novice, expert]), ignore_index=True)
    X = X.drop(['annId', 'urls', 'siteIds', 'animeScore', 'selfAnswer', 'fromList', 'gameMode', 'correct'], axis=1)
    return X

def find_n_correct(players):
    i = 0
    for player in players:
        if player['correct'] == True:
            i += 1
    return i

def find_gt(players):
    guess_times = [x['guessTime'] for x in players if 'guessTime' in x]
    return np.array(guess_times).mean()

def find_ng(players):
    no_guess = np.zeros(len(players))
    for i, x in enumerate(players):
        val = 0
        if x['active'] == True:
            if 'guessTime' in x:
                val = 0
            else:
                val = 1
        else:
            continue
        no_guess[i] = val
    return no_guess.sum()

def find_scores(players):
    scores = np.zeros(len(players))
    for i, x in enumerate(players):
        scores[i] = x['score']
    return scores

def find_jp(names):
    if 'romaji' in names:
        name = names['romaji']
    else:
        raise Exception('No jp name')
    return name

def add_index(players, index):
    for player in players:
        player['songIndex'] = index
    return players

def flatten(l):
    return [item for sublist in l for item in sublist]

def create_features(X):
    X['anime_jp'] = X['anime'].apply(find_jp)
    X['avgGuessTime'] = X['players'].apply(find_gt)
    X['n_correctGuess'] = X['players'].apply(find_n_correct)
    X['n_noGuess'] = X['players'].apply(find_ng)
    X['p_noGuess'] = X['n_noGuess']/X['activePlayers'] *100
    X['p_correctGuess'] = X['n_correctGuess']/X['activePlayers']*100
    X['type_noNumber'] = X['type'].str.split().str[0]
    X['samplePercent'] = X['startSample']/X['videoLength']
    X['AiredDate'] = X['vintage'].replace({'Winter': '1', 'Spring': '4', 'Summer': '7', 'Fall': '10'}, regex=True)
    X['AiredDate'] = pd.to_datetime(X['AiredDate'], format='%m %Y')
    X = X.drop(['players', 'altAnswers', 'startSample', 'videoLength', 'vintage', 'type', 'songIndex', 'players_id', 'n_correctGuess', 'n_noGuess', 'anime'], axis=1)

    return X

def create_last(X):
    last_song = pd.DataFrame(X.loc[((X['songNumber'] == 45) & (X['rankedMode'] == 'Novice')) | (X['songNumber'] == 85)])
    last_song['Scores'] = last_song['players'].apply(find_scores)
    return last_song

def create_players(X):
    X['songIndex'] = np.arange(X.shape[0])
    X['players_id'] = X.apply(lambda x: add_index(x['players'], x['songIndex']), axis=1)
    players = pd.DataFrame(flatten(X['players_id'].to_list()))
    players = players.loc[(players['active'] == True) | (players['guessTime'].notna())]
    players = players.drop(['answer','correctGuesses', 'active', 'positionSlot'], axis=1).astype({'score':np.uint8,
                                                                                 'position':np.uint16,
                                                                                 'songIndex':np.uint32,
                                                                                 'guessTime':np.float32}).set_index('songIndex')

    return players
    
if __name__ == '__main__':
    X = load_data()
    Y = create_last(X)
    Z = create_players(X)
    
    X = create_features(X)
    X.to_parquet('./data/data.parquet')
    Y.to_parquet('./data/last.parquet')
    Z.to_parquet('./data/players.parquet')

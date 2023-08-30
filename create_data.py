import pandas as pd
import numpy as np
import glob
import os

def load_data():
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
    X = novice.drop(['annId', 'urls', 'siteIds', 'animeScore', 'selfAnswer', 'fromList', 'gameMode', 'correct'], axis=1)
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
    X = X.drop(['players', 'altAnswers', 'startSample', 'videoLength', 'vintage', 'type'], axis=1)

    return X

def create_last(X):
    last_song = pd.DataFrame(X.loc[X['songNumber'] == 45]).reset_index(drop=True)
    last_song['Scores'] = last_song['players'].apply(find_scores)
    return last_song
    
if __name__ == '__main__':
    X = load_data()
    Y = create_last(X)
    
    X = create_features(X)
    X.to_parquet('data.parquet')
    Y.to_parquet('last.parquet')
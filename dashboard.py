from dash import Dash, html, dcc, callback, Output, Input
import plotly.express as px
import pandas as pd
import numpy as np
import seaborn as sns
import dash_bootstrap_components as dbc

import glob
import os

# Load files

path_central_novice = r'C:\Users\imnku\Documents\Data Analysis\AMQ\CentralR-N'
all_files_central_novice = glob.glob(os.path.join(path_central_novice, "*.json"))

central_novice = pd.concat((pd.read_json(f) for f in all_files_central_novice), ignore_index=True)

path_east_novice = r'C:\Users\imnku\Documents\Data Analysis\AMQ\EastR-N'
all_files_east_novice = glob.glob(os.path.join(path_east_novice, "*.json"))

east_novice = pd.concat((pd.read_json(f) for f in all_files_east_novice), ignore_index=True)

novice = pd.concat(([central_novice, east_novice]), ignore_index=True)
X = novice.drop(['annId', 'urls', 'siteIds', 'animeScore', 'altAnswers', 'selfAnswer', 'fromList', 'gameMode', 'correct'], axis=1).copy()

# Create data

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

def find_cgt(players):
    guess_times = np.array([x['guessTime'] for x in players if ('guessTime' in x and x['correct'] == True)])
    return np.array(guess_times).mean()

def find_igt(players):
    guess_times = np.array([x['guessTime'] for x in players if ('guessTime' in x and x['correct'] == False)])
    return np.array(guess_times).mean()

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

X['avgGuessTime'] = X['players'].apply(find_gt)
X['n_correctGuess'] = X['players'].apply(find_n_correct)
X['n_noGuess'] = X['players'].apply(find_ng)
X['p_noGuess'] = X['n_noGuess']/X['activePlayers'] *100
X['p_correctGuess'] = X['n_correctGuess']/X['activePlayers']*100
X['avgGuessTime_c'] = X['players'].apply(find_cgt)
X['avgGuessTime_i'] = X['players'].apply(find_igt)
X['type_noNumber'] = X['type'].str.split().str[0]
X['samplePercent'] = X['startSample']/X['videoLength']
X['AiredDate'] = X['vintage'].replace({'Winter': '1', 'Spring': '4', 'Summer': '7', 'Fall': '10'}, regex=True)
X['AiredDate'] = pd.to_datetime(X['AiredDate'], format='%m %Y')
X['anime_jp'] = X['anime'].apply(find_jp)

n_games = len(all_files_central_novice) + len(all_files_east_novice)
songs_played = X.shape[0]
guess_rate = X['p_correctGuess'].mean()
guess_time = X['avgGuessTime'].mean()
no_guess_rate = X['p_noGuess'].mean()
diff = X['difficulty'].mean()

op = X.loc[X['type_noNumber'] == 'Opening']
op_played = op.shape[0]
op_rate = op['p_correctGuess'].mean()
op_diff = op['difficulty'].mean()

ed = X.loc[X['type_noNumber'] == 'Ending']
ed_played = ed.shape[0]
ed_rate = ed['p_correctGuess'].mean()
ed_diff = ed['difficulty'].mean()

ins = X.loc[X['type_noNumber'] == 'Insert']
in_played = ins.shape[0]
in_rate = ins['p_correctGuess'].mean()
in_diff = ins['difficulty'].mean()

last_song = pd.DataFrame(X.loc[X['songNumber'] == 45]).reset_index(drop=True)
last_song['Scores'] = last_song['players'].apply(find_scores)

finalScores = np.concatenate(last_song['Scores'].values).astype(int)
end_mean = finalScores.mean()
end_std = finalScores.std()
end_max = finalScores.max()

# Data names

lbs = {'difficulty':'Difficulty',
      'artist':'Artist',
      'name':'Song',
      'p_correctGuess':'Correct percent',
      'p_noGuess':'No guess percent',
      'anime_jp':'Anime',
      'avgGuessTime':'Average guess time',
      'AiredDate': 'Aired date',
      'avgGuessTime_c':'Correct avg. guess time',
      'avgGuessTime_i':'Incorrect avg. guess time',
      'type_noNumber':'Song type'}

hover_dt = {'artist':True,
           'name':True,
           'anime_jp':True,
           'avgGuessTime':True,
           'difficulty':':.1f',
           'p_noGuess':':.1f',
           'p_correctGuess':':.1f',
           'avgGuessTime':':.1f',
           'avgGuessTime_c':':.1f',
           'avgGuessTime_i':':.1f',}

# Dashboard

app = Dash(__name__, external_stylesheets=[dbc.themes.SANDSTONE])

def app_description():
    return html.Div(
        id='app-description',
        children=[
            html.H2('Ranked Stats'),
        ]
    )

def difficulty_correct_scatter():
    return html.Div(
        id='diff-correct',
        children=[
            dcc.Graph(figure=px.scatter(
        X,
        x='difficulty',
        y='p_correctGuess',
        color='type_noNumber',
        marginal_x='histogram', 
        marginal_y='histogram',
        hover_data=hover_dt,
        labels=lbs,
        trendline=None,
        template='plotly_white',
        range_x=[0, 100],
        range_y=[0, 100],
        opacity=0.5
        ))
        ]
    )

def types_table():
    return html.Div(
        id='types-table',
        children=dbc.Table(
            bordered=True,
            striped=True,
            children=[
                html.Thead(html.Tr(children=[html.Th(''), html.Th('Openings'), html.Th('Endings'), html.Th('Inserts')])),
                html.Tbody([
                    html.Tr(children=[html.Th('Songs played'), html.Td(op_played), html.Td(ed_played), html.Td(in_played)]),
                    html.Tr(children=[html.Th('Correct guess rate'), html.Td('{:.2f}'.format(op_rate)), html.Td('{:.2f}'.format(ed_rate)), html.Td('{:.2f}'.format(in_rate))]),
                    html.Tr(children=[html.Th('Average difficulty'), html.Td('{:.2f}'.format(op_diff)), html.Td('{:.2f}'.format(ed_diff)), html.Td('{:.2f}'.format(in_diff))])]),
                ]
        )
    )

def generic_table():
    return html.Div(
        id='generic-table',
        children=dbc.Table(
            bordered=True,
            striped=True,
            children=[
                html.Thead(html.Tr([
                    html.Th('Games recorded'), html.Th('Songs recorded'), html.Th('Correct guess rate'), html.Th('Average difficulty'), html.Th('Average guess time'), html.Th('No guess rate')
                    ])),
                html.Tbody(html.Tr([
                    html.Td(n_games), html.Td(songs_played), html.Td('{:.2f}'.format(guess_rate)), html.Td('{:.2f}'.format(diff)), html.Td('{:.2f}'.format(guess_time)), html.Td('{:.2f}'.format(no_guess_rate))
                ]))
            ]
        )
    )

def final_table():
    return html.Div(
        id='final-table',
        children=dbc.Table(
            bordered=True,
            striped=True,
            children=[
                html.Thead(html.Tr([
                    html.Th(''), html.Th('Mean'), html.Th('Standard deviation'), html.Th('Highest')
                ])),
                html.Tbody(html.Tr([
                    html.Th('Final score'), html.Td('{:.2f}'.format(end_mean)), html.Td('{:.2f}'.format(end_std)), html.Td(end_max)
                ]))
            ]
        )
    )
def final_hist():
    fig = px.histogram(
        x=finalScores,
        labels={'x':'Final score'},
        template='plotly_white',
        color_discrete_sequence=['lightblue'],
        histnorm='percent',
        cumulative=True,
        marginal='histogram'
    )
    fig.update_traces(marker_line_width=1, marker_line_color='white')
    return html.Div(
        id='final-hist',
        children=[
            dcc.Graph(figure=fig)
        ]
    )

def date_hist():
    fig = px.histogram(X, x='AiredDate', labels=lbs, template='plotly_white', color_discrete_sequence=['pink'], histnorm='percent')
    fig.update_traces(marker_line_width=1, marker_line_color='white')
    return html.Div(
        id='date-hist',
        children=[
            dcc.Graph(figure=fig)
        ]
    )

def date_violin():
    X['Decade'] = X['AiredDate'].dt.year//10*10 
    fig = px.violin(X, x='Decade', y='p_correctGuess', template='plotly_white', box=True, labels=lbs)
    return html.Div(
        id='date-violin',
        children=[
            dcc.Graph(figure=fig)
        ]
    )

app.layout = dbc.Container(
    [
        app_description(),
        generic_table(),
        types_table(),
        difficulty_correct_scatter(),
        final_table(),
        final_hist(),
        date_hist(),
        date_violin()
    ]
)

if __name__ == '__main__':
    app.run(debug=True)

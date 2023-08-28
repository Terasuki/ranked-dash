from dash import Dash, html, dcc, callback, Output, Input
import plotly.express as px
import pandas as pd
import numpy as np
import seaborn as sns
import dash_bootstrap_components as dbc
from sklearn.preprocessing import MultiLabelBinarizer
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import glob
import os

# Load files

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
X = novice.drop(['annId', 'urls', 'siteIds', 'animeScore', 'selfAnswer', 'fromList', 'gameMode', 'correct'], axis=1).copy()

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

X['anime_jp'] = X['anime'].apply(find_jp)
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

mlb = MultiLabelBinarizer()
s = X['tags']
dummy = pd.DataFrame(mlb.fit_transform(s), columns=mlb.classes_, index=X.index)
X = pd.concat([X, dummy], axis=1)

mlb2 = MultiLabelBinarizer()
s = X['genre']
dummy = pd.DataFrame(mlb2.fit_transform(s), columns=mlb2.classes_, index=X.index)
X = pd.concat([X, dummy], axis=1)

X = X.drop(['tags', 'genre'], axis=1)

genres = X[mlb2.classes_].sum().sort_values(ascending=False)
g_perc = pd.Series(index=mlb2.classes_, dtype='float64')
for genre in mlb2.classes_:
    g_perc[genre] = X.loc[X[genre] == 1]['p_correctGuess'].mean()
genres = pd.concat([genres, g_perc], axis=1)
genres.columns = ['Count', 'Guess rate']
genres = genres.sort_values(by='Guess rate', ascending=False)

tags = X[mlb.classes_].sum().sort_values(ascending=False)
t_perc = pd.Series(index=mlb.classes_, dtype='float64')
for tag in mlb.classes_:
    t_perc[tag] = X.loc[X[tag] == 1]['p_correctGuess'].mean()
tags = pd.concat([tags, t_perc], axis=1)
tags.columns = ['Count', 'Guess rate']
tags = tags.sort_values(by='Count', ascending=False).head(30).sort_values(by='Guess rate', ascending=False)

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
    )

def types_table():
    return html.Div(
        id='types-table',
        )

def generic_table():
    return html.Div(
        id='generic-table',
    )

def final_table():
    return html.Div(
        id='final-table',
    )

def final_hist():
    return html.Div(
        id='final-hist',
    )

def date_hist():
    return html.Div(
        id='date-hist',
    )

def date_violin():
    return html.Div(
        id='date-violin',
    )

def region_selector():
    return html.Div(
        children=[
            dbc.RadioItems(
                id='region-checklist',
                options=['East', 'Central', 'Both'],
                value='East'
            )
        ],
        style={'position':'fixed'}
    )

def trend_players():
    return html.Div(
        id='trend-players',
    )

def trend_score():
    return html.Div(
        id='trend-score',
    )

@callback(
    Output('generic-table', 'children'),
    Output('types-table', 'children'),
    Output('final-table', 'children'),
    Output('final-hist', 'children'),
    Output('trend-players', 'children'),
    Output('trend-score', 'children'),
    Output('date-violin', 'children'),
    Output('date-hist', 'children'),
    Output('diff-correct', 'children'),
    Input('region-checklist', 'value'))
def update_graphs(region):

    if region == 'Both':
        X_u = X
    else:
        X_u = X.loc[X['region'].isin([region])]

    n_games = X_u.shape[0]/45
    songs_played = X_u.shape[0]
    guess_rate = X_u['p_correctGuess'].mean()
    guess_time = X_u['avgGuessTime'].mean()
    no_guess_rate = X_u['p_noGuess'].mean()
    diff = X_u['difficulty'].mean()

    op = X_u.loc[X_u['type_noNumber'] == 'Opening']
    op_played = op.shape[0]
    op_rate = op['p_correctGuess'].mean()
    op_diff = op['difficulty'].mean()

    ed = X_u.loc[X_u['type_noNumber'] == 'Ending']
    ed_played = ed.shape[0]
    ed_rate = ed['p_correctGuess'].mean()
    ed_diff = ed['difficulty'].mean()

    ins = X_u.loc[X_u['type_noNumber'] == 'Insert']
    in_played = ins.shape[0]
    in_rate = ins['p_correctGuess'].mean()
    in_diff = ins['difficulty'].mean()

    last_song = pd.DataFrame(X_u.loc[X['songNumber'] == 45]).reset_index(drop=True)
    last_song['Scores'] = last_song['players'].apply(find_scores)

    finalScores = np.concatenate(last_song['Scores'].values).astype(int)
    end_mean = finalScores.mean()
    end_std = finalScores.std()
    end_max = finalScores.max()

    generic_t = dbc.Table(
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
    types_t = dbc.Table(
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
    final_t = dbc.Table(
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
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    hist_density = np.histogram(finalScores, bins=np.arange(47)-0.01, density=True)[0]*100
    fig.add_trace(
        go.Bar(x=np.arange(0, 46), y=hist_density, name='Final score', width=0.98),
        secondary_y=False,
    )
    fig.update_traces(marker_color='rgb(158,202,225)')
    fig.add_trace(
        go.Scatter(x=np.arange(0, 46), y=hist_density.cumsum() ,mode='lines+markers', name='Cumulative'),
        secondary_y=True,
    )
    fig.update_layout(template='plotly_white', yaxis=dict(
            title=dict(text='Percent'),
            side='left',
            range=[0, 5.1],
        ),
        yaxis2=dict(
            title=dict(text='Cumulative percent'),
            side='right',
            range=[0, 102],
            overlaying='y',
            tickmode='auto',
        ),)
    fig.update_xaxes(title_text='Final scores')

    trend_players = dcc.Graph(figure=px.line(last_song, x='logDate', y='totalPlayers',
        markers=True, 
        color='region',
        template='plotly_white',
        labels={'totalPlayers':'Number of players', 'logDate':'Date'}
    ))

    trend_score = dcc.Graph(figure=px.line(last_song, x='logDate', y=last_song['Scores'].apply(np.mean),
        markers=True, 
        color='region',
        template='plotly_white',
        labels={'y':'Mean score', 'logDate':'Date'}
    ))

    X_u['Decade'] = X_u['AiredDate'].dt.year//10*10 
    decade_violin = dcc.Graph(figure=px.violin(X_u, x='Decade', y='p_correctGuess', template='plotly_white', box=True, labels=lbs))

    date_hist = px.histogram(X_u, x='AiredDate', labels=lbs, template='plotly_white', color_discrete_sequence=['pink'], histnorm='percent')
    date_hist.update_traces(marker_line_width=1, marker_line_color='white')

    diff_correct = dcc.Graph(figure=px.scatter(
        X_u,
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
        opacity=0.5,
        category_orders={
            'type_noNumber':['Opening', 'Ending', 'Insert']
        }
        ))

    return generic_t, types_t, final_t, dcc.Graph(figure=fig), trend_players, trend_score, decade_violin, dcc.Graph(figure=date_hist), diff_correct

app.layout = dbc.Container(
    [
        html.Div(region_selector(), style={
            'display': 'inline-block',
            'vertical-align': 'top',
            'z-index':'10'
        }),
        html.Div([
            app_description(),
            generic_table(),
            types_table(),
            difficulty_correct_scatter(),
            final_table(),
            final_hist(),
            date_hist(),
            date_violin(),
            trend_players(),
            trend_score()
        ], style={
            'display': 'inline-block',
            'vertical-align': 'top', 
            'width':'90%',
            'float':'right',
            'z-index':'0'
        })
    ]
)

if __name__ == '__main__':
    app.run(debug=True)

from dash import Dash, html, dcc, callback, Output, Input
import plotly.express as px
import pandas as pd
import numpy as np
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import MultiLabelBinarizer
from datetime import date

# Load files

X = pd.read_parquet('./data/data.parquet')
last_song = pd.read_parquet('./data/last.parquet')
players_all = pd.read_parquet('./data/players.parquet')

def find_scores(players):
    scores = np.zeros(len(players))
    for i, x in enumerate(players):
        scores[i] = x['score']
    return scores

# Data names

lbs = {'difficulty':'Difficulty',
      'artist':'Artist',
      'name':'Song',
      'p_correctGuess':'Correct percent',
      'p_noGuess':'No guess percent',
      'anime_jp':'Anime',
      'avgGuessTime':'Average guess time',
      'AiredDate': 'Aired date',
      'type_noNumber':'Song type',
      'guessTime':'Player guess time',
      'correct':'Player correct'}

hover_dt = {'artist':True,
           'name':True,
           'anime_jp':True,
           'avgGuessTime':True,
           'difficulty':':.1f',
           'p_noGuess':':.1f',
           'p_correctGuess':':.1f',
           'avgGuessTime':':.1f',
        }

hover_dt_p = {'artist':True,
           'name':True,
           'anime_jp':True,
           'avgGuessTime':True,
           'difficulty':':.1f',
           'p_noGuess':':.1f',
           'p_correctGuess':':.1f',
           'avgGuessTime':':.1f',
           'guessTime':':.1f'
        }

# Dashboard

app = Dash(__name__, external_stylesheets=[dbc.themes.SANDSTONE])
app.title = 'AMQ Ranked Stats'
server = app.server

def app_description():
    return html.Div(
        id='app-description',
        children=[
            html.H2('Ranked Stats'),
        ]
    )

def footnote():
    return html.Div(
        id='footnote',
        children=[
            html.H5('Author: Terasuki. Using a free instance, so expect slow dashboard updates.')
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

def mode_selector():
    return html.Div(
        children=[
            dbc.RadioItems(
                id='mode-checklist',
                options=['Novice', 'Expert'],
                value='Novice'
            )
        ],
        style={'position':'fixed',
               'top':'200px'}
    )

def trend_players():
    return html.Div(
        id='trend-players',
    )

def trend_score():
    return html.Div(
        id='trend-score',
    )

def genres():
    return html.Div(
        id='genres',
    )

def tags():
    return html.Div(
        id='tags',
    )

def date_picker():
    return html.Div(
        children=[dcc.DatePickerRange(
            id='date-picker',
            start_date=date(2023, 2, 12),
            end_date=date(2023, 12, 31),
            min_date_allowed=date(2023, 2, 12),
            first_day_of_week=1,
            number_of_months_shown=3
        )],
        style={'margin-bottom':'20px'}
    )

def player_picker():
    return html.Div(
        children=[dbc.Input(
            id='player-picker',
            placeholder='Insert username'
        )],
        style={'margin-bottom':'20px'}
    )

def hist_personal():
    return html.Div(
        id='hist-personal'
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
    Output('genres', 'children'),
    Output('tags', 'children'),
    Output('hist-personal', 'children'),
    Input('region-checklist', 'value'),
    Input('date-picker', 'start_date'),
    Input('date-picker', 'end_date'),
    Input('mode-checklist', 'value'),
    Input('player-picker', 'value'))
def update_graphs(region, start_dt, end_dt, mode, username):

    players_s = players_all.loc[players_all['name'] == username].drop('name', axis=1)
    if not players_s.empty:
        X_u = X.join(players_s).dropna(subset='score')
        last_song_u = last_song.join(players_s).dropna(subset='score')
    else:
        X_u = X
        last_song_u = last_song
    if last_song_u.empty:
            last_song_u = last_song

    if region == 'Both':
        X_u = X_u
        last_song_u = last_song_u
    else:
        X_u = X_u.loc[X['region'].isin([region])]
        last_song_u = last_song_u.loc[last_song_u['region'].isin([region])]

    X_u = X_u.loc[X_u['rankedMode'].isin([mode])]
    last_song_u = last_song_u.loc[last_song_u['rankedMode'].isin([mode])]

    X_u = X_u[(X_u['logDate'] >= pd.Timestamp(date.fromisoformat(start_dt))) & (X_u['logDate'] <= pd.Timestamp(date.fromisoformat(end_dt)))]
    last_song_u = last_song_u[(last_song_u['logDate'] >= pd.Timestamp(date.fromisoformat(start_dt))) & (last_song_u['logDate'] <= pd.Timestamp(date.fromisoformat(end_dt)))]

    n_games = last_song_u.shape[0]
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

    if last_song_u.empty:
            last_song_u['Scores'] = [[0,1]]
            print('No last songs recorded')
    finalScores = np.concatenate(last_song_u['Scores'].values).astype(int)
    end_mean = finalScores.mean()
    end_std = finalScores.std()
    end_max = finalScores.max()

    # Player specific
    guess_p, op_p, ed_p, in_p, gt_p, ng_p, end_p_mean, end_p_std, end_p_max = [0]*9
    if not players_s.empty:
        guess_p = X_u['correct'].sum()/songs_played * 100
        op_p = op['correct'].sum()/op_played*100
        ed_p = ed['correct'].sum()/ed_played*100
        in_p = ins['correct'].sum()/in_played*100
        gt_p = X_u['guessTime'].mean()
        ng_p = X_u['guessTime'].isna().sum()/songs_played*100
        end_p_mean = last_song_u['score'].mean()
        end_p_std = last_song_u['score'].std()
        end_p_max = int(last_song_u['score'].max())

    mlb = MultiLabelBinarizer()
    s = X_u['tags']
    dummy = pd.DataFrame(mlb.fit_transform(s), columns=mlb.classes_, index=X_u.index, dtype=np.bool_)
    X_u = pd.concat([X_u, dummy], axis=1)

    mlb2 = MultiLabelBinarizer()
    s = X_u['genre']
    dummy = pd.DataFrame(mlb2.fit_transform(s), columns=mlb2.classes_, index=X_u.index, dtype=np.bool_)
    X_u = pd.concat([X_u, dummy], axis=1)

    X_u = X_u.drop(['tags', 'genre'], axis=1)

    genres = X_u[mlb2.classes_].sum().sort_values(ascending=False)
    g_perc = pd.Series(index=mlb2.classes_, dtype='float64')
    for genre in mlb2.classes_:
        g_perc[genre] = X_u.loc[X_u[genre] == 1]['p_correctGuess'].mean()
    genres = pd.concat([genres, g_perc], axis=1)
    genres.columns = ['Count', 'Guess rate']
    genres = genres.sort_values(by='Guess rate', ascending=False)

    tags = X_u[mlb.classes_].sum().sort_values(ascending=False)
    t_perc = pd.Series(index=mlb.classes_, dtype='float64')
    for tag in mlb.classes_:
        t_perc[tag] = X_u.loc[X_u[tag] == 1]['p_correctGuess'].mean()
    tags = pd.concat([tags, t_perc], axis=1)
    tags.columns = ['Count', 'Guess rate']
    tags = tags.sort_values(by='Count', ascending=False).head(30).sort_values(by='Guess rate', ascending=False)

    generic_t = dbc.Table(
            bordered=True,
            striped=True,
            children=[
                html.Thead(html.Tr([
                    html.Th('Games recorded'), html.Th('Songs recorded'), html.Th('Correct guess rate'), html.Th('Average difficulty'), html.Th('Average guess time'), html.Th('No guess rate')
                    ])),
                html.Tbody(html.Tr([
                    html.Td(n_games), html.Td(songs_played), html.Td('{:.2f} ({:.2f})'.format(guess_rate, guess_p)), html.Td('{:.2f}'.format(diff)), html.Td('{:.2f} ({:.2f})'.format(guess_time, gt_p)), html.Td('{:.2f} ({:.2f})'.format(no_guess_rate, ng_p))
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
                    html.Tr(children=[html.Th('Correct guess rate'), html.Td('{:.2f} ({:.2f})'.format(op_rate, op_p)), html.Td('{:.2f} ({:.2f})'.format(ed_rate, ed_p)), html.Td('{:.2f} ({:.2f})'.format(in_rate, in_p))]),
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
                    html.Th('Final score'), html.Td('{:.2f} ({:.2f})'.format(end_mean, end_p_mean)), html.Td('{:.2f} ({:.2f})'.format(end_std, end_p_std)), html.Td(f'{end_max} ({end_p_max})')
                ]))
            ]
        )
    
    if mode == 'Expert':
        number_of_songs = 85
    else:
        number_of_songs = 45
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    hist_density = np.histogram(finalScores, bins=np.arange(number_of_songs+2)-0.01, density=True)[0]*100
    fig.add_trace(
        go.Bar(x=np.arange(0, number_of_songs+1), y=hist_density, name='Final score', width=0.98),
        secondary_y=False,
    )
    fig.update_traces(marker_color='rgb(158,202,225)')
    fig.add_trace(
        go.Scatter(x=np.arange(0, number_of_songs+1), y=hist_density.cumsum() ,mode='lines+markers', name='Cumulative'),
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
    hist_pers = dcc.Markdown('')
    if (not last_song_u.empty) and (not players_s.empty):
        hist_p = make_subplots(specs=[[{'secondary_y': True}]])
        hist_temp = np.histogram(last_song_u['score'], bins=np.arange(number_of_songs+2)-0.01, density=True)[0]*100
        hist_p.add_trace(
            go.Bar(x=np.arange(0, number_of_songs+1), y=hist_temp, name=f'{username} - Final score', width=0.98),
            secondary_y=False
        )
        hist_p.update_traces(marker_color='rgb(204,204,255)')
        hist_p.add_trace(
            go.Scatter(x=np.arange(0, number_of_songs+1), y=hist_temp.cumsum() ,mode='lines+markers', name='Cumulative'),
            secondary_y=True,
        )
        hist_p.update_layout(template='plotly_white', yaxis=dict(
            title=dict(text='Percent'),
            side='left',
            range=[0, 51],
        ),
        yaxis2=dict(
            title=dict(text='Cumulative percent'),
            side='right',
            range=[0, 102],
            overlaying='y',
            tickmode='auto',
        ),)
        hist_p.update_xaxes(title_text='Final scores')
        hist_pers = dcc.Graph(figure=hist_p)

    trend_players = dcc.Graph(figure=px.line(last_song_u, x='logDate', y='totalPlayers',
        markers=True, 
        color='region',
        template='plotly_white',
        labels={'totalPlayers':'Number of players', 'logDate':'Date'}
    ))

    score_fig = px.line(last_song_u, x='logDate', y=last_song_u['Scores'].apply(np.mean),
        markers=True, 
        color='region',
        template='plotly_white',
        labels={'y':'Mean score', 'logDate':'Date'}
    )
    if (not last_song_u.empty) and (not players_s.empty):
        if region == 'Both':
            score_fig.add_trace(
                go.Scatter(x=last_song_u.loc[last_song_u['region'] == 'East']['logDate'], y=last_song_u['score'], mode='lines+markers', name=f'{username} - East')
            )
            score_fig.add_trace(
                go.Scatter(x=last_song_u.loc[last_song_u['region'] == 'Central']['logDate'], y=last_song_u['score'], mode='lines+markers', name=f'{username} - Central')
            )
        else: 
            score_fig.add_trace(
                go.Scatter(x=last_song_u['logDate'], y=last_song_u['score'], mode='lines+markers', name=username)
            )
    trend_score = dcc.Graph(figure=score_fig)

    X_u['Decade'] = X_u['AiredDate'].dt.year//10*10 
    violin_graph = px.violin(X_u, x='Decade', y='p_correctGuess', template='plotly_white', box=True, labels=lbs, range_y=[0, 100])
    if not players_s.empty:
        decade_p = X_u.groupby('Decade')['correct'].mean()*100
        violin_graph.add_trace(
            go.Scatter(x=decade_p.keys(), y=decade_p, mode='lines+markers', name=username)
        )
    decade_violin = dcc.Graph(figure=violin_graph)
    date_hist = px.histogram(X_u, x='AiredDate', labels=lbs, template='plotly_white', color_discrete_sequence=['pink'], histnorm='percent')

    if not players_s.empty:
        date_hist = px.histogram(X_u, x='AiredDate', labels=lbs, template='plotly_white', color_discrete_sequence=['pink', 'purple'],
                                  histnorm='percent', color='correct', category_orders={'correct':[True, False]})

    date_hist.update_traces(marker_line_width=1, marker_line_color='white')

    if players_s.empty:
        d_c = px.scatter(
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
            },
            render_mode='webgl'
        )
    else:
        d_c = px.scatter(X_u, x='difficulty', y='p_correctGuess', color='type_noNumber',
                         marginal_x='histogram', marginal_y='histogram', hover_data=hover_dt_p, labels=lbs, template='plotly_white', range_x=[0, 100],
                         range_y=[0, 100], opacity=0.5, category_orders={'type_noNumber':['Opening', 'Ending', 'Insert']}, render_mode='webgl',
                         symbol='correct')
    diff_correct = dcc.Graph(figure=d_c)
    
    genres_bar = make_subplots(specs=[[{"secondary_y": True}]])

    genres_bar.add_trace(
        go.Bar(x=genres['Count'].keys(), y=genres['Count']/songs_played, name='Appareance', width=0.98),
        secondary_y=False,
    )
    genres_bar.update_traces(marker_color='rgb(149,237,158)')
    genres_bar.add_trace(
        go.Scatter(x=genres['Count'].keys(), y=genres['Guess rate'], mode='lines+markers', name='Correct'),
        secondary_y=True,
    )
    genres_bar.update_traces(marker_color='rgb(78, 100, 207)', selector=dict(type='scatter'))
    genres_bar.update_layout(template='plotly_white', yaxis=dict(
            title=dict(text='Appereance rate'),
            side='left',
            range=[0, 1],
        ),
        yaxis2=dict(
            title=dict(text='Correct percent'),
            side='right',
            range=[0, 100],
            overlaying='y',
        ),)
    genres_bar.update_xaxes(title_text='Genres')

    if not players_s.empty:
        g_perc_p = (X_u.loc[X_u['correct'] == True][mlb2.classes_]).sum()/genres['Count'] * 100
        genres_bar.add_trace(
            go.Scatter(x=genres['Count'].keys(), y=g_perc_p.reindex(genres['Count'].keys()), mode='lines+markers', name=username),
            secondary_y=True
        )

    tags_bar = make_subplots(specs=[[{"secondary_y": True}]])

    tags_bar.add_trace(
        go.Bar(x=tags['Count'].keys(), y=tags['Count']/songs_played, name='Appareance', width=0.98),
        secondary_y=False,
    )
    tags_bar.update_traces(marker_color='rgb(149,237,158)')
    tags_bar.add_trace(
        go.Scatter(x=tags['Count'].keys(), y=tags['Guess rate'], mode='lines+markers', name='Correct'),
        secondary_y=True,
    )
    tags_bar.update_traces(marker_color='rgb(78, 100, 207)', selector=dict(type='scatter'))
    tags_bar.update_layout(template='plotly_white', yaxis=dict(
            title=dict(text='Appereance rate'),
            side='left',
            range=[0, 1],
        ),
        yaxis2=dict(
            title=dict(text='Correct percent'),
            side='right',
            range=[0, 100],
            overlaying='y',
        ),)
    tags_bar.update_xaxes(title_text='Tags')

    if not players_s.empty:
        t_perc_p = (X_u.loc[X_u['correct'] == True][tags['Count'].keys()]).sum()/tags['Count'] * 100
        tags_bar.add_trace(
            go.Scatter(x=tags['Count'].keys(), y=t_perc_p.reindex(tags['Count'].keys()), mode='lines+markers', name=username),
            secondary_y=True
        )

    return generic_t, types_t, final_t, dcc.Graph(figure=fig), trend_players, trend_score, decade_violin, dcc.Graph(figure=date_hist), diff_correct, dcc.Graph(figure=genres_bar), dcc.Graph(figure=tags_bar), hist_pers

app.layout = dbc.Container(
    [
        html.Div(children=[region_selector(), mode_selector()], style={
            'display': 'inline-block',
            'vertical-align': 'top',
            'z-index':'10',
            'margin-top':'10px'
        }),
        html.Div([
            app_description(),
            date_picker(),
            player_picker(),
            generic_table(),
            types_table(),
            difficulty_correct_scatter(),
            final_table(),
            final_hist(),
            hist_personal(),
            date_hist(),
            date_violin(),
            genres(),
            tags(),
            trend_players(),
            trend_score(),
            footnote()
        ], style={
            'display': 'inline-block',
            'vertical-align': 'top', 
            'width':'90%',
            'float':'right',
            'z-index':'0',
            'margin-top':'10px'
        })
    ]
)

if __name__ == '__main__':
    app.run(debug=True)

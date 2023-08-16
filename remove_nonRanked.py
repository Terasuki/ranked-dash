import pandas as pd
import numpy as np

FILE_PATH = './song_export_2023-04-01_14-08-03.json'
df = pd.read_json(FILE_PATH)
df = df.loc[df['gameMode'] == 'Ranked'].reset_index(drop=True)
df.to_json(FILE_PATH)

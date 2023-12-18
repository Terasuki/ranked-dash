import pandas as pd

FILE_PATH = ''
df = pd.read_json(FILE_PATH)
df = df.loc[df['gameMode'] == 'Ranked'].reset_index(drop=True)
df.to_json(FILE_PATH)

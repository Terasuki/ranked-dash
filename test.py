import pandas as pd
from memory_profiler import profile

@profile
def load_data():
    return pd.read_parquet('./data.parquet')

if __name__ == '__main__':
    X = load_data()
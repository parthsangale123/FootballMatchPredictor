import pandas as pd
import sqlite3
import glob

def build_unified_db():
    files = glob.glob('global_vault/*.csv')
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    
    # DROP results and odds, but KEEP FTHG and FTAG as targets
    cols_to_drop = ['FTR', 'HTR', 'Date', 'Time', 'Div']
    cols_to_drop += [c for c in df.columns if c.startswith(('B365', 'P', 'Max', 'Avg', 'HTHG', 'HTAG'))]
    
    # Drop only the columns defined in cols_to_drop
    df = df.drop(columns=cols_to_drop, errors='ignore').fillna(0)
    
    with sqlite3.connect('football_data.db') as conn:
        df.to_sql('match_results', conn, if_exists='replace', index=False)
    print("Database built with goal targets preserved.")

if __name__ == "__main__": build_unified_db()

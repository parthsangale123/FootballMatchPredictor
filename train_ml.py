import sqlite3, pandas as pd, numpy as np, tensorflow as tf, joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def train():
    conn = sqlite3.connect('football_data.db')
    df = pd.read_sql("SELECT * FROM match_results", conn)
    conn.close()

    # Identify features
    home_cols = [c for c in df.columns if c.startswith('H') and c not in ['HomeTeam', 'FTHG', 'FTAG']]
    away_cols = [c for c in df.columns if c.startswith('A') and c not in ['AwayTeam', 'FTHG', 'FTAG']]
    
    # Save column lists
    joblib.dump(home_cols, 'home_cols.pkl')
    joblib.dump(away_cols, 'away_cols.pkl')
    
    # Create the feature matrix X with positional flags
    # We add a column of 1s for Home stats and 0s for Away stats
    home_data = df[home_cols].values
    home_flags = np.ones((len(df), 1))
    away_data = df[away_cols].values
    away_flags = np.zeros((len(df), 1))
    
    # Concatenate: [Home_Stats, Home_Flag, Away_Stats, Away_Flag]
    X = np.hstack([home_data, home_flags, away_data, away_flags]).astype(np.float32)
    y = df[['FTHG', 'FTAG']].values.astype(np.float32)
    
    # Scale
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    joblib.dump(scaler, 'scaler.joblib')
    
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=(X.shape[1],)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(2) 
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=50, batch_size=32)
    model.save('score_model.keras')
    print("✅ Model trained with position-aware flags.")

if __name__ == "__main__": train()

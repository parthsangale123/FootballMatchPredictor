import tensorflow as tf, joblib, numpy as np, sqlite3, pandas as pd

def predict_score(team_a, team_b):
    model = tf.keras.models.load_model('score_model.keras')
    scaler = joblib.load('scaler.joblib')
    h_cols, a_cols = joblib.load('home_cols.pkl'), joblib.load('away_cols.pkl')
    
    conn = sqlite3.connect('football_data.db')
    def get_stats(team, is_home):
        col = "HomeTeam" if is_home else "AwayTeam"
        query = "SELECT * FROM match_results WHERE " + col + " = ?"
        df = pd.read_sql(query, conn, params=(team,))
        return df[h_cols if is_home else a_cols].mean().fillna(0).values
    
    stats_a = get_stats(team_a, True)  # Home stats
    stats_b = get_stats(team_b, False) # Away stats
    
    # Add flags: 1 for Home team (A), 0 for Away team (B)
    input_vec = np.concatenate([stats_a, [1.0], stats_b, [0.0]]).reshape(1, -1)
    
    scaled = scaler.transform(input_vec)
    conn.close()
    return model.predict(scaled)[0]

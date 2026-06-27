import streamlit as st, engine, sqlite3, pandas as pd
st.title("⚽ Expected Goal Predictor")
conn = sqlite3.connect('football_data.db')
teams = pd.read_sql("SELECT DISTINCT HomeTeam FROM match_results", conn)['HomeTeam'].unique().tolist()
conn.close()
t1, t2 = st.selectbox("Away Team", teams), st.selectbox("Home Team", teams)
if st.button("Predict"):
    score = engine.predict_score(t1, t2)
    st.write(f"## Team 1 xG - Team 2 xG: {max(0, score[0]):.1f} - {max(0, score[1]):.1f}")

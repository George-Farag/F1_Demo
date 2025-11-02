import streamlit as st
import fastf1 as f1
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

st.set_page_config(page_title="F1 Race Predictor", layout="wide")

st.title("🏎️ F1 Race Position Predictor (Las Vegas GP 2024)")

with st.spinner("Loading race data..."):
    f1.Cache.enable_cache('cache')
    session = f1.get_session(2024, 'Las Vegas', 'R')
    session.load()

laps = session.laps
st.subheader("📊 Raw Lap Data")
st.dataframe(laps.head(10))

df = laps[['Driver', 'LapTime', 'LapNumber', 'Compound', 'TyreLife',
           'Sector1Time', 'Sector2Time', 'Sector3Time',
           'SpeedI1', 'SpeedFL', 'SpeedST']].copy()

for col in ['LapTime', 'Sector1Time', 'Sector2Time', 'Sector3Time']:
    df[col] = df[col].dt.total_seconds()

df_driver = df.groupby('Driver').agg({
    'LapTime': 'mean',
    'Sector1Time': 'mean',
    'Sector2Time': 'mean',
    'Sector3Time': 'mean',
    'TyreLife': 'mean',
    'LapNumber': 'max'
}).reset_index()

results = session.results[['DriverNumber', 'Abbreviation', 'Position']]
df_driver = df_driver.merge(results, left_on='Driver', right_on='Abbreviation')

st.subheader("🏁 Processed Driver Averages")
st.dataframe(df_driver)

x = df_driver[['LapTime', 'Sector1Time', 'Sector2Time', 'Sector3Time', 'TyreLife', 'LapNumber']]
y = df_driver['Position']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(x_train, y_train)
y_pred = model.predict(x_test)

mae = mean_absolute_error(y_test, y_pred)
st.metric(label="Mean Absolute Error", value=f"{mae:.2f}")

fig, ax = plt.subplots()
ax.scatter(y_test, y_pred)
ax.set_xlabel("Actual Position")
ax.set_ylabel("Predicted Position")
ax.set_title("Race Position Prediction")
st.pyplot(fig)

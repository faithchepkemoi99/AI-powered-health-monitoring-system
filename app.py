import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt

# App title
st.title("🩺 AI-Powered Health Monitor")
st.write("Detect anomalies in wearable health data.")

# File upload
uploaded_file = st.file_uploader("Upload CSV file with health data", type=["csv"])

# Use uploaded data or simulate
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("File uploaded successfully.")
else:
    st.info("No file uploaded. Using simulated sample data.")
    timestamps = pd.date_range(start='2025-07-14 00:00', end='2025-07-14 23:55', freq='5min')
    heart_rate = np.random.normal(loc=75, scale=5, size=len(timestamps))
    oxygen_level = np.random.normal(loc=98, scale=1, size=len(timestamps))
    steps = np.random.poisson(lam=5, size=len(timestamps))
    df = pd.DataFrame({
        'timestamp': timestamps,
        'heart_rate': np.round(heart_rate),
        'oxygen_level': np.round(oxygen_level),
        'step_count': steps
    })
    df.loc[100, 'heart_rate'] = 150
    df.loc[200, 'oxygen_level'] = 85
    df.loc[250, 'step_count'] = 200

# Display data
st.subheader("📋 Data Preview")
st.dataframe(df.head())

# Preprocess
features = df[['heart_rate', 'oxygen_level', 'step_count']]
scaler = MinMaxScaler()
features_scaled = scaler.fit_transform(features)
df_scaled = pd.DataFrame(features_scaled, columns=features.columns)

# Anomaly detection
model = IsolationForest(contamination=0.01, random_state=42)
model.fit(df_scaled)
df_scaled['anomaly'] = model.predict(df_scaled).astype(int)
df_scaled['anomaly'] = df_scaled['anomaly'].map({1: 0, -1: 1})
df_scaled['timestamp'] = df['timestamp']

# Plot
st.subheader("📈 Heart Rate with Anomalies")
fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(df_scaled['timestamp'], df['heart_rate'], label='Heart Rate', color='blue')
ax.scatter(df_scaled[df_scaled['anomaly'] == 1]['timestamp'],
           df[df_scaled['anomaly'] == 1]['heart_rate'],
           color='red', label='Anomaly', s=50)
ax.legend()
ax.set_xlabel("Time")
ax.set_ylabel("Heart Rate")
st.pyplot(fig)

# Summary
num_anomalies = df_scaled['anomaly'].sum()
st.subheader("🧠 Summary")
st.write(f"Detected {num_anomalies} anomalies out of {len(df_scaled)} data points.")

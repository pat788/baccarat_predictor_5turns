
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# สร้างข้อมูลจำลอง
np.random.seed(42)
rounds = 1000
results = np.random.choice(['P', 'B', 'T'], size=rounds, p=[0.45, 0.45, 0.10])
df = pd.DataFrame({'result': results})
result_map = {'P': 0, 'B': 1, 'T': 2}
df['result_code'] = df['result'].map(result_map)

# ใช้ผลย้อนหลัง 5 ตา
for i in range(1, 6):
    df[f'prev_{i}'] = df['result_code'].shift(i)
df.dropna(inplace=True)

X = df[['prev_1', 'prev_2', 'prev_3', 'prev_4', 'prev_5']]
y = df['result_code']

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# UI
st.title("Baccarat Predictor (5 ตาหลังสุด)")
st.write("กรอกผลลัพธ์ย้อนหลัง 5 ตา เพื่อทำนายผลลัพธ์ตาถัดไป")

option_map = {"Player (P)": 0, "Banker (B)": 1, "Tie (T)": 2}

col = st.columns(5)
prev_inputs = [col[i].selectbox(f"ผลตาก่อนหน้า {i+1}", list(option_map.keys())) for i in range(5)]

if st.button("ทำนายผลลัพธ์ถัดไป"):
    input_data = [[option_map[i] for i in prev_inputs]]
    prediction = model.predict(input_data)[0]
    result_decode = {0: "Player (P)", 1: "Banker (B)", 2: "Tie (T)"}
    st.success(f"ระบบคาดการณ์ว่า: **{result_decode[prediction]}**")

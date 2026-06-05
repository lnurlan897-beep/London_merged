import streamlit as st
import pandas as pd
import pickle

# Səhifə konfiqurasiyası
st.set_page_config(
    page_title="London Bike Sharing Predictor",
    page_icon="🚲",
    layout="wide"
)

# CSS dizayn
st.markdown("""
<style>
.main {
    background-color: #f5f7fa;
}
.big-font {
    font-size:40px !important;
    font-weight:bold;
    text-align:center;
    color:#1f77b4;
}
.result-box {
    padding:20px;
    border-radius:15px;
    background-color:#d4edda;
    color:#155724;
    font-size:30px;
    text-align:center;
    font-weight:bold;
}
</style>
""", unsafe_allow_html=True)

# Modeli yükləyirik
model = pickle.load(open("model.pkl", "rb"))

# Başlıq
st.markdown('<p class="big-font">🚲 London Bike Sharing Prediction</p>',
            unsafe_allow_html=True)

st.write(
    """
    Bu tətbiq hava şəraiti və digər məlumatlardan istifadə edərək
    velosiped kirayəsi sayını proqnozlaşdırır.
    """
)

st.divider()

# Input hissəsi
col1, col2, col3 = st.columns(3)

with col1:
    t1 = st.number_input(
        "🌡️ Real Temperature (°C)",
        min_value=-20.0,
        max_value=50.0,
        value=20.0
    )

    hum = st.slider(
        "💧 Humidity (%)",
        0,
        100,
        60
    )

    wind_speed = st.slider(
        "🌬️ Wind Speed",
        0.0,
        100.0,
        15.0
    )

with col2:
    weather_code = st.selectbox(
        "☁️ Weather Code",
        [1, 2, 3, 4, 7, 10, 26]
    )

    is_holiday = st.selectbox(
        "🎉 Holiday",
        [0, 1]
    )

    is_weekend = st.selectbox(
        "🏖️ Weekend",
        [0, 1]
    )

with col3:
    season = st.selectbox(
        "🌸 Season",
        [0, 1, 2, 3]
    )

    day = st.slider(
        "📅 Day",
        1,
        31,
        15
    )

    hour = st.slider(
        "🕒 Hour",
        0,
        23,
        12
    )

# Əgər datasetdə bu sütunlar varsa onları əlavə et
temp = t1
feels_like_temp = t1

# Model inputu
input_data = pd.DataFrame({
    't1': [temp],
    't2': [feels_like_temp],
    'hum': [hum],
    'wind_speed': [wind_speed],
    'weather_code': [weather_code],
    'is_holiday': [is_holiday],
    'is_weekend': [is_weekend],
    'season': [season],
    'day': [day],
    'hour': [hour]
})

st.divider()

# Predict düyməsi
if st.button("🚀 Predict Bike Rentals", use_container_width=True):

    prediction = model.predict(input_data)[0]

    st.markdown(
        f"""
        <div class="result-box">
            🚴 Estimated Bike Rentals<br>
            {int(prediction):,}
        </div>
        """,
        unsafe_allow_html=True
    )

    st.balloons()

# Footer
st.markdown("---")
st.caption("Built with ❤️ using Streamlit & XGBoost")
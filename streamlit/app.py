import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go
from pathlib import Path

st.set_page_config(page_title="Prediksi Kepuasan Penumpang", page_icon="✈️", layout="wide")

# tema
if "theme" not in st.session_state:
    st.session_state.theme = "dark"

def toggle_theme():
    st.session_state.theme = "light" if st.session_state.theme == "dark" else "dark"

if st.session_state.theme == "light":
    st.markdown("""
    <style>
    .main { background: #ffffff; }
    h1, h2, h3, p, span, label, div { color: #1a1a1a !important; }
    .stDataFrame { color: #1a1a1a; }
    </style>
    """, unsafe_allow_html=True)

# load model
BASE_DIR = Path(__file__).resolve().parent.parent / "model"
lr_model = joblib.load(BASE_DIR / "logistic_regression.pkl")
dt_model = joblib.load(BASE_DIR / "decision_tree.pkl")
rf_model = joblib.load(BASE_DIR / "random_forest.pkl")
xgb_model = joblib.load(BASE_DIR / "xgboost.pkl")
encoders = joblib.load(BASE_DIR / "label_encoders.pkl")
scaler = joblib.load(BASE_DIR / "standard_scaler.pkl")

top_left, top_right = st.columns([6, 1])
with top_left:
    st.title("✈️ Prediksi Kepuasan Penumpang Maskapai")
    st.caption("Masukkan data penumpang untuk memprediksi apakah penumpang merasa puas atau tidak.")
with top_right:
    mode_label = "☀️ Light" if st.session_state.theme == "dark" else "🌙 Dark"
    st.button(mode_label, on_click=toggle_theme)

st.divider()

col_input, col_result = st.columns([1.4, 1], gap="large")

with col_input:

    st.subheader("Data Penumpang")

    r1c1, r1c2 = st.columns(2)
    customer_type = r1c1.selectbox("Tipe Pelanggan", ["Loyal Customer", "Disloyal Customer"])
    travel_type = r1c2.selectbox("Tujuan Perjalanan", ["Business travel", "Personal Travel"])

    r2c1, r2c2, r2c3 = st.columns(3)
    travel_class = r2c1.selectbox("Kelas", ["Business", "Eco", "Eco Plus"])
    age = r2c2.number_input("Umur", 7, 85, 35)
    flight_distance = r2c3.number_input("Jarak (mil)", 50, 7000, 1500)

    r3c1, r3c2 = st.columns(2)
    departure_delay = r3c1.number_input("Delay Berangkat (menit)", 0, 2000, 0)
    arrival_delay = r3c2.number_input("Delay Tiba (menit)", 0, 2000, 0)

    st.divider()
    st.subheader("Rating Layanan (0-5)")

    r4c1, r4c2, r4c3 = st.columns(3)
    seat_comfort = r4c1.slider("Seat Comfort", 0, 5, 3)
    food_drink = r4c2.slider("Food & Drink", 0, 5, 3)
    inflight_wifi = r4c3.slider("Inflight Wifi", 0, 5, 3)

    r5c1, r5c2, r5c3 = st.columns(3)
    inflight_entertainment = r5c1.slider("Entertainment", 0, 5, 3)
    onboard_service = r5c2.slider("On-board Service", 0, 5, 3)
    leg_room = r5c3.slider("Leg Room", 0, 5, 3)

    r6c1, r6c2, r6c3 = st.columns(3)
    online_booking = r6c1.slider("Online Booking", 0, 5, 3)
    online_boarding = r6c2.slider("Online Boarding", 0, 5, 3)
    online_support = r6c3.slider("Online Support", 0, 5, 3)

    r7c1, r7c2, r7c3 = st.columns(3)
    baggage = r7c1.slider("Baggage Handling", 1, 5, 3)
    checkin = r7c2.slider("Check-in Service", 0, 5, 3)
    cleanliness = r7c3.slider("Cleanliness", 0, 5, 3)

    r8c1, r8c2 = st.columns(2)
    gate_location = r8c1.slider("Gate Location", 0, 5, 3)
    departure_arrival = r8c2.slider("Dep/Arr Time Convenient", 0, 5, 3)

    st.markdown("")
    predict = st.button("Prediksi", type="primary")


with col_result:

    if predict:

        # encode
        ct_enc = encoders["Customer Type"].transform([customer_type])[0]
        tt_enc = encoders["Type of Travel"].transform([travel_type])[0]
        tc_enc = encoders["Class"].transform([travel_class])[0]

        input_df = pd.DataFrame({
            "Customer Type": [ct_enc], "Age": [age],
            "Type of Travel": [tt_enc], "Class": [tc_enc],
            "Flight Distance": [flight_distance],
            "Seat comfort": [seat_comfort],
            "Departure/Arrival time convenient": [departure_arrival],
            "Food and drink": [food_drink],
            "Gate location": [gate_location],
            "Inflight wifi service": [inflight_wifi],
            "Inflight entertainment": [inflight_entertainment],
            "Online support": [online_support],
            "Ease of Online booking": [online_booking],
            "On-board service": [onboard_service],
            "Leg room service": [leg_room],
            "Baggage handling": [baggage],
            "Checkin service": [checkin],
            "Cleanliness": [cleanliness],
            "Online boarding": [online_boarding],
            "Departure Delay in Minutes": [departure_delay],
            "Arrival Delay in Minutes": [arrival_delay]
        })

        input_scaled = scaler.transform(input_df)

        # prediksi semua model
        preds = {}
        for name, mdl, data in [
            ("Logistic Regression", lr_model, input_scaled),
            ("Decision Tree", dt_model, input_df),
            ("Random Forest", rf_model, input_df),
            ("XGBoost", xgb_model, input_df),
        ]:
            p = mdl.predict(data)[0]
            prob = mdl.predict_proba(data)[0].max()
            preds[name] = (p, prob)

        # hasil utama
        best_pred, best_prob = preds["XGBoost"]
        label = "Satisfied" if best_pred == 1 else "Dissatisfied"

        st.subheader("Hasil Prediksi")

        if best_pred == 1:
            st.success(f"**{label}** — confidence {best_prob*100:.1f}%")
        else:
            st.error(f"**{label}** — confidence {best_prob*100:.1f}%")

        st.caption("Berdasarkan model XGBoost (akurasi tertinggi)")

        # tabel semua model
        st.divider()
        st.subheader("Perbandingan Model")

        tabel = []
        for name, (p, prob) in preds.items():
            tabel.append({
                "Model": name,
                "Prediksi": "Satisfied" if p == 1 else "Dissatisfied",
                "Confidence": f"{prob*100:.1f}%"
            })

        st.dataframe(pd.DataFrame(tabel), width="stretch", hide_index=True)

        # grafik performa
        st.divider()
        st.subheader("Performa Training")

        perf = pd.DataFrame({
            "Model": ["Logistic Reg.", "Decision Tree", "Random Forest", "XGBoost"],
            "Accuracy": [82.89, 93.23, 95.58, 95.80],
            "Precision": [84.45, 93.77, 96.89, 97.03],
            "Recall": [84.25, 93.86, 94.97, 95.25],
            "F1 Score": [84.35, 93.82, 95.92, 96.13]
        })

        fig = go.Figure()
        for metric, color in [("Accuracy", "#636efa"), ("Precision", "#ab63fa"),
                              ("Recall", "#00cc96"), ("F1 Score", "#ffa15a")]:
            fig.add_trace(go.Bar(name=metric, x=perf["Model"], y=perf[metric], marker_color=color))

        fig.update_layout(
            barmode="group",
            yaxis=dict(range=[75, 100], ticksuffix="%"),
            legend=dict(orientation="h", y=1.15, x=0.5, xanchor="center"),
            margin=dict(l=0, r=0, t=10, b=0),
            height=300,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(size=11)
        )

        st.plotly_chart(fig, width="stretch")

    else:
        st.info("Isi data penumpang di sebelah kiri, lalu tekan **Prediksi**.")

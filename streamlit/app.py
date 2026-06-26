# ==========================================================
# IMPORT LIBRARY
# ==========================================================

import streamlit as st
import pandas as pd
import joblib

# ==========================================================
# KONFIGURASI HALAMAN
# ==========================================================

st.set_page_config(
    page_title="Airline Passenger Satisfaction Prediction",
    page_icon="✈️",
    layout="wide"
)

# ==========================================================
# LOAD MODEL
# ==========================================================

logistic_model = joblib.load("../model/logistic_regression.pkl")
decision_tree_model = joblib.load("../model/decision_tree.pkl")
random_forest_model = joblib.load("../model/random_forest.pkl")
xgboost_model = joblib.load("../model/xgboost.pkl")

# ==========================================================
# LOAD ENCODER & SCALER
# ==========================================================

encoders = joblib.load("../model/label_encoders.pkl")
scaler = joblib.load("../model/standard_scaler.pkl")

# ==========================================================
# CSS
# ==========================================================

st.markdown("""
<style>

.main{
    background-color:#F4F7FB;
}

.block-container{
    padding-top:2rem;
    padding-bottom:2rem;
}

h1{
    color:#0F4C81;
    text-align:center;
    font-weight:bold;
}

h2,h3{
    color:#2C3E50;
}

[data-testid="stMetric"]{

    background:white;

    border-radius:15px;

    padding:18px;

    box-shadow:0px 3px 12px rgba(0,0,0,.12);

    border-left:6px solid #2E86DE;

}

div[data-testid="stDataFrame"]{

    background:white;

    border-radius:15px;

    padding:10px;

    box-shadow:0px 3px 12px rgba(0,0,0,.10);

}

.stButton>button{

    width:100%;

    height:55px;

    background:#2E86DE;

    color:white;

    border-radius:10px;

    font-size:18px;

    font-weight:bold;

    border:none;

}

.stButton>button:hover{

    background:#1B4F72;

    color:white;

}

</style>
""",unsafe_allow_html=True)

# ==========================================================
# FUNGSI MENGUBAH HASIL PREDIKSI
# ==========================================================

def prediction_label(value):
    return "😊 Satisfied" if value == 1 else "😞 Dissatisfied"


# ==========================================================
# HEADER
# ==========================================================

st.markdown("""
# ✈ Airline Passenger Satisfaction Prediction

### Machine Learning Dashboard

Prediksi kepuasan penumpang menggunakan **4 algoritma Machine Learning**
""")

st.divider()

col1,col2,col3,col4 = st.columns(4)

col1.metric("Dataset","129.880")

col2.metric("Feature","21")

col3.metric("Model","4")

col4.metric("Best Accuracy","95.80%")

st.divider()


# ==========================================================
# LAYOUT
# ==========================================================

left,right = st.columns([1.3,1])


# ==========================================================
# INPUT DATA PENUMPANG
# ==========================================================

with left:

    st.subheader("📝 Input Data Penumpang")

    col1,col2 = st.columns(2)

    with col1:

        customer_type = st.selectbox(
            "Customer Type",
            ["Loyal Customer","Disloyal Customer"]
        )

        age = st.number_input(
            "Age",
            7,
            85,
            35
        )

        travel_type = st.selectbox(
            "Type of Travel",
            ["Business travel","Personal Travel"]
        )

        travel_class = st.selectbox(
            "Class",
            ["Business","Eco","Eco Plus"]
        )

        flight_distance = st.number_input(
            "Flight Distance",
            50,
            7000,
            1500
        )

        departure_delay = st.number_input(
            "Departure Delay",
            0,
            2000,
            0
        )

    with col2:

        arrival_delay = st.number_input(
            "Arrival Delay",
            0,
            2000,
            0
        )

        seat_comfort = st.slider(
            "Seat Comfort",
            0,
            5,
            4
        )

        departure_arrival = st.slider(
            "Departure / Arrival Time",
            0,
            5,
            4
        )

        food_drink = st.slider(
            "Food and Drink",
            0,
            5,
            4
        )

        gate_location = st.slider(
            "Gate Location",
            0,
            5,
            4
        )

        inflight_wifi = st.slider(
            "Inflight Wifi",
            0,
            5,
            4
        )

        st.subheader("⭐ Penilaian Pelayanan")

    col3,col4 = st.columns(2)

    with col3:

        inflight_entertainment = st.slider(
            "Inflight Entertainment",
            0,
            5,
            4
        )

        online_support = st.slider(
            "Online Support",
            0,
            5,
            4
        )

        online_booking = st.slider(
            "Online Booking",
            0,
            5,
            4
        )

        onboard_service = st.slider(
            "On-board Service",
            0,
            5,
            4
        )

        leg_room = st.slider(
            "Leg Room",
            0,
            5,
            4
        )

    with col4:

        baggage = st.slider(
            "Baggage Handling",
            1,
            5,
            4
        )

        checkin = st.slider(
            "Check-in Service",
            0,
            5,
            4
        )

        cleanliness = st.slider(
            "Cleanliness",
            0,
            5,
            4
        )

        online_boarding = st.slider(
            "Online Boarding",
            0,
            5,
            4
        )

        st.divider()

    predict = st.button(
        "🚀 Prediksi",
        use_container_width=True
    )



# ==========================================================
# PREDIKSI
# ==========================================================

if predict:

    # ------------------------------------------
    # Encoding data kategorikal
    # ------------------------------------------

    customer_type_encode = encoders["Customer Type"].transform(
        [customer_type]
    )[0]

    travel_type_encode = encoders["Type of Travel"].transform(
        [travel_type]
    )[0]

    travel_class_encode = encoders["Class"].transform(
        [travel_class]
    )[0]

    # ------------------------------------------
    # Membuat DataFrame
    # ------------------------------------------

    input_df = pd.DataFrame({

        "Customer Type":[customer_type_encode],
        "Age":[age],
        "Type of Travel":[travel_type_encode],
        "Class":[travel_class_encode],
        "Flight Distance":[flight_distance],
        "Seat comfort":[seat_comfort],
        "Departure/Arrival time convenient":[departure_arrival],
        "Food and drink":[food_drink],
        "Gate location":[gate_location],
        "Inflight wifi service":[inflight_wifi],
        "Inflight entertainment":[inflight_entertainment],
        "Online support":[online_support],
        "Ease of Online booking":[online_booking],
        "On-board service":[onboard_service],
        "Leg room service":[leg_room],
        "Baggage handling":[baggage],
        "Checkin service":[checkin],
        "Cleanliness":[cleanliness],
        "Online boarding":[online_boarding],
        "Departure Delay in Minutes":[departure_delay],
        "Arrival Delay in Minutes":[arrival_delay]

    })

    # ------------------------------------------
    # Logistic Regression
    # ------------------------------------------

    input_scaled = scaler.transform(input_df)

    lr_predict = logistic_model.predict(input_scaled)[0]

    lr_probability = logistic_model.predict_proba(
        input_scaled
    )[0].max()


    # ------------------------------------------
    # Decision Tree
    # ------------------------------------------

    dt_predict = decision_tree_model.predict(input_df)[0]

    dt_probability = decision_tree_model.predict_proba(
        input_df
    )[0].max()

    
    # ------------------------------------------
    # Random Forest
    # ------------------------------------------

    rf_predict = random_forest_model.predict(input_df)[0]

    rf_probability = random_forest_model.predict_proba(
        input_df
    )[0].max()


    # ------------------------------------------
    # XGBoost
    # ------------------------------------------

    xgb_predict = xgboost_model.predict(input_df)[0]

    xgb_probability = xgboost_model.predict_proba(
        input_df
    )[0].max()


# ==========================================================
# DASHBOARD HASIL
# ==========================================================

with right:

    st.subheader("📊 Hasil Prediksi")

    if predict:
        
        st.subheader("📋 Ringkasan Input")

        summary = pd.DataFrame({

            "Parameter":[
                "Customer Type",
                "Age",
                "Type of Travel",
                "Class",
                "Flight Distance"
            ],

            "Value":[
                customer_type,
                age,
                travel_type,
                travel_class,
                flight_distance
            ]

        })

        st.dataframe(
            summary,
            use_container_width=True,
            hide_index=True
        )

        st.divider()
        col1, col2 = st.columns(2)

        with col1:

            st.success("🤖 Logistic Regression")

            st.metric(
                "Prediction",
                prediction_label(lr_predict),
                f"{lr_probability*100:.2f}%"
            )

            st.progress(float(lr_probability))

            st.markdown("---")

            st.success("🌳 Decision Tree")

            st.metric(
                "Prediction",
                prediction_label(dt_predict),
                f"{dt_probability*100:.2f}%"
            )

            st.progress(float(dt_probability))

        with col2:

            st.success("🌲 Random Forest")

            st.metric(
                "Prediction",
                prediction_label(rf_predict),
                f"{rf_probability*100:.2f}%"
            )

            st.progress(float(rf_probability))

            st.markdown("---")

            st.success("🏆 XGBoost")

            st.metric(
                "Prediction",
                prediction_label(xgb_predict),
                f"{xgb_probability*100:.2f}%"
            )

            st.progress(float(xgb_probability))

        st.divider()

        st.info("""
        🏆 **Model Terbaik**

        XGBoost dipilih sebagai model terbaik karena memiliki:

        - Accuracy : **95.80%**
        - Precision : **97.03%**
        - Recall : **95.25%**
        - F1 Score : **96.13%**
        """)

        st.subheader("📈 Performa Model")

        performance = pd.DataFrame({

            "Model":[
                "Logistic Regression",
                "Decision Tree",
                "Random Forest",
                "XGBoost"
            ],

            "Accuracy":[82.89,93.23,95.58,95.80],
            "Precision":[84.45,93.77,96.89,97.03],
            "Recall":[84.25,93.86,94.97,95.25],
            "F1 Score":[84.35,93.82,95.92,96.13]

        })

        st.dataframe(
            performance,
            use_container_width=True,
            hide_index=True
        )
        st.info("""
        Grafik berikut menunjukkan perbandingan performa
        keempat algoritma Machine Learning berdasarkan
        hasil evaluasi model.
        """)

        st.subheader("📊 Grafik Accuracy")
        chart_data = performance.set_index("Model")
        st.bar_chart(
            chart_data["Accuracy"]
        )
        st.subheader("📈 Grafik F1 Score")
        st.bar_chart(
            chart_data["F1 Score"]
        )

    else:

        st.info(
            "Silakan isi data penumpang kemudian tekan tombol Prediksi."
        )

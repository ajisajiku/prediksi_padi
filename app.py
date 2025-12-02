import streamlit as st
import pandas as pd
import joblib

# =========================================
# LOAD MODEL & ENCODER TARGET SAJA
# =========================================
model = joblib.load("model_random_forest_variety.pkl")
le_y = joblib.load("encoder_target.pkl")

# =========================================
# JUDUL APLIKASI
# =========================================
st.title("🌾 Prediksi Variety Padi - Random Forest")
st.write("Silakan input data sesuai karakteristik lahan & pemupukan:")

# =========================================
# INPUT FITUR (14 FITUR)
# =========================================

trash = st.number_input("Trash (bundles)", min_value=0.0)
yield_padi = st.number_input("Paddy Yield (Kg)", min_value=0.0)

soil = st.selectbox("Soil Types", ["alluvial", "clay"])
nursery = st.selectbox("Nursery", ["dry", "wet"])

lp_nursery = st.number_input("LP Nursery Area (Tonnes)", min_value=0.0)
seedrate = st.number_input("Seedrate (Kg)", min_value=0.0)
nursery_area = st.number_input("Nursery Area (Cents)", min_value=0.0)
micro = st.number_input("Micronutrients 70 Days", min_value=0.0)
weed = st.number_input("Weed 28D Thiobencarb", min_value=0.0)
urea = st.number_input("Urea 40 Days", min_value=0.0)
potash = st.number_input("Potash 50 Days", min_value=0.0)
lp_mainfield = st.number_input("LP Mainfield (Tonnes)", min_value=0.0)
pest = st.number_input("Pest 60 Day (ml)", min_value=0.0)
dap = st.number_input("DAP 20 Days", min_value=0.0)

# =========================================
# TOMBOL PREDIKSI
# =========================================
if st.button("🔍 Prediksi Variety"):

    # ================================
    # ENCODING MANUAL (ANTI ERROR)
    # ================================
    if soil == "alluvial":
        soil_val = 0
    else:  # clay
        soil_val = 1

    if nursery == "dry":
        nursery_val = 0
    else:  # wet
        nursery_val = 1

    # ================================
    # DATAFRAME SESUAI URUTAN TRAINING
    # ================================
    data_input = pd.DataFrame([[  
        trash,
        yield_padi,
        soil_val,
        nursery_val,
        lp_nursery,
        seedrate,
        nursery_area,
        micro,
        weed,
        urea,
        potash,
        lp_mainfield,
        pest,
        dap
    ]], columns=[
        "Trash(in bundles)",
        "Paddy yield(in Kg)",
        "Soil Types",
        "Nursery",
        "LP_nurseryarea(in Tonnes)",
        "Seedrate(in Kg)",
        "Nursery area (Cents)",
        "Micronutrients_70Days",
        "Weed28D_thiobencarb",
        "Urea_40Days",
        "Potassh_50Days",
        "LP_Mainfield(in Tonnes)",
        "Pest_60Day(in ml)",
        "DAP_20days"
    ])

    # ================================
    # PREDIKSI
    # ================================
    hasil = model.predict(data_input)
    hasil_label = le_y.inverse_transform(hasil)

    st.success(f"✅ Hasil Prediksi Variety: **{hasil_label[0]}**")

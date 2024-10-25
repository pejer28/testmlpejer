import streamlit as st
import pandas as pd
import pickle
import os
from streamlit_option_menu import option_menu

with st.sidebar:
    selected = option_menu('Desain Steamlit UTS ML 24/25',
                           ['Klasifikasi',
                            'Regresi'],
                            default_index=0)

if selected == 'Klasifikasi':
    st.title('Klasifikasi')

    file = st.file_uploader("Masukkan File", type = ["csv", "txt"])
    if file is not None:
        input_data = pd.read_csv(file)

        model_path = r'BestModel_CLF_GBT_pytorch.pkl'

        if os.path.exists(model_path):
            with open(model_path, 'rb') as f:
                loaded_model = pickle.load(f)

            scaler = loaded_model[0]
            feature_selector = loaded_model[1]
            GBT_model = loaded_model[2]

            squaremeters = st.number_input("Square Meters", 0)
            numberofrooms = st.slider("Number of Rooms", 0, 100)
            hasyard = st.radio("Has Yard?",["Yes", "No"])
            floors = st.slider("Number of Floors", 0, 100)
            citycode = st.number_input("City Code", 0)
            citypartrange = st.slider("City Part Range", 0, 10)
            numprevowners = st.slider("Number of Previous Owner", 0, 10)
            made = st.number_input("Year Made", 0)
            isnewbuilt = st.radio("Is New Built?",["New", "Old"])
            hasstromprotector = st.radio("Has Storm Protector?",["Yes", "No"])
            basement = st.number_input("Basement", 0)
            attic = st.number_input("Attic", 0)
            garage = st.number_input("Garage", 0)
            hasstorage = st.radio("Has Storage?",["Yes", "No"])
            hasguestroom = st.slider("Has Guest Room", 0, 10)

            if hasyard == "Yes":
                input_hasyard_Yes = 1
                input_hasyard_No = 0
            elif hasyard == "No":
                input_hasyard_Yes = 0
                input_hasyard_No = 1

            if isnewbuilt == "New":
                input_isnewbuilt_New = 1
                input_isnewbuilt_Old = 0
            elif isnewbuilt == "No":
                input_isnewbuilt_New = 0
                input_isnewbuilt_Old = 1
            
            if hasstromprotector == "Yes":
                input_hasstormprotector_Yes = 1
                input_hasstormprotector_No = 0
            elif hasyard == "No":
                input_hasstormprotector_Yes = 0
                input_hasstormprotector_No = 1

            if hasstorage == "Yes":
                input_hasstorage_Yes = 1
                input_hasstorage_No = 0
            elif hasyard == "No":
                input_hasstorage_Yes = 0
                input_hasstorage_No = 1
        
            input_data = [input_hasyard_Yes, input_hasyard_No, input_isnewbuilt_New, input_isnewbuilt_Old, 
                          input_hasstormprotector_Yes, input_hasstormprotector_No, input_hasstorage_Yes, input_hasstorage_No,
                          squaremeters, numberofrooms, floors, citycode, citypartrange, numprevowners,
                          made, basement, attic, garage, hasguestroom]
        
            input_data_scaled = scaler.transform
            input_data_selected = feature_selector.transform(input_data_scaled)

        if st.button("Prediksi"):
            GBT_model_prediction = GBT_model.predict(input_data)
            outcome = {0: 'Basic', 1: 'Middle', 2:'Luxury'}
            st.write(f"Bangunan tersebut masuk ke Kategori: **{outcome[GBT_model_prediction[0]]}**")
    else:
        st.error("Model tidak ditemukan, silahkan cek file model di direktori...")

if selected == 'Regresi':
    st.title('Regresi')

    file = st.file_uploader("Masukkan File", type = ["csv", "txt"])
    if file is not None:
        input_data = pd.read_csv(file)

        model_path = r'BestModel_REG_RFR_pytorch.pkl'

        if os.path.exists(model_path):
            with open(model_path, 'rb') as f:
                loaded_model = pickle.load(f)

            scaler = loaded_model[0]
            feature_selector = loaded_model[1]
            RFR_model = loaded_model[2]

            squaremeters = st.number_input("Square Meters", 0)
            numberofrooms = st.slider("Number of Rooms", 0, 100)
            hasyard = st.radio("Has Yard?",["Yes", "No"])
            floors = st.slider("Number of Floors", 0, 100)
            citycode = st.number_input("City Code", 0)
            citypartrange = st.slider("City Part Range", 0, 10)
            numprevowners = st.slider("Number of Previous Owner", 0, 10)
            made = st.number_input("Year Made", 0)
            isnewbuilt = st.radio("Is New Built?",["New", "Old"])
            hasstromprotector = st.radio("Has Storm Protector?",["Yes", "No"])
            basement = st.number_input("Basement", 0)
            attic = st.number_input("Attic", 0)
            garage = st.number_input("Garage", 0)
            hasstorage = st.radio("Has Storage?",["Yes", "No"])
            hasguestroom = st.slider("Has Guest Room", 0, 10)

            if hasyard == "Yes":
                input_hasyard_Yes = 1
                input_hasyard_No = 0
            elif hasyard == "No":
                input_hasyard_Yes = 0
                input_hasyard_No = 1

            if isnewbuilt == "New":
                input_isnewbuilt_New = 1
                input_isnewbuilt_Old = 0
            elif isnewbuilt == "No":
                input_isnewbuilt_New = 0
                input_isnewbuilt_Old = 1
            
            if hasstromprotector == "Yes":
                input_hasstormprotector_Yes = 1
                input_hasstormprotector_No = 0
            elif hasyard == "No":
                input_hasstormprotector_Yes = 0
                input_hasstormprotector_No = 1

            if hasstorage == "Yes":
                input_hasstorage_Yes = 1
                input_hasstorage_No = 0
            elif hasyard == "No":
                input_hasstorage_Yes = 0
                input_hasstorage_No = 1
        
            input_data = [input_hasyard_Yes, input_hasyard_No, input_isnewbuilt_New, input_isnewbuilt_Old, 
                          input_hasstormprotector_Yes, input_hasstormprotector_No, input_hasstorage_Yes, input_hasstorage_No,
                          squaremeters, numberofrooms, floors, citycode, citypartrange, numprevowners,
                          made, basement, attic, garage, hasguestroom]
        
            input_data_scaled = scaler.transform
            input_data_selected = feature_selector.transform(input_data_scaled)

        if st.sidebar.button("Prediksi"):
            RFR_model_predict = RFR_model.predict(input_data_selected)
            st.write(f">Prediksi Harga adalah: {RFR_model_predict[0]:.2f}")
    else:
        st.error("Model tidak ditemukan, silahkan cek file model di direktori...")

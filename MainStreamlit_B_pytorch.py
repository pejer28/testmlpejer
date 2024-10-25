import streamlit as st
import pandas as pd
import pickle
import os
import numpy as np
from streamlit_option_menu import option_menu
import sklearn



# model_path = 'BestModel_CLF_GBT_pytorch.pkl'

# # Cek apakah file ada dan bisa dibaca
# if os.path.exists(model_path):
#     st.write("Model file ditemukan.")
#     if os.access(model_path, os.R_OK):
#         st.write("File model memiliki izin baca.")
#     else:
#         st.write("File model **tidak memiliki izin baca**.")
# else:
#     st.write("Model file **tidak ditemukan**.")


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
        st.write("<h3 style='text-align: center; color: #0073e6;'>Data yang diupload :</h3>", unsafe_allow_html=True)
        st.dataframe(input_data)
    else:
        st.error("Model BestModel_CLF_GBT_pytorch tidak ditemukan, silahkan cek file model di direktori...")

    model_path = 'BestModel_CLF_GBT_pytorch.pkl'

    if os.path.exists(model_path):
        with open(model_path, 'rb') as f:
            loaded_model = pickle.load(f)

        # scaler = loaded_model[0]
        # feature_selector = loaded_model[1]
        GBT_model = loaded_model
           

        squaremeters = st.number_input("Square Meters", 0)
        numberofrooms = st.slider("Number of Rooms", 0, 100)
        hasyard = st.radio("Has Yard?",["Yes", "No"])
        haspool = st.radio("Has Pool?",["Yes", "No"])
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
        else:
            input_hasyard_Yes = 0
            input_hasyard_No = 1

        if haspool == "Yes":
            input_haspool_Yes = 1
            input_haspool_No = 0
        else:
            input_haspool_Yes = 0
            input_haspool_No = 1

        if isnewbuilt == "New":
            input_isnewbuilt_New = 1
            input_isnewbuilt_Old = 0
        else:
            input_isnewbuilt_New = 0
            input_isnewbuilt_Old = 1
            
        if hasstromprotector == "Yes":
            input_hasstormprotector_Yes = 1
            input_hasstormprotector_No = 0
        else:
            input_hasstormprotector_Yes = 0
            input_hasstormprotector_No = 1

        if hasstorage == "Yes":
            input_hasstorage_Yes = 1
            input_hasstorage_No = 0
        else:
            input_hasstorage_Yes = 0
            input_hasstorage_No = 1

        
        input_data = np.array([input_hasyard_Yes, input_hasyard_No, input_haspool_Yes, input_haspool_No, 
                        input_isnewbuilt_New, input_isnewbuilt_Old, 
                        input_hasstormprotector_Yes, input_hasstormprotector_No, input_hasstorage_Yes, input_hasstorage_No,
                        squaremeters, numberofrooms, floors, citycode, citypartrange, numprevowners,
                        made, basement, attic, garage, hasguestroom]).reshape(1, -1)
        
        # input_data_scaled = scaler.transform([input_data])
        # input_data_selected = feature_selector.transform(input_data_scaled)
        print(f"Jumlah fitur input_data: {input_data.shape[1]}")  # Harus 21
        print(sklearn.__version__)
        if st.button("Prediksi"):
            if st.button("Prediksi"):
                GBT_model_prediction = GBT_model.predict(input_data)
                try:
                    # Pastikan konversi prediksi ke integer
                    prediksi_kategori = int(GBT_model_prediction[0])
                    outcome = {0: 'Basic', 1: 'Middle', 2: 'Luxury'}
                    
                    # Pastikan hasil prediksi masuk kategori
                    if prediksi_kategori in outcome:
                        st.write(f"Bangunan tersebut masuk ke Kategori: **{outcome[prediksi_kategori]}**")
                    else:
                        st.error("Prediksi tidak valid atau tidak termasuk dalam kategori.")
                except (ValueError, IndexError):
                    st.error("Prediksi tidak dapat dikonversi ke kategori yang diharapkan.")


            
            st.write(f"Bangunan tersebut masuk ke Kategori: **{outcome[GBT_model_prediction[0]]}**")

            

     

        
if selected == 'Regresi':
    st.title('Regresi')

    file = st.file_uploader("Masukkan File", type = ["csv", "txt"])
    if file is not None:
        input_data = pd.read_csv(file)
        st.write("<h3 style='text-align: center; color: #0073e6;'>Data yang diupload :</h3>", unsafe_allow_html=True)
        st.dataframe(input_data)
    else:
        st.error("Model BestModel_REG_RFR_pytorch tidak ditemukan, silahkan cek file model di direktori...")

    model_path = 'BestModel_REG_RFR_pytorch.pkl'
    
    if os.path.exists(model_path):
        with open(model_path, 'rb') as f:
            loaded_model = pickle.load(f)

            scaler = loaded_model[0]
            feature_selector = loaded_model[1]
            RFR_model = loaded_model[2]

        squaremeters = st.number_input("Square Meters", 0)
        numberofrooms = st.slider("Number of Rooms", 0, 100)
        hasyard = st.radio("Has Yard?",["Yes", "No"])
        haspool = st.radio("Has Pool?",["Yes", "No"])
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

        if haspool == "Yes":
            input_haspool_Yes = 1
            input_haspool_No = 0
        elif haspool == "No":
            input_haspool_Yes = 0
            input_haspool_No = 1

        if isnewbuilt == "New":
            input_isnewbuilt_New = 1
            input_isnewbuilt_Old = 0
        elif isnewbuilt == "No":
            input_isnewbuilt_New = 0
            input_isnewbuilt_Old = 1
            
        if hasstromprotector == "Yes":
            input_hasstormprotector_Yes = 1
            input_hasstormprotector_No = 0
        elif hasstromprotector == "No":
            input_hasstormprotector_Yes = 0
            input_hasstormprotector_No = 1

        if hasstorage == "Yes":
            input_hasstorage_Yes = 1
            input_hasstorage_No = 0
        elif hasstorage == "No":
            input_hasstorage_Yes = 0
            input_hasstorage_No = 1
        
        input_data = [input_hasyard_Yes, input_hasyard_No, input_haspool_Yes, input_haspool_No, input_isnewbuilt_New, input_isnewbuilt_Old, 
                        input_hasstormprotector_Yes, input_hasstormprotector_No, input_hasstorage_Yes, input_hasstorage_No,
                        squaremeters, numberofrooms, floors, citycode, citypartrange, numprevowners,
                        made, basement, attic, garage, hasguestroom]
        
        input_data_scaled = scaler.transform([input_data])
        input_data_selected = feature_selector.transform(input_data_scaled)

        if st.button("Prediksi"):
                RFR_model_predict = RFR_model.predict(input_data_selected)
                st.write(f">Prediksi Harga adalah: {RFR_model_predict[0]:.2f}")
    

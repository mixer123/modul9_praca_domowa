import os
import json
import pandas as pd
import boto3
from io import BytesIO
from dotenv import load_dotenv
from datetime import datetime
import re
from faker import Faker
import numpy as np
import tempfile
import streamlit as st
from pycaret.regression import setup, compare_models, tune_model, save_model, load_model, predict_model, create_model
from openai import OpenAI

# =========================
# 🔑 Konfiguracja i stałe
# =========================
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DO_SPACE_NAME = 'pracadomowamodul9'
DO_ENDPOINT_URL = 'https://fra1.digitaloceanspaces.com'
DO_ACCESS_KEY = os.getenv('aws_access_key_id')
DO_SECRET_KEY = os.getenv('aws_secret_access_key')
DATA_FILE_2023 = 'halfmarathon_wroclaw_2023__final.csv'
DATA_FILE_2024 = 'halfmarathon_wroclaw_2024__final.csv'
MODEL_FILE = "models/best_gbr_model.pkl"

REQUIRED_COLS = ["Czas", "Rocznik", "Płeć", "5 km Tempo", "10 km Tempo", "15 km Tempo"]

# =========================
# 🚀 Klienci
# =========================
s3_client = boto3.client(
    's3',
    region_name="fra1",
    endpoint_url=DO_ENDPOINT_URL,
    aws_access_key_id=DO_ACCESS_KEY,
    aws_secret_access_key=DO_SECRET_KEY
)

session = boto3.session.Session()
client_do = session.client('s3',
    region_name='fra1',  
    endpoint_url=DO_ENDPOINT_URL,
    aws_access_key_id=DO_ACCESS_KEY,
    aws_secret_access_key=DO_SECRET_KEY
)

session = boto3.session.Session()


openai_client = OpenAI(api_key=OPENAI_API_KEY)

# =========================
# 🧩 Utils
# =========================
def validate_csv(df: pd.DataFrame) -> bool:
    missing = [col for col in REQUIRED_COLS if col not in df.columns]
    if missing:
        st.error(f"❌ Brak wymaganych kolumn w pliku CSV: {', '.join(missing)}")
        st.info("Wymagane kolumny: " + ", ".join(REQUIRED_COLS))
        return False
    return True

def convert_time_to_seconds(time):
    if pd.isnull(time) or time in ['DNS', 'DNF']:
        return None
    time = time.split(':')
    return int(time[0]) * 3600 + int(time[1]) * 60 + int(time[2])

def list_files(prefix: str = "models/"):
    try:
        response = s3_client.list_objects_v2(Bucket=DO_SPACE_NAME, Prefix=prefix)
        if "Contents" not in response:
            st.info("📭 Brak plików w przestrzeni DO dla podanego prefiksu.")
            return []
        return [obj["Key"] for obj in response["Contents"]]
    except Exception as e:
        st.error(f"❌ Błąd pobierania listy plików: {e}")
        return []

def delete_file_from_do(file_key: str):
    try:
        s3_client.delete_object(Bucket=DO_SPACE_NAME, Key=file_key)
        st.success(f"🗑️ Plik {file_key} został usunięty z DigitalOcean")
    except Exception as e:
        st.error(f"❌ Błąd usuwania pliku {file_key}: {e}")

def download_model():
    if not all([DO_SPACE_NAME, DO_ENDPOINT_URL, DO_ACCESS_KEY, DO_SECRET_KEY]):
        st.error("❌ Brak kluczy DO lub nazwy przestrzeni.")
        return None
    try:
        response = s3_client.list_objects_v2(Bucket=DO_SPACE_NAME, Prefix=MODEL_FILE)
        if "Contents" not in response:
            st.warning(f"⚠️ Plik {MODEL_FILE} nie istnieje w przestrzeni {DO_SPACE_NAME}.")
            return None
        with BytesIO() as data:
            s3_client.download_fileobj(DO_SPACE_NAME, MODEL_FILE, data)
            data.seek(0)
            with open("best_gbr_model.pkl", "wb") as f:
                f.write(data.read())
        st.success(f"✅ Model {MODEL_FILE} pobrany z DigitalOcean")
        return load_model("best_gbr_model")
    except Exception as e:
        st.error(f"❌ Błąd pobierania modelu: {e}")
        return None

def train_model(df_train: pd.DataFrame):
    try:
        if not pd.api.types.is_numeric_dtype(df_train["Czas"]):
            df_train["Czas"] = pd.to_numeric(df_train["Czas"], errors="coerce")
        df_train = df_train.dropna(subset=["Czas"])
        exp = setup(df_train, target="Czas", silent=True, session_id=123)
        best = compare_models()
        tuned = tune_model(best)
        save_model(tuned, "best_gbr_model")
        s3_client.upload_file("best_gbr_model.pkl", DO_SPACE_NAME, MODEL_FILE)
        st.success("✅ Model wytrenowany i zapisany w DigitalOcean")
    except Exception as e:
        st.error("❌ Błąd podczas trenowania modelu.")
        st.exception(e)

def parse_pace(text):
    if isinstance(text, (int, float)):
        return float(text)
    if not isinstance(text, str):
        raise ValueError("Nieznany format tempa.")
    t = text.strip().lower().replace("/km", "").replace("min/km", "").replace("min", "").replace(",", ".")
    if ":" in t:
        mm, ss = t.split(":")
        return float(mm) + float(ss)/60.0
    return float(t)

def extract_data_with_llm(text):
    prompt = f"""
Użytkownik podał dane: {text}
Zwróć TYLKO poprawny JSON, bez dodatkowego tekstu.
Klucze: "Wiek", "Płeć", "5 km Tempo", "10 km Tempo", "15 km Tempo"
"""
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": "Zwracaj wyłącznie poprawny JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )
        raw_text = response.choices[0].message.content
        json_str_match = re.search(r'\{.*\}', raw_text, re.DOTALL)
        if not json_str_match:
            raise ValueError("Brak obiektu JSON w odpowiedzi modelu.")
        return json.loads(json_str_match.group())
    except Exception as e:
        st.error("⚠️ Nie udało się sparsować odpowiedzi LLM do JSON.")
        st.exception(e)
        return None

# =========================
# 🎯 Aplikacja
# =========================
def main():
    st.title("🏃‍♂️ Predykcja czasu biegu")
    tab1, tab2 = st.tabs(["🔮 Predykcja", "⚙️ Trenowanie"])

    # --- Zakładka Predykcja ---
    with tab1:
        user_input = st.text_area(
            "Podaj dane (wiek, płeć, tempo na 5km, 10km, 15km):",
            placeholder="np. Mam 35 lat, jestem mężczyzną, biegam 5 km w tempie 5:20/km..."
        )
        col_a, col_b = st.columns(2)
        with col_a:
            predict_btn = st.button("Przewiduj czas")
        with col_b:
            reload_model = st.button("🔄 Pobierz/odśwież model z DO")

        model = None
        if reload_model:
            model = download_model()

        if predict_btn:
            if not user_input.strip():
                st.error("⚠️ Podaj dane wejściowe.")
                return
            data = extract_data_with_llm(user_input)
            if not data:
                return
            required_keys = ["Wiek", "Płeć", "5 km Tempo", "10 km Tempo", "15 km Tempo"]
            if not all(k in data for k in required_keys):
                st.error("⚠️ Brak wszystkich wymaganych danych w JSON.")
                st.json(data)
                return
            try:
                birth_year = datetime.now().year - int(data["Wiek"])
                gender = 1 if str(data["Płeć"]).strip().upper().startswith("M") else 0
                df_input = pd.DataFrame([{
                    "Rocznik": birth_year,
                    "Płeć": gender,
                    "5 km Tempo": parse_pace(data["5 km Tempo"]),
                    "10 km Tempo": parse_pace(data["10 km Tempo"]),
                    "15 km Tempo": parse_pace(data["15 km Tempo"])
                }])
            except Exception as e:
                st.error(f"❌ Problem z przetwarzaniem danych wejściowych: {e}")
                return

            if model is None:
                model = download_model()
            if model:
                try:
                    prediction = predict_model(model, data=df_input)
                    st.success(f"⏱️ Przewidywany czas biegu: {prediction['prediction_label'].iloc[0]:.2f} minut")
                    with st.expander("Dane wejściowe użyte do predykcji"):
                        st.dataframe(df_input)
                except Exception as e:
                    st.error("❌ Błąd podczas predykcji.")
                    st.exception(e)
            else:
                st.warning("⚠️ Brak modelu – wytrenuj go w zakładce Trenowanie.")

    # --- Zakładka Trenowanie ---
    with tab2:
        st.subheader("📂 Wgraj dane treningowe (CSV)")
        uploaded_files = st.file_uploader("Wybierz pliki CSV", type=["csv"], accept_multiple_files=True)
        if uploaded_files:
            dfs = []
            for uploaded_file in uploaded_files:
                try:
                    df = pd.read_csv(uploaded_file)
                    dfs.append(df)
                    st.success(f"📄 Wczytano plik: {uploaded_file.name} ({df.shape[0]} rekordów)")
                    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                            tmp_file.write(uploaded_file.getbuffer())
                            tmp_file_path = tmp_file.name
                        
                        # Upload do DO
                    object_name = f"{uploaded_file.name}"
                    client_do.upload_file(tmp_file_path, DO_SPACE_NAME, object_name)
                    st.success(f"✅ Plik {uploaded_file.name} wysłany do DigitalOcean")
                    
                except Exception as e:
                    st.error(f"⚠️ Nie udało się wczytać pliku {uploaded_file.name}")
                    st.exception(e)

            # if st.button("⬆️ Upload wszystkich plików do DigitalOcean"):
            #     for uploaded_file in uploaded_files:
            #         try:
            #             # Tworzymy plik tymczasowy
            #             with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            #                 tmp_file.write(uploaded_file.getbuffer())
            #                 tmp_file_path = tmp_file.name
                        
            #             # Upload do DO
            #             object_name = f"{uploaded_file.name}"
            #             client_do.upload_file(tmp_file_path, DO_SPACE_NAME, object_name)
            #             st.success(f"✅ Plik {uploaded_file.name} wysłany do DigitalOcean")
                        
            #             # Opcjonalnie usuń tymczasowy plik
            #             os.remove(tmp_file_path)
            
            #         except Exception as e:
            #             st.error(f"❌ Błąd uploadu pliku {uploaded_file.name}")
            #             st.exception(e)

            if dfs:
                df_train = pd.concat(dfs, ignore_index=True)
                st.info(f"📊 Łącznie {df_train.shape[0]} rekordów z {len(dfs)} plików")
                with st.expander("Podgląd połączonych danych"):
                    st.dataframe(df_train.head())
                obj_2023 = s3_client.get_object(Bucket=DO_SPACE_NAME, Key=DATA_FILE_2023)
                df_2023 = pd.read_csv(BytesIO(obj_2023['Body'].read()),sep=';')
                obj_2024 = s3_client.get_object(Bucket=DO_SPACE_NAME, Key=DATA_FILE_2024)
                df_2024 = pd.read_csv(BytesIO(obj_2024['Body'].read()),sep=';')

                # Utworzenie kolumny Rok
                df_2023['Rok']=2023
                df_2024['Rok']=2024

                # Wypełniam puste miejsca
                df = pd.concat([df_2023, df_2024], ignore_index=True)
                # Wybieram tylko te rekordy które nie maja nan w kolumnie Czas czyli ci co ukonczyli bieg
                df = df[df['Czas'].notna()]

                def fill_empty(name_column):
                    if 'Miejsce' in name_column:        
                        msc_sex_fill = list(df[name_column].dropna())
                        msc_sex_all = list(range(1, len(df) + 1))
                        msc_sex_to_fill = list(set(msc_sex_all) - set(msc_sex_fill))
                        mask = df[name_column].isna()
                        df.loc[mask, name_column] = np.random.choice(msc_sex_to_fill, size=mask.sum(), replace=False)
                        df[name_column] = df[name_column].astype(int)
                    else:
                        
                        if df[name_column].dtype == 'object':
                            df[name_column] = df[name_column].apply(convert_time_to_seconds)       
                        srednia = round(df[name_column].mean(skipna=True))       
                        df[name_column] = df[name_column].fillna(srednia)        
                        if (df[name_column] % 1 == 0).all():
                            df[name_column] = df[name_column].fillna(round(srednia)).astype(int)

                list_columns = [df.columns[x] for x in range(12,24)]

                for col in list_columns:    
                    fill_empty(col)

                # Pomijam 4 kolumny na pewno nie maja wpływu na czas
                df = df.drop(['Numer startowy','Miasto','Kraj','Drużyna','Miejsce','Płeć Miejsce','Kategoria wiekowa Miejsce'],axis=1)
               
                # Usuwanie rekordów w których nie ma płci
                index_ = list(df[df['Płeć'].isnull()].index)
                df = df.drop(index_, axis=0)
                # Anonimmizacja
                fake = Faker()
                df['Imię'] = [fake.first_name() for _ in range(len(df['Imię']))]
                df['Nazwisko'] = [fake.last_name() for _ in range(len(df['Nazwisko']))]

               

                # Zamiana wartości kategorycznych w kolumnie Płeć
                mapping = {'M': 1, 'K': 0}
                df['Płeć'] = df['Płeć'].replace(mapping)
                # Usuwam rekordy które nie mają kategorii wiekowej
                index_ = list(df[df['Kategoria wiekowa'].isnull()].index)
                df = df.drop(index=index_)
                # Wypełnianie Nan w kolumnie Rocznik
                df['Rocznik'] = df['Rocznik'].fillna(
                    df.groupby('Kategoria wiekowa')['Rocznik'].transform('mean').round()
                )

                df['Rocznik']=df['Rocznik'].astype(int)

                # Wstawić średnie tempo stabilnosc dla każdej grupy wiekowe
                df['Tempo Stabilność'] = df['Tempo Stabilność'].fillna(
                    df.groupby('Kategoria wiekowa')['Tempo Stabilność'].transform('mean').round()
                )

                # Kolumna kategoria wiekowa. Konwersja/zamiana wartości na liczby całkowite

                list_category_ages= list(df['Kategoria wiekowa'].unique())
                for x in list_category_ages:
                    df.loc[df['Kategoria wiekowa'] == x, 'Kategoria wiekowa'] = int(x[1:])

                df['Kategoria wiekowa'] = df['Kategoria wiekowa'].astype(int)
              
                # Usuwam te rekordy co mają DNS lub DNF w kolumnie czas
                index_ = list(df[df['Czas']=='DNS'].index)
                df = df.drop(index_, axis=0)
                index_ = list(df[df['Czas']=='DNF'].index)
                df = df.drop(index_, axis=0)
            
                df_train = df.loc[:,['Płeć','5 km Tempo', '10 km Tempo','15 km Tempo','Rocznik','Czas']]
                df_train['Czas'] = df_train['Czas'].apply(convert_time_to_seconds)
               
                st.success(f"✅ Wczytano plik , {len(df_train)} rekordów po oczyszczeniu danych ")
               

                exp = setup(data=df_train, target='Czas', session_id=123, normalize=True)
                best = compare_models()
                create_model_ = create_model('gbr')
                tuned_lr = tune_model(create_model_)

                # Zapis modelu
                save_model(tuned_lr, 'best_gbr_model')
                file_name = "best_gbr_model.pkl"
                object_name = "models/best_gbr_model.pkl"
                client_do.upload_file(file_name, DO_SPACE_NAME, object_name)
                print("✅ Model zapisany w DigitalOcean Spaces!")


                # df_train = pd.read_csv(uploaded_file)
                

                # if validate_csv(df_train):
                #     if st.button("⚡ Wytrenuj model na połączonych plikach"):
                #         train_model(df_train)

        st.subheader("🗂️ Pliki w DigitalOcean (prefix: models/)")
        files = list_files(prefix="models/")
        if files:
            file_to_delete = st.selectbox("📂 Wybierz plik do usunięcia:", files)
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🗑️ Usuń zaznaczony plik"):
                    delete_file_from_do(file_to_delete)
                    st.experimental_rerun()
            with col2:
                if st.button("🔄 Odśwież listę"):
                    st.experimental_rerun()
        else:
            st.info("Brak plików do wyświetlenia.")

if __name__ == "__main__":
    main()

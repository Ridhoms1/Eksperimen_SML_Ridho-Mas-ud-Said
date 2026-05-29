import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

def preprocess_data():
    raw_dir = 'namadataset_raw'
    processed_dir = 'namadataset_preprocessing'
    
    # Buat direktori
    os.makedirs(processed_dir, exist_ok=True)
    
    # Cari file CSV di folder raw
    raw_files = [f for f in os.listdir(raw_dir) if f.endswith('.csv')]
    if not raw_files:
        print("Tidak ada file CSV ditemukan di folder namadataset_raw.")
        return
    
    raw_file_path = os.path.join(raw_dir, raw_files[0])
    print(f"Membaca data raw dari: {raw_file_path}")
    df = pd.read_csv(raw_file_path)
    
    # 1. Encoding Fitur Kategorikal (One-Hot Encoding)
    print("Menerapkan One-Hot Encoding...")
    categorical_features = ['Cloud Cover', 'Season', 'Location']
    # Memastikan kolom ada untuk menghindari error
    cat_cols_present = [col for col in categorical_features if col in df.columns]
    df_processed = pd.get_dummies(df, columns=cat_cols_present, drop_first=True)
    
    # 2. Encoding Target (Label Encoding)
    print("Menerapkan Label Encoding pada target...")
    if 'Weather Type' in df_processed.columns:
        label_encoder = LabelEncoder()
        df_processed['Weather Type'] = label_encoder.fit_transform(df_processed['Weather Type'])
        mapping_label = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
        print(f"Mapping Label Cuaca: {mapping_label}")
    
    # 3. Scaling (Standarisasi Fitur Numerik)
    print("Menerapkan StandardScaler pada fitur numerik...")
    numeric_features = ['Temperature', 'Humidity', 'Wind Speed', 'Precipitation (%)',
                        'Atmospheric Pressure', 'UV Index', 'Visibility (km)']
    num_cols_present = [col for col in numeric_features if col in df_processed.columns]
    
    if num_cols_present:
        scaler = StandardScaler()
        df_processed[num_cols_present] = scaler.fit_transform(df_processed[num_cols_present])
    
    # Simpan hasil akhir
    processed_file_path = os.path.join(processed_dir, 'weather_processed.csv')
    df_processed.to_csv(processed_file_path, index=False)
    print(f"Bagus! Dataset berhasil diproses dan disimpan di: {processed_file_path}")

if __name__ == "__main__":
    preprocess_data()
import pandas as pd
import numpy as np

# Carga el archivo CSV en un DataFrame de pandas.
# Asegúrate de que el nombre del archivo sea exactamente el mismo.
try:
    df_ndvi = pd.read_csv('NDVI_data_for_parcels.csv')
    print("DataFrame cargado exitosamente.")
    print("Primeras 5 filas del DataFrame:")
    print(df_ndvi.head())
except FileNotFoundError:
    print("Error: El archivo 'NDVI_data_for_parcels.csv' no se encontró.")
    print("Asegúrate de haber subido el archivo a tu entorno de Colab o que el nombre sea correcto.")
    
# --- Limpieza de datos ---
# Elimina la columna '.geo' que no es necesaria para el análisis.
if '.geo' in df_ndvi.columns:
    df_ndvi = df_ndvi.drop(columns=['.geo'])
    print("\nColumna '.geo' eliminada.")
    
# Verificamos si hay valores nulos en el DataFrame.
print("\nConteo de valores nulos por columna:")
print(df_ndvi.isnull().sum())

print("\nDataFrame final después de la limpieza:")
print(df_ndvi.head())

import pandas as pd
import numpy as np

df = pd.read_csv("heart.csv")

print("==SITUAZIONE INIZIALE==")
print(f"Pazienti con Colesterolo = 0: {(df['Cholesterol'] == 0).sum()}")

#Sostituzione di 0 con NaN (Not a Number)
df['Cholesterol'] = df['Cholesterol'].replace(0, np.nan)

#Calcoliamo la mediana del colesterolo (ignorando i NaN)
mediana_colesterolo = df['Cholesterol'].median()
print(f"Mediana del colesterolo (sui dati validi): {mediana_colesterolo:.1f} mg/dl")

#Sostitutuzione dei NaN con la mediana
df['Cholesterol'] = df['Cholesterol'].fillna(mediana_colesterolo)

#Piccolo check per stare sicuri
print("\n=== DOPO LA PULIZIA ===")
print(f"Pazienti con Colesterolo = 0: {(df['Cholesterol'] == 0).sum()}")

#Tecnica dell'One Hot Encoding
colonne_testo=["Sex", "ChestPainType", "RestingECG", "ExerciseAngina", "ST_Slope"]
#traduzione di testo in numeri
df_pronto = pd.get_dummies(df, columns=colonne_testo, drop_first=True, dtype=int)

df_pronto.to_csv("heart_pulito.csv", index=False)

print(f"Dimensioni dataset originale: {df.shape}")
print(f"Dimensioni nuovo dataset pulito: {df_pronto.shape}")
print("\nIl file 'heart_pulito.csv' è stato salvato con successo!")
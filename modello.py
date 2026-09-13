import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib

df = pd.read_csv('heart_pulito.csv')

#isolo input e output
#Tutte le caratteristiche dei pazienti
X = df.drop(columns=['HeartDisease'])
#Output che il cervello deve imparare e predire
y = df['HeartDisease']

print(f"Numero totale di caratteristiche cliniche: {X.shape[1]}")

#divido i dati per training e testing
#sintassi estremamente elegante ed efficace
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
#strytify mantiene lo stesso rapporto di malati e sani sia nel gruppo di Test che di Training
print(f"Pazienti usati per il TRAINING: {X_train.shape[0]}")
print(f"Pazienti usati per il TESTING: {X_test.shape[0]}")

#Inizializziamo e addestriamo l'algoritmo
#Creo la foresta di alberi decisionali (n_estimators = 100 )
modello = RandomForestClassifier(random_state=42)
#FASE DI APPRENDIMENTO
modello.fit(X_train, y_train)

print("\n==Addestramento completato con successo==")

#PREDIZIONE SUI PAZIENTI DEL TEST SET
y_pred = modello.predict(X_test)

# CALCOLO DELLE METRICHE
accuratezza = accuracy_score(y_test, y_pred)
matrice_confusione = confusion_matrix(y_test, y_pred)

print("\n==========================================")
print(f" ACCURATEZZA GLOBALE: {accuratezza * 100:.2f}%")
print("==========================================")

print("\nMATRICE DI CONFUSIONE:")
print(f" [Sani presi per Sani   (Veri Negativi)]:  {matrice_confusione[0][0]}")
print(f" [Sani presi per Malati (Falsi Positivi)]: {matrice_confusione[0][1]}")
print(f" [Malati presi per Sani (Falsi Negativi)]: {matrice_confusione[1][0]}  <-- CRITICO IN MEDICINA")
print(f" [Malati presi per Malati(Veri Positivi)]: {matrice_confusione[1][1]}")

# 6. SALVATAGGIO DEL MODELLO PER LA WEB APP (STEP 5)
joblib.dump(modello, "modello_cardio.pkl")
print("\n[OK] Il modello addestrato è stato salvato nel file 'modello_cardio.pkl'")




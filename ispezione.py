#ii file in questione serve per avere un'idea generale dei dati con cui si lavora
import pandas as pd

try:
    df = pd.read_csv('heart.csv') #dataframe
    print("Dataset caricato correttamente\n")
except FileNotFoundError:
    print("Errore: Dataset non caricato correttamente\n")
    exit()

print("==DIMENSIONI DATASET==")
#df.shape restituisce una coppia di valori dove il primo intero indica il numero di righe, l'altro il numero di colonne
print(f"Numero pazienti: {df.shape[0]}")
print(f"Numero paramentri medici: {df.shape[1]}")


print("==ANTEPRIMA DATI==")
#il comando df.head serve per stampare un'anteprima dei primi 5 pazienti dello studio
print(df.head(10)) #metto come parametro 10 per vederne proprio 10

print("==INFORMAZIONI E DATI MANCANTI")
#questa sezione è fondamentale per gestire eventuali buchi nei dati ( creerebbero instabilità al modello )
print(df.info())

print("==DISTRIBUZIONE DELLA PATOLOGIA==")
conteggio_pazienti = df["HeartDisease"].value_counts() #conto i pazienti sani e quelli malati
conteggio_percentuale = df["HeartDisease"].value_counts(normalize=True) * 100 #restistuisce la percentuale [con normalize = True ci restituisce la proporzione del conteggio compresa tra 0.0 e 1.0]

print(f"Pazienti Sani (0): {conteggio_pazienti[0]} ({conteggio_percentuale[0]:.1f}%)")
print(f"Pazienti a Rischio (1): {conteggio_pazienti[1]} ({conteggio_percentuale[1]:.1f}%)")


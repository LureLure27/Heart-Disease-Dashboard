#il seguente file ci permette di avere una visione concreta dei dati studiati
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#imposto tema per i grafici con seaborn
sns.set_theme(style="whitegrid")

#importo i dati grazie a pandas
df = pd.read_csv('heart.csv')

# ==========================================
# GRAFICO 1: ETÀ (Temporaneamente commentato)
# ==========================================
#Creo il primo grafico che ci permette di capire se l'età influsice nel rischio di malattie cardiache [in questo dataset]
plt.figure(figsize=(10,8))
#ora che abbiamo fatto la tela con plt facciamo un istrogramma con seaborn (uso histplot per intervalli continui)
sns.histplot(
    data = df,
    x = "Age",
    hue = "HeartDisease", #valore discriminante ( cosi da poter paragonare i risultati )
    kde = True,
    fill = True,
    element = "step"
)

plt.title("Distribuzione dell'Età: Sani vs A Rischio")
plt.xlabel("Età")
plt.ylabel("Numero Pazienti")

plt.savefig("grafico_eta.png")
plt.show()

# ==========================================
# GRAFICO 2: SESSO (In fase di analisi)
# ==========================================
#Ora andiamo a osservare come contribuisce il fattore Sesso verso la predisposizione di malattie cardiovascolari
plt.figure(figsize=(10,8))
#ora creo un countplot [ uso countplot per classificare dati discreti ]
sns.countplot(
    data = df,
    x = "Sex",
    hue = "HeartDisease"
)
plt.title("Distribuzione del Sesso: Sani vs A Rischio")
plt.xlabel("Sesso")
plt.ylabel("Numero Pazienti")
plt.savefig("grafico_sesso.png")
plt.show()


# ==========================================
# GRAFICO 3: BOXPLOT COLESTEROLO & PRESSIONE
# ==========================================
plt.figure(figsize=(10, 8))

# Creiamo due sotto-grafici affiancati (1 riga, 2 colonne)
plt.subplot(1, 2, 1)
sns.boxplot(data=df, y="Cholesterol", color="skyblue")
plt.title("Analisi Outlier: Colesterolo")
plt.subplot(1, 2, 2)
sns.boxplot(data=df, y="RestingBP", color="salmon")
plt.title("Analisi Outlier: Pressione a Riposo")

plt.tight_layout()
plt.savefig("grafico_outlier.png")
plt.show()

# ==========================================
# GRAFICO 4: HEATMAP DELLE CORRELAZIONI
# ==========================================
plt.figure(figsize=(10, 8))

# 1. Selezioniamo solo le colonne numeriche
df_numerico = df.select_dtypes(include=['number'])

# 2. Calcoliamo la matrice di correlazione
correlazione = df_numerico.corr()

# 3. Disegnamo la mappa di calore
sns.heatmap(
    correlazione,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    linewidths=0.5
)

plt.title("Matrice di Correlazione delle Variabili Numeriche")
plt.tight_layout()
plt.savefig("grafico_correlazione.png")
plt.show()

# GRAFICO EXTRA EDA: TIPO DI DOLORE TORACICO VS RISCHIO
plt.figure(figsize=(10, 8))

sns.countplot(
    data=df,
    x="ChestPainType",
    hue="HeartDisease"
)

plt.title("Rischio Cardiaco per Tipo di Dolore Toracico")
plt.xlabel("Tipo di Dolore (ASY, NAP, ATA, TA)")
plt.ylabel("Numero di Pazienti")

plt.savefig("grafico_dolore.png")
plt.show()
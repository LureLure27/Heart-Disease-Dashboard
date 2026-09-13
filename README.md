# 🫀 Sistema Intelligente di Supporto Decisionale per il Rischio Cardiaco

> **Nota Clinico-Legale:** Questo sistema è un prototipo sviluppato a scopo di ricerca, studio e implementazione di portfolii di Data Science. Non costituisce in alcun modo un dispositivo medico certificato e non sostituisce in alcun modo il giudizio, la diagnosi o il parere di un medico specialista.

Una web app interattiva e un modello di Machine Learning avanzato per la stima precoce delle patologie cardiache, progettato con un'architettura **Random Forest** e un'interfaccia utente basata su **Streamlit** e **Plotly**.

---

## 📋 Panoramica del Progetto
Le malattie cardiovascolari rappresentano una delle principali cause di mortalità globale. Questo progetto sviluppa un sistema predittivo capace di analizzare un profilo clinico multidimensionale per stimare la probabilità di patologia cardiaca (`HeartDisease`). L'obiettivo primario è fornire uno strumento di supporto decisionale rapido e trasparente, ottimizzato per minimizzare i **Falsi Negativi** e garantire la massima sicurezza clinica.

---

## 🧠 Tecniche di Intelligenza Artificiale e Machine Learning
Il nucleo predittivo del sistema non si basa su una singola regola rigida, ma su un approccio d'insieme (**Ensemble Learning**):

* **Random Forest Classifier:** Il modello costruisce in parallelo **100 alberi decisionali** indipendenti durante la fase di addestramento. Ciascun albero formula un proprio "voto" (Sano / A Rischio) e il verdetto finale è determinato dalla media delle probabilità espresse dall'intera foresta.
* **Criterio di Scelta delle Domande (Indice di Gini):** Ogni albero sceglie le domande da porre ai dati calcolando matematicamente la **purità** dei gruppi risultanti, riducendo al minimo il caos (impurità di Gini) e massimizzando il guadagno di informazione.
* **Casualità Controllata (Bootstrap e Feature Randomness):** Per evitare che tutti gli alberi siano identici, ciascuno di essi viene addestrato su un campione casuale estratto con reinserimento dei dati (Bootstrap) e ha a disposizione solo un sottoinsieme casuale di variabili cliniche a ogni nodo. Questo garantisce robustezza e previene l'overfitting.
* **Pre-elaborazione e One-Hot Encoding:** Le variabili categoriche (come il tipo di dolore o l'inclinazione del tratto ST) sono state convertite in formato binario tramite One-Hot Encoding, mentre i valori anomali riscontrati (es. valori di colesterolo pari a zero) sono stati trattati con tecniche di imputazione basate sulla mediana per preservare l'integrità statistica.

---

## 🧬 Dizionario dei Parametri Clinici Raccolti
Il modello valuta i seguenti 11 parametri clinici fondamentali del paziente:

* **Age (Età):** Età anagrafica espressa in anni. Fattore di rischio primario correlato all'invecchiamento vascolare.
* **Sex (Sesso):** Sesso biologico del paziente (`Maschio` / `Femmina`), statisticamente correlato a differenti profili di rischio a parità di età.
* **ChestPainType (Tipo di Dolore al Petto):** La classificazione sintomatica del dolore toracico:
  * *TA (Typical Angina):* Angina tipica, forte correlazione cardiaca.
  * *ATA (Atypical Angina):* Sintomi di dolore meno classici.
  * *NAP (Non-Anginal Pain):* Dolore probabilmente non di origine cardiaca.
  * *ASY (Asymptomatic):* Paziente asintomatico, spesso indicatore silenzioso di rischio elevato.
* **RestingBP (Pressione Sanguigna a Riposo):** Valore della pressione sistolica a riposo (mm Hg). Pressioni elevate aumentano il post-carico ventricolare.
* **Cholesterol (Colesterolo Sierico):** Concentrazione ematica di colesterolo (mg/dl). Valori alterati favoriscono i processi di aterosclerosi coronarica.
* **FastingBS (Glicemia a Digiuno):** Indicatore binario dello zucchero nel sangue (`1` se > 120 mg/dl, `0` altrimenti). Il diabete è un co-fattore cardiovascolare critico.
* **RestingECG (Elettrocardiogramma a Riposo):** Esito del tracciato elettrico basale:
  * *Normal:* Tracciato regolare.
  * *ST:* Presenza di anomalie dell'onda ST-T (segno di sofferenza miocardica).
  * *LVH:* Ipertrofia ventricolare sinistra (ispessimento patologico delle pareti cardiache).
* **MaxHR (Frequenza Cardiaca Massima):** Il battito cardiaco massimo raggiunto dal paziente sotto sforzo o test (valore numerico).
* **ExerciseAngina (Angina da Esercizio):** Eventuale insorgenza di dolore anginoso indotto dallo sforzo fisico (`Sì` / `No`).
* **Oldpeak (Depressione ST):** Misura numerica della depressione del tratto elettrocardiografico ST indotta dall'esercizio fisico rispetto allo stato di riposo.
* **ST_Slope (Inclinazione del Segmento ST):** La pendenza del tratto ST al picco dell'esercizio:
  * *Up (Crescente):* Profilo generalmente favorevole.
  * *Flat (Piatto):* Spesso associato a ischemia miocardica severa.
  * *Down (Decrescente):* Indicativo di sofferenza cardiaca importante.

---

## 📊 Dashboard e Visualizzazioni Interattive (Plotly)
L'interfaccia utente include due visualizzazioni avanzate realizzate con **Plotly** per interpretare in tempo reale l'output del modello:

1. **Gauge Chart (Tachimetro del Rischio):** Un indicatore grafico a semicerchio che traduce la percentuale di confidenza complessiva dei 100 alberi in una scala di colori immediata:
   * *Verde (0% - 30%):* Basso rischio clinico.
   * *Giallo (30% - 60%):* Area di attenzione / Medio rischio.
   * *Rosso (60% - 100%):* Alto rischio di patologia cardiaca.

2. **Feature Importance (Bar Chart Orizzontale):** 
   * *Che cos'è:* Questo grafico mostra la **classifica dei fattori clinici che contano di più in assoluto per la Random Forest**. Misura quanta "riduzione di impurità di Gini" ha generato ciascuna variabile durante l'addestramento dell'intera foresta di alberi.
   * *Come va interpretato:* Non indica quanto è grave quel valore *per il singolo paziente*, ma mostra **quali leve cliniche il modello guarda con più attenzione** per prendere le sue decisioni (es. se l'inclinazione del tratto ST `ST_Slope_Flat` o la depressione `Oldpeak` sono in cima alla lista, significa che la loro presenza spinge drasticamente il modello a cambiare idea verso il verdetto di rischio).

---

## ⚙️ Guida all'Installazione e Avvio Locale

1. **Clona la repository:**
   ```bash
   git clone [https://github.com/LureLure27/Heart-Disease-Dashboard](https://github.com/Heart-Disease-Dashboard)
   cd Nome-Repository
2. **Crea ambiente virtuale:**
    ```bash
   python -m venv venv
3. **Attiva ambiente virtuale:**

   ***Windows (Terminale):***
    ```bash
   venv\Scripts\activate
   ```
   
    ***Mac/Linux:***
    ```bash
   source venv/bin/activate
   ```
   
4. **Installa dipendenze necessarie:**

    Con un unico comando da terminale **installiamo tutte le librerie usate** 
     ```bash
   pip install streamlit pandas scikit-learn joblib plotly
    ```
5. **Avvia Web App:**
    ```bash
    streamlit run app.py
    

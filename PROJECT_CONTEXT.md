# SmartScheduler - Piano di Sviluppo e Istruzioni per Claude Code

## 1. Panoramica e Obiettivo del Progetto
Sviluppare **SmartScheduler**, un sistema basato su agenti multi-fase per la schedulazione dei turni ospedalieri. Questo progetto unisce le capacità di ragionamento dei **Large Language Models (LLMs)** con le tecniche di **Programmazione a Vincoli (CP)** utilizzando Google OR-Tools (`ortools.sat.python.cp_model`).
Il sistema deve generare orari bilanciati che rispettino rigorosamente i requisiti istituzionali (Vincoli Hard) tenendo in considerazione le preferenze e il benessere dei lavoratori (Vincoli Soft/Equità).

---

## Fase 0: Setup dell'Infrastruttura (Esecuzione Sicura e Orchestrazione)
* **Obiettivo:** Costruire una base Python robusta per l'interazione con l'LLM e l'esecuzione del codice.
* **Backend LLM:** Inferenza remota tramite **Google Gemini 2.5 Flash** (`google-genai`, `google.genai.Client`). Prerequisito: impostare la variabile d'ambiente `GEMINI_API_KEY` ottenibile da [Google AI Studio](https://aistudio.google.com/app/apikey).
* **Attività:**
  - Creare la classe `AgentExecutor` per gestire l'interazione con Google Gemini 2.5 Flash (usando `google-genai`).
  - Implementare una funzione `safe_execute(code_str)`:
    - Utilizzare `exec()` all'interno di un robusto blocco `try...except` per eseguire il codice OR-Tools generato.
    - Se OR-Tools fallisce (errore di sintassi o di logica), catturare lo stack trace dell'errore.
    - Inviare l'errore all'LLM per l'auto-correzione, impostando un limite massimo di tentativi (es. 3-5) per evitare loop infiniti.

---

## Fase 1: Definizione delle Preferenze (Workers Agent)
* **Obiettivo:** Raccogliere e formalizzare le preferenze espresse in linguaggio naturale convertendole in dati strutturati.
* **Meccanismo:**
  - **Input:** Un file di testo che dettaglia turni, lavoratori (13 in totale) e regole. I lavoratori forniscono input in linguaggio naturale (es. turni preferiti, disponibilità, tolleranza per i turni di notte/festivi).
  - **Prompt LLM:** Istruire il **Workers Agent** per analizzare questo testo.
  - **Output:** Estrarre i dati in modelli JSON/Pydantic, distinguendo chiaramente tra Vincoli Hard e Vincoli Soft. Assegnare un "modello di soddisfazione" per quantificare le preferenze di ciascun lavoratore.

---

## Fase 2: Bozza della Schedulazione (Drafting Agent)
* **Obiettivo:** Generare un'impostazione iniziale della schedulazione che soddisfi tutti i Vincoli Hard e massimizzi la soddisfazione generale.
* **Meccanismo:** Il **Drafting Agent** scrive lo script Python utilizzando `ortools.sat.python.cp_model`.
* **Configurazione:**
  - **Orizzonte temporale:** 1 mese (dal 7 Dicembre 2026 al 6 Gennaio 2027).
  - **Turni:** Mattina (8-14), Pomeriggio (14-20), Notte (20-8).
* **Vincoli Hard da codificare:**
  1. Massimo 36 ore settimanali per dipendente.
  2. Esattamente 25 turni al mese per ogni lavoratore.
  3. Massimo 1 turno al giorno; divieto ASSOLUTO di turni consecutivi a cavallo di due giorni.
  4. Il turno di notte vale come *turno doppio* (carico di lavoro = 2).
  5. Obbligo inderogabile di 2 giorni liberi dopo ogni turno di notte.
  6. Garantire 1 giorno di riposo a settimana (valutando le preferenze espresse).

* **Implementazione degli Use Case (Totale 13 Lavoratori):**
  - **Use Case A (Lavoratori Omogenei):** 13 lavoratori totali (tutti con lo stesso ruolo). Almeno 2 lavoratori assegnati a ogni turno.
  - **Use Case B (Lavoratori Specializzati):** 20 lavoratori totali (es. 13 standard, 7 specializzati). Minimo 2 standard + 1 specializzato per turno. Gli specializzati possono coprire i ruoli standard se necessario.

---

## Fase 3: Verifica della Schedulazione (Verification Agents)
* **Obiettivo:** Valutare la correttezza matematica e l'equità (fairness) della bozza generata.
* **Meccanismo:**
  - **Controllo Vincoli Hard:** Il **Verification Agent** analizza l'output del solver (OR-Tools). Se ci sono violazioni, rifiuta il piano e rimanda la traccia dell'errore indietro al Drafting Agent.
  - **Valutazione Equità:** Se il piano è matematicamente valido, il **Fairness Verification Agent** calcola le metriche di equità sull'intera forza lavoro. Deve identificare esplicitamente il *lavoratore più svantaggiato/meno soddisfatto*.

---

## Fase 4: Raffinamento della Schedulazione (Ciclo Iterativo)
* **Obiettivo:** Migliorare iterativamente l'equità senza violare i Vincoli Hard.
* **Meccanismo:**
  - **Prompt di Feedback:** Il Drafting Agent viene istruito a raffinare la schedulazione con l'obiettivo specifico di migliorare il punteggio di soddisfazione del *lavoratore meno soddisfatto* identificato nella Fase 3.
  - **Vincolo di Ottimizzazione:** Il raffinamento NON DEVE peggiorare il livello minimo di soddisfazione degli altri colleghi.
  - **Terminazione del Ciclo:** Il ciclo continua finché non è possibile ottenere alcun ulteriore miglioramento (es. OR-Tools restituisce lo status INFEASIBLE quando si cerca di forzare un livello di equità superiore) o finché non viene raggiunto un limite massimo di iterazioni.

---

## 5. Checklist di Consegna (Deliverables)
- [ ] **Codice Sorgente:** Un file `.zip` contenente il framework Python completamente implementato.
- [ ] **Output di Esempio:** Il modello `cp_model` parziale generato dall'LLM e la schedulazione finale risultante (es. in formato CSV o in visualizzazione Pandas DataFrame).
- [ ] **Breve Relazione (Report):** Documento che descrive l'approccio utilizzato, le scelte di design (come l'architettura Multi-Agente basata su Google Gemini 2.5 Flash e google-genai) e una discussione sulla qualità e sull'equità dei risultati ottenuti per gli Use Case A e B.
# SmartScheduler

**Un sistema multi-agente neuro-simbolico per la pianificazione equa dei turni ospedalieri.**

Progetto realizzato per il corso di **Intelligenza Artificiale**
*Corso di Laurea Magistrale in AI & ML, Università della Calabria (A.A. 2025/2026)*

**Autori:**
- Felice Dardis
- Sebastiano D'Urso
- Denis Mungo

---

## Descrizione del Progetto

**SmartScheduler** affronta la sfida della pianificazione dei turni in ambito ospedaliero (un classico problema *Nurse Rostering* NP-hard) garantendo contemporaneamente il rigoroso rispetto dei vincoli istituzionali (hard constraints) e la massimizzazione equa delle preferenze dei lavoratori (soft constraints espresse in linguaggio naturale).

L'approccio adottato è di natura **neuro-simbolica**:
- **La componente neurale (LLM):** Utilizza Google Gemini 2.5 Flash per interpretare le preferenze del personale espresse in testo libero (ferie desiderate, preferenze tra turni di mattina, pomeriggio e notte) e per generare in modo dinamico il codice per un solutore Constraint Programming.
- **La componente simbolica (CP-SAT & Verificatore):** Utilizza il motore Google OR-Tools per trovare soluzioni matematicamente valide, affiancato da un modulo di verifica deterministico completamente indipendente dall'LLM che certifica l'assenza di violazioni.

## Architettura e Fasi

Il sistema è orchestrato da una pipeline strutturata in 4 agenti principali:

1. **Fase 1 — Workers Agent** (`Fase1_workers_agent.py`):
   Analizza le preferenze in linguaggio naturale (`worker_preferences.txt`) e le formalizza in strutture dati rigorose e validate (tramite Pydantic).
2. **Fase 2 — Drafting Agent** (`Fase2_drafting_agent.py`):
   L'LLM genera la prima bozza del modello Constraint Programming (`cp_model`) definendo le variabili, i vincoli logici e la funzione obiettivo iniziale.
3. **Fase 3 — Verification Agent** (`Fase3_verification_agent.py`):
   Un verificatore puramente deterministico ri-controlla *da zero* le schedulazioni prodotte per certificare l'aderenza a tutti i vincoli (es. max 36 ore settimanali, 25 turni pesati al mese, coperture minime, riposi obbligatori). Calcola inoltre le metriche di *fairness*.
4. **Fase 4 — Refinement Agent** (`Fase4_refinement_agent.py`):
   Esegue un ciclo di ottimizzazione basato sul principio del **Max-Min Fairness (Leximin)**, migliorando iterativamente e in maniera bilanciata la soddisfazione (normalizzata) dei lavoratori più svantaggiati, senza violare i vincoli hard.

## Requisiti e Installazione

Il progetto richiede **Python 3.10+**.

Le dipendenze principali includono:
- `google-genai` (per l'interazione con l'API di Gemini)
- `ortools` (Google Optimization Tools, per il solver CP-SAT)
- `pydantic` (per la validazione dei dati)
- `customtkinter` (per l'interfaccia grafica opzionale)
- `python-dotenv` (per il caricamento delle variabili d'ambiente)

È necessario configurare una chiave API per l'LLM:
Creare un file `.env` nella cartella principale e inserire la chiave:
```env
GEMINI_API_KEY=la_tua_api_key_qui
```

## Come Eseguire il Progetto

Il sistema offre due modalità di esecuzione:

### 1. Interfaccia Grafica (GUI)
Avvia una dashboard interattiva basata su CustomTkinter, dalla quale è possibile lanciare la pipeline e visualizzarne i risultati.
```bash
python gui.py
```

### 2. Esecuzione da Terminale (CLI)
Avvia l'orchestratore direttamente da riga di comando per processare i due casi di studio inclusi nel progetto (Caso A: lavoratori omogenei; Caso B: lavoratori standard + specializzati).
```bash
python run_pipeline.py
```

## Casi di Studio Inclusi

- **Caso A**: 13 lavoratori omogenei.
- **Caso B**: 13 lavoratori standard e 7 lavoratori specializzati, con vincoli di copertura differenziata (è richiesto almeno uno specializzato per turno, che può fungere anche da standard).

Al termine dell'esecuzione, i file generati (codici generati dall'LLM, preferenze formalizzate e le schedulazioni in formato `.csv`) vengono salvati nella directory principale del progetto.

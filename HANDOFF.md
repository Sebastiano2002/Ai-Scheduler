# HANDOFF — SmartScheduler (continuazione in altra sessione)

> Documento di passaggio di consegne. Aggiornato al 2026-06-21.
> NB: gli ALTRI file `.md` del progetto (README.md, PROJECT_CONTEXT.md) sono
> VECCHI e da ignorare. La fonte di verità dei requisiti è **`Progetto IA.pdf`**.

---

## 1. Cos'è il progetto

**SmartScheduler**: sistema multi-agente che genera turni ospedalieri equi
combinando LLM (Google Gemini 2.5 Flash) + verifica simbolica + CP (OR-Tools
CP-SAT). Pipeline in 4 fasi:

1. **Fase 1 — Workers Agent** (`Fase1_workers_agent.py` + `models.py`): preferenze
   in linguaggio naturale → `formalized_preferences_case_X.py` (HARD/SOFT + pesi
   di soddisfazione). Validazione Pydantic.
2. **Fase 2 — Drafting Agent** (`Fase2_drafting_agent.py`): l'LLM SCRIVE il
   `cp_model` della bozza (two-phase warm-start). Output `draft_code_case_X.txt`
   + `schedule_case_X.csv`.
3. **Fase 3 — Verification** (`Fase3_verification_agent.py`): verifica hard
   INDIPENDENTE (ricodifica le leggi) + fairness metrics + lavoratore più
   svantaggiato.
4. **Fase 4 — Refinement** (`Fase4_refinement_agent.py`): ciclo leximin
   (Max-Min Fairness). L'LLM scrive UNA volta un template parametrico, il loop
   lo ri-esegue bloccando i pavimenti.

Infra: `llm_engine.py` (AgentExecutor, auto-correzione), `input_data.py` (dati
problema), `run_pipeline.py` (orchestratore), `gui.py` (GUI customtkinter).

**Use case**: A = 13 lavoratori omogenei (≥2/turno); B = 13 standard + 7
specializzati (≥1 spec + ≥3 totali per turno). Orizzonte 7/12/2026 → 6/1/2027 (31
gg). Vincoli hard: 25 turni pesati/mese (Notte=2), ≤36h/sett (finestra
scorrevole), 1 turno/giorno, divieto Notte→Mattina, 2 riposi dopo Notte, ≥1
riposo/sett, turni vietati per-lavoratore.

---

## 2. Stato attuale (cosa è già OK)

- **Entrambi i casi A e B** sono generati e **verificati hard-validi** (verifica
  indipendente eseguita: `[OK] Vincoli hard: TUTTI rispettati`).
- Deliverable presenti per A e B: `formalized_preferences_case_*.py`,
  `draft_code_case_*.txt`, `schedule_case_*.csv`, `final_code_case_*.txt`,
  `schedule_case_*_final.csv`.
- Robustezza pipeline: backoff esponenziale nei tentativi di bozza (run_pipeline)
  + meccanismo "Deep Dive" nella Fase 4 (su stallo raddoppia il tempo, poi forza
  il lock). NB: il Deep-Dive rende la terminazione EURISTICA, non una prova di
  ottimalità leximin → da dichiarare nella relazione.

---

## 3. LAVORO APPENA FATTO — Normalizzazione della soddisfazione (proportional fairness)

### Problema risolto
La soddisfazione ASSOLUTA non è confrontabile tra lavoratori: chi predilige la
Notte (peso 1) ha un massimo strutturale di ~8-12, chi predilige Mattina/Pom.
arriva a ~125. Il vecchio leximin assoluto inseguiva W12 (già al suo soffitto) e
muoveva il minimo solo da 7,0→8,0. **Soluzione**: normalizzare ogni soddisfazione
sul MASSIMO INDIVIDUALE → `sat_norm(w) = sat(w)/sat_max(w)` e fare leximin sui
valori normalizzati ("ognuno egualmente vicino al proprio ottimo").

### Modifiche implementate (NON ancora committate, da verificare/ri-eseguire)
- **`Fase2_drafting_agent.py`**:
  - `compute_sat_max(data)` → mini-modello CP per-lavoratore (vincoli hard
    individuali, NO staffing) che massimizza la sola soddisfazione. Cache
    `_SAT_MAX_CACHE` per caso.
  - `worker_satisfaction_pct(sat_abs, sat_max)` → % del proprio massimo.
  - `export_csv` → aggiunte colonne `sat_max` e `soddisfazione_pct`.
- **`Fase3_verification_agent.py`**:
  - `FairnessMetrics` esteso con campi normalizzati (`*_pct`, `sat_max_per_worker`,
    `satisfaction_pct_per_worker`).
  - `FairnessVerifier.evaluate` identifica il più svantaggiato sulla scala
    NORMALIZZATA; calcola entrambe le metriche.
  - `print_report` mostra ASSOLUTA vs NORMALIZZATA affiancate + classifica.
- **`Fase4_refinement_agent.py`**:
  - Costante `NORM_SCALE = 100` (risoluzione 1%).
  - `_sat_max_scaled(data)` + `_build_refinement_context` inietta `SAT_MAX_SCALED`
    e `NORM_SCALE` nel namespace.
  - **Scheletro del prompt leximin riscritto**: obiettivo = max-min NORMALIZZATO,
    linearizzato `z * SAT_MAX_SCALED[w] <= sat[w] * NORM_SCALE`; pavimenti in
    scala normalizzata `sat[w]*NORM_SCALE >= floor*SAT_MAX_SCALED[w]`; `BIG=1000000`.
  - Bookkeeping del loop in permille→%: `norm_scaled[w] = (sat_scaled[w]*NORM_SCALE)
    // sat_max_scaled[w]` (divisione FLOOR per garantire fattibilità dei pavimenti);
    `z_star` = minimo normalizzato; display in %.
  - `RefinementOutcome` esteso con vettori `*_pct`; `print_refinement_summary`
    mostra PRIMA→DOPO su ENTRAMBE le scale; `improved` valutato sul normalizzato.

### Risultati VALIDATI (senza LLM, su CSV esistenti)
- Fase 3 Caso A: il "più svantaggiato" passa da **W12=8.0 (assoluto)** a
  **W02=26,7% (normalizzato)** — W12 risulta al **100%** (è al suo soffitto, quindi
  trattata bene).
- Fase 3 Caso B: da W12/W16=8.0 a **W17=45,6%**.
- Test deterministico del nuovo leximin (vincoli hard Caso A + obiettivo
  normalizzato, eseguito a mano senza LLM): minimo normalizzato **26,7% → 52,0%**
  in un solo livello, tutti i vincoli hard rispettati. La formulazione è corretta.

---

## 4. DA FARE (prossimi passi)

1. **RI-ESEGUIRE LA PIPELINE con la `GEMINI_API_KEY`** (non impostata
   nell'ambiente di sviluppo): serve per rigenerare gli output finali col leximin
   normalizzato via LLM. Comandi:
   ```
   python run_pipeline.py --case all
   # oppure solo raffinamento da bozze esistenti:
   python Fase4_refinement_agent.py --case A --from-draft
   python Fase4_refinement_agent.py --case B --from-draft
   ```
   → Verificare che l'LLM generi correttamente il template normalizzato (la prima
   iterazione è una chiamata LLM; se sbaglia, l'auto-correzione interviene).
2. **DECISIONE APERTA**: allineare anche la **Fase 2** (bozza) al normalizzato?
   Oggi la Fase 2 usa soddisfazione assoluta + penalità di fairness; l'equità vera
   la fa la Fase 4. L'utente deve decidere se cambiarla.
3. **Commit** delle modifiche (working tree attualmente pulito ESCLUSE le mie
   modifiche non committate alle 3 fasi).
4. **Relazione**: integrare il testo già pronto sulla normalizzazione (vedi §6).

---

## 5. GOTCHAS / note tecniche

- **`cp_model.LinearExpr.sum(... for ...)` FALLISCE con i generatori** in questa
  versione di OR-tools (`TypeError: only accept linear expressions and constants`).
  Le modifiche utente "Patch sum" usano generatori nei prompt → funzionano solo
  grazie all'auto-correzione LLM. Nei MIEI file ho usato la forma a LISTA
  `LinearExpr.sum([...])` o `sum()` nativo. Valutare di uniformare i prompt.
- La chiave API si imposta via env `GEMINI_API_KEY` o file `.env` (caricato da
  `llm_engine.py` se python-dotenv è installato).
- Pipeline NON deterministica (LLM): run diversi danno orari diversi anche con
  `temperature=0.1`. Il verificatore deterministico garantisce la correttezza.
- `compute_sat_max` ignora lo staffing (è il punto ideale individuale): scelta di
  design corretta per la proportional fairness; cache-ata per non rallentare la
  Fase 3 chiamata in loop.

---

## 6. Testo pronto per la RELAZIONE (normalizzazione)

> Il *satisfaction model* assegna a ogni turno un peso (positivo per i graditi,
> negativo per gli indesiderati, magnitudo legata a `flexibility_score`). La
> soddisfazione assoluta NON è confrontabile tra lavoratori: l'intervallo
> raggiungibile dipende dai pesi individuali. Poiché il PDF chiede il più
> svantaggiato **rispetto agli altri**, adottiamo come metrica di equità la
> soddisfazione **normalizzata** `s̃(w) = s(w)/s_max(w) ∈ [0,1]`, dove `s_max(w)`
> è il massimo ottenibile da w rispettando i vincoli hard (risolto una tantum). Il
> Fairness Verifier (Fase 3) e il leximin (Fase 4) operano su `s̃`; la
> soddisfazione assoluta resta riportata come modello descrittivo (Stage 1).
> È una nozione di *proportional fairness*: equità = ognuno egualmente vicino al
> massimo che le sue preferenze consentono. (Da dichiarare anche: interpretazione
> "25 turni pesati / 36h finestra scorrevole"; Deep-Dive = terminazione euristica;
> non-determinismo LLM.)

---

## 7. Criticità ancora APERTE (analisi)

- 🟠 **Leximin non "puro"**: l'obiettivo `z*BIG + Σsat` ha una coda utilitaristica
  (massimizza la somma a parità di minimo). Ai livelli successivi non è leximin
  esatto. Migliorabile con max-min ricorsivo sui liberi.
- 🟡 Ridondanze: vincolo Notte→Mattina sussunto dai 2 riposi; penalità di fairness
  Fase 2 scavalcate dal leximin Fase 4.
- 🟡 Niente test automatici; fairness metrics non misurano la distribuzione di
  notti/festivi (solo soddisfazione).

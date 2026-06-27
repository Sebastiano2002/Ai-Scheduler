# HANDOFF — SmartScheduler (continuazione in altra sessione)

> Documento di passaggio di consegne. Aggiornato al 2026-06-23.
> NB: gli ALTRI file `.md` del progetto (README.md, PROJECT_CONTEXT.md) sono
> VECCHI e da ignorare. La fonte di verità dei requisiti è **`Progetto IA.pdf`**.

---

## 1. Cos'è il progetto

**SmartScheduler**: sistema multi-agente che genera turni ospedalieri equi
combinando LLM (Google Gemini 2.5 Flash) + verifica simbolica + CP (OR-Tools
CP-SAT). Pipeline in 4 fasi:

1. **Fase 1 — Workers Agent** (`Fase1_workers_agent.py` + `models.py`): preferenze
   in linguaggio naturale (`worker_preferences.txt`) → `formalized_preferences_case_X.py`
   (HARD/SOFT + pesi di soddisfazione). Validazione Pydantic con validatori custom.
2. **Fase 2 — Drafting Agent** (`Fase2_drafting_agent.py`): l'LLM SCRIVE il
   `cp_model` della bozza (two-phase warm-start). Output `draft_code_case_X.txt`
   + `schedule_case_X.csv`.
3. **Fase 3 — Verification** (`Fase3_verification_agent.py`): verifica hard
   INDIPENDENTE (ricodifica le leggi, niente LLM, deterministica) + fairness
   metrics + identificazione del lavoratore più svantaggiato (su scala normalizzata).
4. **Fase 4 — Refinement** (`Fase4_refinement_agent.py`): ciclo leximin
   (Max-Min Fairness). L'LLM scrive UNA volta un template parametrico, il loop
   lo ri-esegue bloccando i pavimenti (`LOCKED_FLOORS`).

Infra: `llm_engine.py` (AgentExecutor, auto-correzione), `input_data.py` (dati
problema), `run_pipeline.py` (orchestratore), `gui.py` (GUI customtkinter).

**Use case**: A = 13 lavoratori omogenei (≥2/turno); B = 13 standard + 7
specializzati (≥2 standard + ≥1 spec per turno; lo spec può coprire lo standard).
Orizzonte 7/12/2026 → 6/1/2027 (31 gg). Vincoli hard: 25 turni pesati/mese
(Notte=2), ≤36h/sett (finestra scorrevole), 1 turno/giorno, divieto
Notte→Mattina, 2 riposi dopo Notte, ≥1 riposo/sett, turni vietati per-lavoratore.

---

## 2. Stato attuale (cosa è già OK)

- **Entrambi i casi A e B** generati e **verificati hard-validi**
  (`[OK] Vincoli hard: TUTTI rispettati`).
- Deliverable presenti per A e B: `formalized_preferences_case_*.py`,
  `draft_code_case_*.txt`, `schedule_case_*.csv`, `final_code_case_*.txt`,
  `schedule_case_*_final.csv`.
- Robustezza pipeline: backoff esponenziale (rate-limit 429) + meccanismo
  "Deep Dive" in Fase 4 (su stallo raddoppia il tempo a 120s, poi forza il lock).
  NB: il Deep-Dive rende la terminazione EURISTICA, non una prova di ottimalità
  leximin → già dichiarato nella relazione.
- **Equità su soddisfazione NORMALIZZATA** (proportional fairness):
  `norm(w)=sat(w)/sat_max(w)`. Il leximin lavora sul minimo normalizzato così da
  non inseguire chi ha pesi piccoli (es. W12, soffitto ~8).

---

## 3. LAVORO APPENA FATTO — Relazione tecnica finale (Short Relation)

- Creato **`Relazione_SmartScheduler.txt`**: relazione in **LaTeX**, in italiano,
  accademica, ~15 pagine. Contiene i 6 punti richiesti: introduzione, struttura/
  scelte di design (incl. motivazione cloud per mancanza hardware), analisi delle
  4 fasi vs traccia, risultati casi A/B in tabelle, scenari futuri, conclusioni.
- 5 placeholder figure (commenti `% [FIGURA n — DA INSERIRE]`) con descrizione di
  cosa mettere: schema architetturale, flusso auto-correzione, screenshot GUI,
  classifica fairness, grafico prima/dopo.
- Tratta gli spunti critici: sicurezza `exec()` (Docker/gVisor/WASM), prompt da
  "incollaggio" → funzioni helper di astrazione, e la riflessione "LLM come
  strumento che accelera, non sostituto del progettista".
- **Estensione `.txt` voluta** (richiesta utente) ma è LaTeX: per compilare
  rinominare in `.tex` o `pdflatex Relazione_SmartScheduler.txt`. Pacchetti:
  babel, booktabs, listings, hyperref, amsmath, graphicx, xcolor, float.

### Dati usati nella relazione (verificati da CSV/log)
| Metrica (norm.)        | Caso A          | Caso B           |
|------------------------|-----------------|------------------|
| Min normalizzato       | 44.0% → 52.0%   | 43.4% → 52.0%    |
| Media normalizzata     | 69.6% → 71.1%   | 71.1% → 73.6%    |
| Totale assoluto        | 856.6 → 870.0   | 1287.8 → 1321.0  |
| Scarto max-min (norm.) | —               | 56.6% → 48.0%    |
| Dev. std (norm.)       | —               | 18.5% → 16.1%    |
| Più svantaggiato       | W05 (52.0%)     | W14→W05 (52.0%)  |

> I valori aggregati del Caso A sono stati RICALCOLATI dai CSV
> (`schedule_case_A.csv` vs `_final.csv`), perché `Output terminale.txt`
> registrava solo il Caso B. I valori del Caso B vengono direttamente dai log
> di Fase 3/4.

---

## 4. Cose da NON rifare / trappole note

- Non trattare le ferie (`giorni_indesiderati`) come hard: sono soft ad alta
  penalità (`UNDESIRED_DAY_PENALTY=50`) per non rendere il problema INFEASIBLE.
- Il vincolo "25 turni" è PESATO (Notte=2), non un semplice conteggio.
- "Due turni consecutivi vietati" = solo la catena contigua Notte→Mattina; i
  giorni diurni adiacenti sono PERMESSI (altrimenti infeasible con 25/31).
- Il warm-start (`AddHint`) è indispensabile: senza, il vincolo `==25` può
  lasciare lo status su UNKNOWN entro il time limit.
- `safe_execute` usa un namespace UNICO (globals + context fusi) per evitare il
  NameError nelle comprehension annidate — non separarli di nuovo.

---

## 5. Possibili prossimi passi

- Generare le figure reali della relazione (5 placeholder).
- Compilare il LaTeX e rileggere l'impaginato (controllo ~15 pagine).
- Eventuale zip dei deliverable per la consegna (Project Files + esempio output +
  Short Relation).
- Eventuali estensioni: helper di astrazione nei prompt, sandbox per `exec()`,
  certificati di ottimalità leximin sul Caso B.

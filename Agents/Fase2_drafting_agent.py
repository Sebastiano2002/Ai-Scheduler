"""
Fase2_drafting_agent.py
=================
Fase 2 - Drafting Agent (Bozza della Schedulazione).

La bozza e' generata ESCLUSIVAMENTE tramite l'LLM

  il Drafting Agent costruisce un prompt dettagliato che istruisce l'LLM a
  SCRIVERE il codice `ortools.sat.python.cp_model`. Il codice generato viene
  eseguito in sicurezza tramite il motore della Fase 0
  (`AgentExecutor.run_with_retry`), con auto-correzione sugli errori.

Il percorso produce una struttura dati di output (`ScheduleResult`) per le
Fasi 3-4 (verifica + raffinamento). La validazione formale
dei vincoli hard NON e' compito di questa fase: spetta esclusivamente al
Verification Agent della Fase 3.
"""

import argparse
import csv
import datetime
import importlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from ortools.sat.python import cp_model

import input_data


# ===========================================================================
# 0. CARICAMENTO DEI DATI DEL PROBLEMA PER UNO USE CASE
# ===========================================================================
# Le preferenze sono i pesi di soddisfazione formalizzati dalla Fase 1
SATISFACTION_SCALE = 10  # i pesi hanno al massimo 1 cifra decimale -> *10.

# Penalita' (in punti di soddisfazione) applicata quando un lavoratore viene
# assegnato in un suo "giorno indesiderato" (richiesta di ferie). E' un vincolo
# SOFT ad alta penalita': il solver evita questi giorni con forza, ma in caso di
# emergenza di organico puo' comunque assegnarli invece di fallire (INFEASIBLE).
UNDESIRED_DAY_PENALTY = 50.0

# Penalita' di EQUITA'. La bozza non deve solo massimizzare la soddisfazione TOTALE, ma anche "distribuire i
# turni indesiderati in modo equo... evitando di penalizzare in modo
# sproporzionato singoli lavoratori". Modelliamo l'equita' tra lavoratori.
#   - turni di Notte (il turno piu' gravoso, doppio);
#   - VIOLAZIONI di ferie, cioe' turni che un lavoratore e' costretto a fare in
#     un SUO giorno indesiderato (vedi UNDESIRED_DAYS). 
# Da ricalibrare dopo un run reale: alzandoli si ottiene piu' equita' a scapito della
# soddisfazione totale, abbassandoli il contrario.
NIGHT_BALANCE_PENALTY = 2.0      
UNDESIRED_BALANCE_PENALTY = 2.0  


@dataclass
class ProblemData:
    """Dati completi del problema per uno specifico use case (A o B)."""

    case_label: str
    worker_ids: List[str]
    worker_names: Dict[str, str]
    standard_ids: List[str]
    specialized_ids: List[str]
    # Vincoli SOFT per lavoratore (turni_preferiti/indesiderati,
    # giorni_indesiderati, flexibility_score, satisfaction_weights) dalla Fase 1.
    preferences: Dict[str, dict]
    # Giorni indesiderati / richieste di ferie (vincolo SOFT ad alta penalita')
    undesired_days: Dict[str, set]
    # Turni assolutamente vietati (vincolo HARD) per lavoratore
    forbidden: Dict[str, set]
    staffing: dict


def load_problem_data(case_label: str) -> ProblemData:
    """
    Raccoglie in un'unica struttura tutto cio' che serve al modello CP per uno
    use case: anagrafica lavoratori, ruoli, preferenze formalizzate (Fase 1) e
    requisiti di staffing.
    """
    case_label = case_label.upper()
    if case_label not in input_data.USE_CASES:
        raise ValueError(f"Use case sconosciuto: {case_label!r} (ammessi: A, B).")

    use_case = input_data.USE_CASES[case_label]
    workers = use_case["workers"]

    # Preferenze formalizzate prodotte dalla Fase 1: due strutture parallele
    try:
        mod = importlib.import_module(f"formalized_preferences_case_{case_label}")
        soft_constraints = dict(mod.SOFT_CONSTRAINTS)
        hard_per_worker = dict(mod.HARD_CONSTRAINTS["per_worker"])
    except ModuleNotFoundError as exc:
        raise FileNotFoundError(
            f"Manca 'formalized_preferences_case_{case_label}.py'. "
            f"Esegui prima la Fase 1 (Fase1_workers_agent.py)."
        ) from exc
    except (AttributeError, KeyError) as exc:
        raise ValueError(
            f"'formalized_preferences_case_{case_label}.py' non espone le strutture "
            f"attese HARD_CONSTRAINTS/SOFT_CONSTRAINTS. Rigenera la Fase 1 "
            f"(Fase1_workers_agent.py)."
        ) from exc

    # Le preferenze (soft) servono per la funzione obiettivo e i
    # ragionamenti sulla soddisfazione delle fasi 2/3/4.
    preferences = soft_constraints

    worker_ids = [w["id"] for w in workers]
    worker_names = {w["id"]: w["nome"] for w in workers}
    standard_ids = [w["id"] for w in workers if w["ruolo"] == "standard"]
    specialized_ids = [w["id"] for w in workers if w["ruolo"] == "specializzato"]

    # Mappa data ISO -> indice giorno per tradurre le date in indici.
    iso_to_index = {d.isoformat(): i for i, d in enumerate(input_data.PLANNING_DATES)}
    # Giorni indesiderati (SOFT, da SOFT_CONSTRAINTS) e turni vietati (HARD).
    undesired_days = {}
    forbidden = {}
    for wid in worker_ids:
        giorni = soft_constraints.get(wid, {}).get("giorni_indesiderati", [])
        undesired_days[wid] = {
            iso_to_index[g] for g in giorni if g in iso_to_index
        }
        hc = hard_per_worker.get(wid, {})
        forbidden[wid] = {
            s for s in hc.get("turni_vietati", []) if s in input_data.SHIFT_CODES
        }

    return ProblemData(
        case_label=case_label,
        worker_ids=worker_ids,
        worker_names=worker_names,
        standard_ids=standard_ids,
        specialized_ids=specialized_ids,
        preferences=preferences,
        undesired_days=undesired_days,
        forbidden=forbidden,
        staffing=use_case["staffing"],
    )


# ===========================================================================
# 1. STRUTTURA DI OUTPUT DELLA BOZZA (percorso LLM)
# ===========================================================================
@dataclass
class ScheduleResult:
    """
    Risultato di una bozza di schedulazione.

    """

    case_label: str
    status_name: str
    feasible: bool
    schedule: Dict[str, Dict[int, Optional[str]]] = field(default_factory=dict)
    objective_value: Optional[float] = None
    satisfaction_per_worker: Dict[str, float] = field(default_factory=dict)
    source: str = "llm"
    # Sorgente cp_model effettivamente eseguito che ha prodotto questa bozza
    # E' l'input testuale della Fase 4.
    generated_code: Optional[str] = None


# ===========================================================================
# 2. PERCORSO LLM (l'LLM SCRIVE il cp_model)
# ===========================================================================
def build_drafting_prompt(data: ProblemData) -> str:
    """
    Costruisce il prompt che istruisce l'LLM a generare lo script
    `ortools.sat.python.cp_model` per la bozza di schedulazione.

    Il codice generato gira nel namespace di esecuzione (safe_execute)
    """
    shifts = input_data.SHIFTS
    descrizione_turni = "\n".join(
        f"  - '{c}': {shifts[c]['nome']} {shifts[c]['inizio']}-{shifts[c]['fine']}, "
        f"{shifts[c]['durata_ore']}h, peso={shifts[c]['peso_turni']}"
        f"{' (turno DOPPIO)' if shifts[c]['turno_doppio'] else ''}"
        for c in input_data.SHIFT_CODES
    )

    if data.case_label == "A":
        regola_staffing = (
            f"- Caso A: almeno {data.staffing['min_lavoratori_per_turno']} "
            f"lavoratori assegnati a OGNI turno di OGNI giorno."
        )
    else:
        min_std = data.staffing["min_standard_per_turno"]
        min_spec = data.staffing["min_specializzati_per_turno"]
        regola_staffing = (
            f"- Caso B: per OGNI turno di OGNI giorno almeno {min_spec} lavoratore/i "
            f"'specializzato' (SPECIALIZED_IDS) e almeno {min_std + min_spec} "
            f"lavoratori TOTALI (gli specializzati possono coprire anche i ruoli "
            f"standard). In formule, per ogni giorno d e turno s:\n"
            f"    cp_model.LinearExpr.sum([x[(w, d, s)] for w in SPECIALIZED_IDS]) >= {min_spec}\n"
            f"    cp_model.LinearExpr.sum([x[(w, d, s)] for w in WORKER_IDS]) >= {min_std + min_spec}"
        )

    prompt = f"""Sei il "Drafting Agent" di SmartScheduler, un sistema di schedulazione di
turni ospedalieri. Devi SCRIVERE codice Python che usa
`ortools.sat.python.cp_model` per generare una bozza di orario mensile per lo
USE CASE {data.case_label}.

### ORIZZONTE E TURNI
Orizzonte: {input_data.NUM_DAYS} giorni (indici 0..{input_data.NUM_DAYS - 1}),
dal {input_data.START_DATE.isoformat()} al {input_data.END_DATE.isoformat()}.
Turni (codici): {", ".join(input_data.SHIFT_CODES)}
{descrizione_turni}

### VARIABILI GIA' DISPONIBILI NEL NAMESPACE (NON ridefinirle)
- WORKER_IDS      : list[str] -> id dei lavoratori di questo use case
- NUM_DAYS        : int       -> numero di giorni dell'orizzonte
- SHIFT_CODES     : list[str] -> ['M', 'P', 'N']
- SHIFT_HOURS     : dict      -> ore per turno  {{'M':6,'P':6,'N':12}}
- SHIFT_WEIGHT    : dict      -> peso per turno {{'M':1,'P':1,'N':2}} (Notte doppia)
- PREFERENCES     : dict      -> {{wid: {{'satisfaction_weights': {{'M':..,'P':..,'N':..}}, ...}}}}
- UNDESIRED_DAYS  : dict      -> {{wid: set(indici_giorno) che il lavoratore preferirebbe NON lavorare (ferie)}}
- UNDESIRED_DAY_PENALTY : float -> penalita' di soddisfazione per ogni turno in un giorno indesiderato
- FORBIDDEN_SHIFTS: dict      -> {{wid: set(codici_turno) ASSOLUTAMENTE vietati al lavoratore}}
- HOLIDAY_DAYS    : set[int]  -> indici dei giorni festivi nell'orizzonte (informativo)
- NIGHT_BALANCE_PENALTY     : float -> peso di equita' sullo scarto (max-min) del numero di Notti
- UNDESIRED_BALANCE_PENALTY : float -> peso di equita' sullo scarto (max-min) delle violazioni di ferie
- STANDARD_IDS    : list[str] -> id dei lavoratori standard
- SPECIALIZED_IDS : list[str] -> id dei lavoratori specializzati (vuota nel Caso A)
- MAX_TIME        : float     -> tempo massimo di risoluzione in secondi
- cp_model        : modulo ortools.sat.python.cp_model gia' importato

### STRUTTURA OBBLIGATORIA: build_model() + DUE FASI DI RISOLUZIONE
FONDAMENTALE PER LA FATTIBILITA' (soprattutto CASO B con 20 lavoratori). Il
modello e' molto vincolato: "ESATTAMENTE 25 turni pesati" + "2 riposi dopo la
Notte" lasciano una regione fattibile piccolissima. Se risolvi in un colpo solo
con l'obiettivo di soddisfazione, il solver spesso NON trova alcuna soluzione
entro il tempo (status UNKNOWN), perche' quell'obiettivo NON e' allineato al
vincolo dei 25 turni. Devi quindi risolvere in DUE FASI con warm-start (hint),
incapsulando i vincoli in una funzione `build_model()` riutilizzabile:
  - `build_model()`: crea il modello, le variabili x[(w,d,s)] e AGGIUNGE TUTTI i
    vincoli hard qui sotto (NESSUN obiettivo); ritorna (model, x).
  - FASE A (warm-start): obiettivo = SOMMA PESATA dei turni (allineato al vincolo
    dei 25) -> trova in fretta un assetto FATTIBILE.
  - FASE B (reale): ricostruisci il modello, passa la soluzione della FASE A come
    HINT (model.AddHint) e usa l'obiettivo reale (soddisfazione + ferie + equita').
    L'hint garantisce SEMPRE almeno una soluzione FEASIBLE (mai UNKNOWN).
I vincoli seguenti vanno messi DENTRO build_model(). Lo scheletro completo e'
nella sezione finale "### SCHELETRO DI CODICE OBBLIGATORIO".

### VINCOLI HARD DA CODIFICARE (sono LEGGI inviolabili)
1. Max 36 ore settimanali per lavoratore (per ogni finestra di 7 giorni consecutivi).
2. ESATTAMENTE 25 turni/mese per lavoratore, conteggiati col peso (la Notte vale 2):
   sum(SHIFT_WEIGHT[s] * x[w,d,s]) == 25.
3. Max 1 turno al giorno. Divieto ASSOLUTO solo della catena Notte(d)->Mattina(d+1).
   (NON significa "vietato lavorare in due giorni consecutivi": i turni di giorno
   consecutivi sono permessi.)
4. La Notte e' un turno doppio: gia' riflesso in SHIFT_WEIGHT e SHIFT_HOURS.
5. Dopo OGNI turno di Notte: 2 giorni di riposo TOTALE (nessun turno in d+1 e d+2).
6. Almeno 1 giorno di riposo a settimana (per ogni finestra di 7 giorni: <=6 lavorati).
- TURNI VIETATI: nessun turno di tipo s per il lavoratore w se s in FORBIDDEN_SHIFTS[w]
  (divieto assoluto, es. per motivi di salute). Pattern:
  ```python
  for w in WORKER_IDS:
      for s in FORBIDDEN_SHIFTS[w]:
          for d in range(NUM_DAYS):
              model.Add(x[(w, d, s)] == 0)
  ```
{regola_staffing}

### IMPLEMENTAZIONE CORRETTA DEI VINCOLI - PATTERN OBBLIGATORI

**Performance Tip:** Usa SEMPRE `cp_model.LinearExpr.sum(...)` invece della funzione nativa Python `sum(...)` per sommare le variabili booleane del modello, in quanto è molto più efficiente computazionalmente.

**Vincolo 5 (riposi dopo Notte) - usa ESATTAMENTE questo pattern:**
```python
for w in WORKER_IDS:
    for d in range(NUM_DAYS):
        for k in range(1, 3):          # k=1 e k=2 (i due giorni di riposo)
            if d + k < NUM_DAYS:       # INDISPENSABILE: verifica che il giorno esista
                for s in SHIFT_CODES:
                    model.Add(x[(w, d, 'N')] + x[(w, d + k, s)] <= 1)
```
   TRAPPOLA COMUNE #1: dimenticare il controllo `if d + k < NUM_DAYS`.
   Senza questo check, l'ultimo giorno (indice {input_data.NUM_DAYS - 1}) puo' avere
   una Notte seguita da un altro turno l'indomani, violando H5.
   Esempio critico: Notte al giorno {input_data.NUM_DAYS - 2} -> i giorni
   {input_data.NUM_DAYS - 1} e {input_data.NUM_DAYS} devono essere liberi,
   ma il giorno {input_data.NUM_DAYS} non esiste: gestisci solo il giorno
   {input_data.NUM_DAYS - 1} con il check `if d + k < NUM_DAYS`.

**Vincolo 1 (finestra settimanale) - usa ESATTAMENTE questo pattern:**
```python
for w in WORKER_IDS:
    for t in range(NUM_DAYS - 6):      # finestre di 7 giorni: t, t+1, ..., t+6
        finestra = range(t, t + 7)
        model.Add(cp_model.LinearExpr.sum([SHIFT_HOURS[s] * x[(w, d, s)]
                      for d in finestra for s in SHIFT_CODES]) <= 36)
        model.Add(cp_model.LinearExpr.sum([x[(w, d, s)]
                      for d in finestra for s in SHIFT_CODES]) <= 6)
```

**Vincolo 3b (Notte->Mattina) - usa ESATTAMENTE questo pattern:**
```python
for w in WORKER_IDS:
    for d in range(NUM_DAYS - 1):      # NUM_DAYS - 1, NON NUM_DAYS
        model.Add(x[(w, d, 'N')] + x[(w, d + 1, 'M')] <= 1)
```
  TRAPPOLA COMUNE #2: usare `range(NUM_DAYS)` invece di `range(NUM_DAYS - 1)`,
   che creerebbe un accesso fuori bounds a x[(w, NUM_DAYS, 'M')].

### TRAPPOLE COMUNI DA EVITARE
- NON usare `solver.parameters.log_search_progress = True` (produce output indesiderato).
- NON chiamare `print()` in nessuna parte del codice.
- NON usare indici di giorno fuori dall'intervallo [0, {input_data.NUM_DAYS - 1}].
- Inizializza le variabili x PRIMA di usarle in qualsiasi vincolo.
- Popola RESULT_SCHEDULE come dict annidato: {{wid: {{day_index: codice_o_None}}}}.
  Usa None (non stringa vuota, non '-') per i giorni liberi.

### FUNZIONE OBIETTIVO (si applica nella FASE B, sul modello reale)
Massimizza un obiettivo UNICO = SODDISFAZIONE - PENALITA' FERIE - PENALITA' DI EQUITA'.
Poiche' CP-SAT lavora con interi, scala TUTTO per {SATISFACTION_SCALE} e arrotonda.

**(a) Soddisfazione + penalita' per i GIORNI INDESIDERATI (ferie).**
  - soddisfazione dei turni: sum( PREFERENCES[w]['satisfaction_weights'][s] * x[w,d,s] );
  - per ogni turno assegnato a w in un giorno d in UNDESIRED_DAYS[w], sottrai
    UNDESIRED_DAY_PENALTY. Questa penalita' e' molto alta: il solver evitera' con
    forza quei giorni, ma in emergenza potra' usarli (NON e' un divieto).
```python
obiettivo = []
for w in WORKER_IDS:
    for d in range(NUM_DAYS):
        for s in SHIFT_CODES:
            peso = int(round(PREFERENCES[w]['satisfaction_weights'][s] * {SATISFACTION_SCALE}))
            if d in UNDESIRED_DAYS[w]:
                peso -= int(round(UNDESIRED_DAY_PENALTY * {SATISFACTION_SCALE}))
            obiettivo.append(peso * x[(w, d, s)])
```

**(b) EQUITA' (Fairness-Oriented Allocation).** Distribuisci il carico gravoso in
modo equo penalizzando lo SCARTO (max - min) tra lavoratori di DUE quantita':
  - il numero di turni di Notte (il turno piu' gravoso);
  - il numero di VIOLAZIONI di ferie, cioe' turni assegnati a un lavoratore in un
    SUO giorno indesiderato (UNDESIRED_DAYS[w]). Cosi' le violazioni inevitabili
    non si concentrano su pochi sfortunati. NON bilanciare i "turni festivi
    grezzi": nell'orizzonte i festivi coincidono con le ferie, quindi bilanciarli
    forzerebbe i lavoratori proprio sui giorni che vogliono liberi.
Sono termini SOFT (non vincoli): spingono solo verso una ripartizione bilanciata.
Usa ESATTAMENTE questo pattern con variabili ausiliarie IntVar + AddMaxEquality/AddMinEquality:
```python
# --- bilanciamento delle Notti tra lavoratori ---
night_counts = []
for w in WORKER_IDS:
    nc = model.NewIntVar(0, NUM_DAYS, f"nights_{{w}}")
    model.Add(nc == cp_model.LinearExpr.sum([x[(w, d, 'N')] for d in range(NUM_DAYS)]))
    night_counts.append(nc)
max_n = model.NewIntVar(0, NUM_DAYS, "max_nights")
min_n = model.NewIntVar(0, NUM_DAYS, "min_nights")
model.AddMaxEquality(max_n, night_counts)
model.AddMinEquality(min_n, night_counts)
obiettivo.append(-int(round(NIGHT_BALANCE_PENALTY * {SATISFACTION_SCALE})) * (max_n - min_n))

# --- bilanciamento delle VIOLAZIONI di ferie tra lavoratori ---
violation_counts = []
for w in WORKER_IDS:
    vc = model.NewIntVar(0, NUM_DAYS, f"viol_{{w}}")
    model.Add(vc == cp_model.LinearExpr.sum([x[(w, d, s)] for d in UNDESIRED_DAYS[w] for s in SHIFT_CODES]))
    violation_counts.append(vc)
max_v = model.NewIntVar(0, NUM_DAYS, "max_viol")
min_v = model.NewIntVar(0, NUM_DAYS, "min_viol")
model.AddMaxEquality(max_v, violation_counts)
model.AddMinEquality(min_v, violation_counts)
obiettivo.append(-int(round(UNDESIRED_BALANCE_PENALTY * {SATISFACTION_SCALE})) * (max_v - min_v))

model.Maximize(cp_model.LinearExpr.sum(obiettivo))
```

### SCHELETRO DI CODICE OBBLIGATORIO (assemblalo ESATTAMENTE cosi')
Tutti i vincoli hard vanno dentro build_model(); risolvi in DUE FASI; popola
RESULT_SCHEDULE e SOLVER_STATUS dalla soluzione della FASE B.
```python
def build_model():
    model = cp_model.CpModel()
    x = {{(w, d, s): model.NewBoolVar(f"x_{{w}}_{{d}}_{{s}}")
          for w in WORKER_IDS for d in range(NUM_DAYS) for s in SHIFT_CODES}}
    # >>> QUI tutti i vincoli hard descritti sopra: max 1 turno/giorno, turni
    #     vietati, ESATTAMENTE 25 turni pesati (cp_model.LinearExpr.sum([SHIFT_WEIGHT[s]*x]) == 25),
    #     finestre settimanali (<=36h e <=6 turni), N->M, 2 riposi dopo la Notte,
    #     staffing dello use case. Usa 'model' e 'x'. NESSUN obiettivo qui dentro.
    return model, x

# --- FASE A: warm-start. STESSI vincoli hard (25 esatto), ma obiettivo =
#     MASSIMIZZA la somma pesata dei turni. La guida dell'LP spinge a riempire
#     fino a 25 e trova in pochi secondi un assetto FATTIBILE con tutti a 25,
#     da usare come hint per la FASE B. (Senza questo obiettivo la sola
#     feasibility del modello esatto puo' richiedere oltre 2 minuti.) ---
model_a, x_a = build_model()
model_a.Maximize(cp_model.LinearExpr.sum([SHIFT_WEIGHT[s] * x_a[(w, d, s)]
                     for w in WORKER_IDS for d in range(NUM_DAYS) for s in SHIFT_CODES]))
solver_a = cp_model.CpSolver()
solver_a.parameters.max_time_in_seconds = max(20.0, MAX_TIME * 0.4)
solver_a.parameters.num_search_workers = 8
solver_a.parameters.log_search_progress = False
status_a = solver_a.Solve(model_a)  # tipicamente OPTIMAL/FEASIBLE in ~10s
warm = None
if status_a in (cp_model.OPTIMAL, cp_model.FEASIBLE):
    warm = {{key: int(solver_a.Value(var)) for key, var in x_a.items()}}

# --- FASE B: modello reale con HINT dalla FASE A ---
model, x = build_model()
if warm is not None:
    for key, var in x.items():
        model.AddHint(var, warm[key])
obiettivo = []
# >>> costruisci 'obiettivo' ESATTAMENTE come nella sezione FUNZIONE OBIETTIVO:
#     termine (a) soddisfazione + penalita' ferie, termine (b) equita' notti/violazioni-ferie.
model.Maximize(cp_model.LinearExpr.sum(obiettivo))
solver = cp_model.CpSolver()
solver.parameters.max_time_in_seconds = max(15.0, MAX_TIME * 0.6)
solver.parameters.num_search_workers = 8
solver.parameters.log_search_progress = False
status = solver.Solve(model)

# --- OUTPUT: dalla soluzione della FASE B ---
RESULT_SCHEDULE = {{}}
for w in WORKER_IDS:
    RESULT_SCHEDULE[w] = {{}}
    for d in range(NUM_DAYS):
        code = None
        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            for s in SHIFT_CODES:
                if solver.Value(x[(w, d, s)]) == 1:
                    code = s
                    break
        RESULT_SCHEDULE[w][d] = code
SOLVER_STATUS = solver.StatusName(status)
```

NON stampare nulla, NON leggere/scrivere file, NON ridefinire le variabili gia'
disponibili. Restituisci SOLO un blocco di codice Python valido (racchiuso tra
```python e ```).
"""
    return prompt


def worker_satisfaction(data: ProblemData, schedule: dict, w: str) -> float:
    """
    Soddisfazione di un lavoratore secondo il modello unico usato in tutte le fasi:

        sat(w) = somma(satisfaction_weights[turno]  per ogni turno assegnato)
               - UNDESIRED_DAY_PENALTY * (numero di turni in giorni indesiderati)

    Il termine di penalita' rende coerenti l'obiettivo del solver (Fasi 2/4) con
    le metriche di equita' (Fase 3): un lavoratore costretto a lavorare in un
    giorno di ferie risulta correttamente molto piu' insoddisfatto.
    """
    pesi = data.preferences[w]["satisfaction_weights"]
    giorni_ind = data.undesired_days.get(w, set())
    totale = 0.0
    for d in range(input_data.NUM_DAYS):
        code = schedule[w].get(d)
        if code is None:
            continue
        totale += pesi[code]
        if d in giorni_ind:
            totale -= UNDESIRED_DAY_PENALTY
    return round(totale, 2)


# ---------------------------------------------------------------------------
# SODDISFAZIONE NORMALIZZATA (proportional fairness)
# ---------------------------------------------------------------------------
# La soddisfazione ASSOLUTA non e' confrontabile tra lavoratori(chi predilige
# la Notte, peso modesto e fortemente limitata dai riposi, ha un massimo di poche
# unita'; chi predilige Mattina/Pomeriggio puo' raggiungere valori di un ordine di
# grandezza superiore). Per misurare l'equita' "rispetto agli altri"
#  normalizziamo la soddisfazione sul MASSIMO INDIVIDUALE raggiungibile da
# ciascun lavoratore. Misura quanto un lavoratore e' vicino al PROPRIO ottimo, su una scala comune a tutti.
_SAT_MAX_CACHE: Dict[str, Dict[str, float]] = {}


def compute_sat_max(data: ProblemData) -> Dict[str, float]:
    """
    Per ogni lavoratore calcola la soddisfazione ASSOLUTA massima teorica
    ottenibile rispettando i SOLI vincoli hard INDIVIDUALI (esclusi i vincoli di
    copertura/staffing, che accoppiano piu' lavoratori): e' il "punto ideale" del
    lavoratore. Usato come denominatore per la
    soddisfazione normalizzata.
    """
    if data.case_label in _SAT_MAX_CACHE:
        return _SAT_MAX_CACHE[data.case_label]

    num_days = input_data.NUM_DAYS
    codes = input_data.SHIFT_CODES
    weight = {s: input_data.SHIFTS[s]["peso_turni"] for s in codes}
    hours = {s: input_data.SHIFTS[s]["durata_ore"] for s in codes}

    sat_max: Dict[str, float] = {}
    for w in data.worker_ids:
        model = cp_model.CpModel()
        x = {(d, s): model.NewBoolVar(f"x_{d}_{s}")
             for d in range(num_days) for s in codes}
       
        for d in range(num_days):
            model.AddAtMostOne(x[(d, s)] for s in codes)
        
        for s in data.forbidden.get(w, set()):
            for d in range(num_days):
                model.Add(x[(d, s)] == 0)
       
        model.Add(sum(
            weight[s] * x[(d, s)] for d in range(num_days) for s in codes) == 25)
       
        for t in range(num_days - 6):
            finestra = range(t, t + 7)
            model.Add(sum(
                hours[s] * x[(d, s)] for d in finestra for s in codes) <= 36)
            model.Add(sum(
                x[(d, s)] for d in finestra for s in codes) <= 6)
        
        for d in range(num_days - 1):
            model.Add(x[(d, "N")] + x[(d + 1, "M")] <= 1)
        # 2 giorni di riposo totale dopo ogni Notte.
        for d in range(num_days):
            for k in (1, 2):
                if d + k < num_days:
                    for s in codes:
                        model.Add(x[(d, "N")] + x[(d + k, s)] <= 1)
        # Obiettivo: massimizza la sola soddisfazione (pesi turno scalati).
        pesi = data.preferences[w]["satisfaction_weights"]
        model.Maximize(sum(
            int(round(pesi[s] * SATISFACTION_SCALE)) * x[(d, s)]
            for d in range(num_days) for s in codes))
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = 10.0
        solver.parameters.num_search_workers = 8
        solver.parameters.log_search_progress = False
        status = solver.Solve(model)
        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            sat_max[w] = round(solver.ObjectiveValue() / SATISFACTION_SCALE, 2)
        else:
            sat_max[w] = 0.0

    _SAT_MAX_CACHE[data.case_label] = sat_max
    return sat_max


def worker_satisfaction_pct(sat_abs: float, sat_max: float) -> float:
    """
    Soddisfazione NORMALIZZATA in percentuale: quanto il lavoratore e' vicino al
    proprio ottimo individuale. 100% = orario ideale per lui; valori bassi (anche
    negativi, se forzato su ferie/turni sgraditi) = lontano dal proprio ottimo.

    """
    denom = sat_max if sat_max > 0.1 else 0.1
    return round(100.0 * sat_abs / denom, 1)


def _build_llm_context(data: ProblemData, max_time: float) -> dict:
    """
    Pre-popola il namespace di esecuzione con i dati che il codice LLM usera'.
    
    """
    shifts = input_data.SHIFTS
    return {
        "WORKER_IDS": list(data.worker_ids),
        "NUM_DAYS": input_data.NUM_DAYS,
        "SHIFT_CODES": list(input_data.SHIFT_CODES),
        "SHIFT_HOURS": {s: shifts[s]["durata_ore"] for s in input_data.SHIFT_CODES},
        "SHIFT_WEIGHT": {s: shifts[s]["peso_turni"] for s in input_data.SHIFT_CODES},
        "PREFERENCES": data.preferences,
        "UNDESIRED_DAYS": {w: set(v) for w, v in data.undesired_days.items()},
        "UNDESIRED_DAY_PENALTY": UNDESIRED_DAY_PENALTY,
        "FORBIDDEN_SHIFTS": {w: set(v) for w, v in data.forbidden.items()},
        "HOLIDAY_DAYS": {
            i for i, d in enumerate(input_data.PLANNING_DATES)
            if d in input_data.HOLIDAYS
        },
        "NIGHT_BALANCE_PENALTY": NIGHT_BALANCE_PENALTY,
        "UNDESIRED_BALANCE_PENALTY": UNDESIRED_BALANCE_PENALTY,
        "STANDARD_IDS": list(data.standard_ids),
        "SPECIALIZED_IDS": list(data.specialized_ids),
        "MAX_TIME": max_time,
        "cp_model": cp_model,
    }


def build_schedule_result(
    data: ProblemData,
    raw_schedule: dict,
    status_name: str,
    source: str = "llm",
    generated_code: Optional[str] = None,
) -> ScheduleResult:
    """
    Costruisce una `ScheduleResult` normalizzata a partire dall'output grezzo di
    un'esecuzione cp_model.

    Centralizza la normalizzazione delle chiavi-giorno e il calcolo della
    soddisfazione per lavoratore, cosi' che sia la Fase 2 (bozza iniziale) sia la
    Fase 4 (raffinamento) producano risultati con la stessa identica semantica.
    """
    schedule: Dict[str, Dict[int, Optional[str]]] = {}
    for w in data.worker_ids:
        giorni = raw_schedule.get(w, {})
        schedule[w] = {
            int(d): giorni.get(d, giorni.get(int(d)))
            for d in range(input_data.NUM_DAYS)
        }

    result = ScheduleResult(
        case_label=data.case_label,
        status_name=str(status_name),
        feasible=any(s for g in schedule.values() for s in g.values()),
        schedule=schedule,
        source=source,
        generated_code=generated_code,
    )
    for w in data.worker_ids:
        result.satisfaction_per_worker[w] = worker_satisfaction(data, schedule, w)
    result.objective_value = round(sum(result.satisfaction_per_worker.values()), 2)
    return result


def save_generated_code(case_label: str, code: str, path: Optional[str] = None) -> str:
    """
    Salva su .txt il codice cp_model prodotto dall'LLM. E' anche l'input testuale della Fase 4.
    """
    if path is None:
        path = f"draft_code_case_{case_label}.txt"
    with open(path, "w", encoding="utf-8") as f:
        f.write(code)
    print(f"[+] Codice cp_model della bozza salvato in: {path}")
    return path


def run_llm_drafting(
    executor,
    data: ProblemData,
    max_time: float = 60.0,
    max_retries: int = 3,
    feedback: Optional[str] = None,
) -> Optional[ScheduleResult]:
    """
    Percorso LLM ibrido: l'LLM scrive il cp_model, il motore della Fase 0 lo
    esegue con auto-correzione. Ritorna ScheduleResult oppure ritorna None se fallisce.

    `feedback`: traccia d'errore prodotta dal Verification Agent
    (Fase 3) su una bozza precedente RIFIUTATA. Viene accodata al prompt cosi'
    che il Drafting Agent revisioni il piano correggendo le violazioni rilevate.
    """
    prompt = build_drafting_prompt(data)
    if feedback:
        prompt += (
            f"\n### FEEDBACK DEL VERIFICATION AGENT (bozza precedente RIFIUTATA)\n"
            f"La tua bozza precedente ha violato i vincoli hard. Traccia d'errore:\n"
            f"---\n{feedback}\n---\n"
            f"Genera una nuova versione del codice che NON commetta queste violazioni.\n"
        )
    context_vars = _build_llm_context(data, max_time)

    successo, risultato = executor.run_with_retry(
        prompt, context_vars=context_vars, max_retries=max_retries
    )
    if not successo:
        print(f"[!] Il percorso LLM non ha prodotto codice eseguibile per il Caso "
              f"{data.case_label}.\n    Ultimo errore:\n{risultato}")
        return None

    raw_schedule = risultato.get("RESULT_SCHEDULE")
    status_name = risultato.get("SOLVER_STATUS", "UNKNOWN")
    if not isinstance(raw_schedule, dict):
        print("[!] Il codice LLM non ha definito un 'RESULT_SCHEDULE' valido.")
        return None

    return build_schedule_result(
        data,
        raw_schedule,
        status_name,
        source="llm",
        generated_code=executor.last_code,
    )


# ===========================================================================
# 3. OUTPUT: STAMPA E SALVATAGGIO CSV
# ===========================================================================
def export_csv(data: ProblemData, result: ScheduleResult, path: Optional[str] = None) -> str:
    """Salva la schedulazione in CSV (righe = lavoratori, colonne = date)."""
    if path is None:
        path = f"schedule_case_{data.case_label}.csv"

    # Colonne finali: soddisfazione ASSOLUTA (modello di base) + il MASSIMO
    # individuale e la soddisfazione NORMALIZZATA in % (metrica di equita').
    sat_max = compute_sat_max(data)
    intestazioni = ["worker_id", "nome"] + [
        d.isoformat() for d in input_data.PLANNING_DATES
    ] + ["tot_turni_pesati", "soddisfazione", "sat_max", "soddisfazione_pct"]

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(intestazioni)
        weight = {s: input_data.SHIFTS[s]["peso_turni"] for s in input_data.SHIFT_CODES}
        for w in data.worker_ids:
            riga = [w, data.worker_names[w]]
            tot_peso = 0
            for d in range(input_data.NUM_DAYS):
                s = result.schedule[w].get(d)
                riga.append(s if s else "-")
                if s:
                    tot_peso += weight[s]
            sat_abs = result.satisfaction_per_worker.get(w, 0.0)
            riga.append(tot_peso)
            riga.append(sat_abs)
            riga.append(sat_max.get(w, 0.0))
            riga.append(worker_satisfaction_pct(sat_abs, sat_max.get(w, 0.0)))
            writer.writerow(riga)

    print(f"[+] Schedulazione salvata in: {path}")
    return path


def print_summary(data: ProblemData, result: ScheduleResult) -> None:
    """Stampa un riepilogo compatto della bozza generata."""
    print(f"\n{'='*64}")
    print(f"BOZZA SCHEDULAZIONE - Caso {data.case_label} "
          f"(sorgente: {result.source})")
    print(f"{'='*64}")
    print(f"Status solver        : {result.status_name}")
    print(f"Fattibile            : {result.feasible}")
    if not result.feasible:
        print("[!] Nessuna soluzione: i vincoli hard sono leggi e NON vengono "
              "rilassati. Rivedere i dati/lo staffing dello use case.")
        return

    print(f"Valore obiettivo     : {result.objective_value}")
    sat = result.satisfaction_per_worker
    if sat:
        peggiore = min(sat, key=sat.get)
        migliore = max(sat, key=sat.get)
        print(f"Soddisfazione totale : {round(sum(sat.values()), 2)}")
        print(f"Meno soddisfatto     : {peggiore} ({data.worker_names[peggiore]}) "
              f"= {sat[peggiore]}")
        print(f"Piu' soddisfatto     : {migliore} ({data.worker_names[migliore]}) "
              f"= {sat[migliore]}")

    # Distribuzione dei turni per giorno.
    print("\nConteggio turni assegnati per codice:")
    for s in input_data.SHIFT_CODES:
        tot = sum(
            1 for w in data.worker_ids for d in range(input_data.NUM_DAYS)
            if result.schedule[w][d] == s
        )
        print(f"  {s} ({input_data.SHIFTS[s]['nome']}): {tot}")

    # Distribuzione del carico GRAVOSO tra i lavoratori (equita' della Fase 2):
    # uno scarto max-min basso indica una ripartizione equa di Notti e violazioni
    # di ferie.
    notti = {
        w: sum(1 for d in range(input_data.NUM_DAYS) if result.schedule[w][d] == "N")
        for w in data.worker_ids
    }
    violazioni = {
        w: sum(
            1 for d in data.undesired_days.get(w, set())
            if result.schedule[w][d] is not None
        )
        for w in data.worker_ids
    }
    print("\nDistribuzione carico gravoso (equita' - scarto basso = piu' equo):")
    print(f"  Notti           per lavoratore: min {min(notti.values())}, "
          f"max {max(notti.values())}, scarto {max(notti.values()) - min(notti.values())}")
    print(f"  Violazioni ferie per lavoratore: min {min(violazioni.values())}, "
          f"max {max(violazioni.values())}, scarto {max(violazioni.values()) - min(violazioni.values())}")


# ===========================================================================
# 4. MAIN / CLI
# ===========================================================================
def run_case(case_label: str, max_time: float = 60.0) -> Optional[ScheduleResult]:
    """Esegue la Fase 2 (percorso LLM) per un singolo use case."""
    data = load_problem_data(case_label)

    print(f"\n{'#'*64}")
    print(f"# FASE 2 - DRAFTING AGENT (LLM) | Caso {data.case_label}")
    print(f"# Lavoratori: {len(data.worker_ids)} "
          f"(standard: {len(data.standard_ids)}, specializzati: {len(data.specialized_ids)})")
    print(f"{'#'*64}")

    # Inferenza via Google Gemini 2.5 Flash.
    from llm_engine import AgentExecutor
    executor = AgentExecutor()
    result = run_llm_drafting(executor, data, max_time=max_time)
    if result is None:
        return None

    print_summary(data, result)

    if result.feasible:
        # La validazione formale dei vincoli spetta alla Fase 3 (Verification
        # Agent): qui salviamo soltanto la bozza prodotta dall'LLM.
        export_csv(data, result)
        if result.generated_code:
            save_generated_code(data.case_label, result.generated_code)

    return result


def main():
    parser = argparse.ArgumentParser(
        description="SmartScheduler Fase 2 - Drafting Agent (percorso LLM)."
    )
    parser.add_argument(
        "--case", choices=["A", "B", "all"], default="all",
        help="Use case da risolvere (default: all).",
    )
    parser.add_argument(
        "--max-time", type=float, default=60.0,
        help="Tempo massimo di risoluzione del solver in secondi (default: 60).",
    )
    args = parser.parse_args()

    casi = ["A", "B"] if args.case == "all" else [args.case]
    for case_label in casi:
        run_case(case_label, max_time=args.max_time)


if __name__ == "__main__":
    main()

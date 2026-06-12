"""
drafting_agent.py
=================
Fase 2 - Drafting Agent (Bozza della Schedulazione).

La bozza e' generata ESCLUSIVAMENTE tramite l'LLM, come richiesto dal
PROJECT_CONTEXT.md:

  *Percorso LLM* (`build_drafting_prompt` + `run_llm_drafting`):
  il Drafting Agent costruisce un prompt dettagliato che istruisce l'LLM a
  SCRIVERE il codice `ortools.sat.python.cp_model`. Il codice generato viene
  eseguito in sicurezza tramite il motore della Fase 0
  (`AgentExecutor.run_with_retry`), con auto-correzione sugli errori.

Il percorso produce una struttura dati di output (`ScheduleResult`), cosi' le
Fasi 3-4 (verifica + raffinamento) possono consumarla. La validazione formale
dei vincoli hard NON e' compito di questa fase: spetta esclusivamente al
Verification Agent della Fase 3.

Esecuzione (richiede la variabile d'ambiente GEMINI_API_KEY):
    python drafting_agent.py --case A
    python drafting_agent.py --case B
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
# (satisfaction_weights), scalati a interi per la funzione obiettivo CP-SAT.
SATISFACTION_SCALE = 10  # i pesi hanno al massimo 1 cifra decimale -> *10.


@dataclass
class ProblemData:
    """Dati completi del problema per uno specifico use case (A o B)."""

    case_label: str
    worker_ids: List[str]
    worker_names: Dict[str, str]
    standard_ids: List[str]
    specialized_ids: List[str]
    preferences: Dict[str, dict]
    # Indisponibilita' come indici di giorno (0-based sull'orizzonte).
    unavailable: Dict[str, set]
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

    # Preferenze formalizzate prodotte dalla Fase 1.
    try:
        mod = importlib.import_module(f"formalized_preferences_case_{case_label}")
        preferences = dict(mod.WORKER_PREFERENCES)
    except ModuleNotFoundError as exc:
        raise FileNotFoundError(
            f"Manca 'formalized_preferences_case_{case_label}.py'. "
            f"Esegui prima la Fase 1 (workers_agent.py)."
        ) from exc

    worker_ids = [w["id"] for w in workers]
    worker_names = {w["id"]: w["nome"] for w in workers}
    standard_ids = [w["id"] for w in workers if w["ruolo"] == "standard"]
    specialized_ids = [w["id"] for w in workers if w["ruolo"] == "specializzato"]

    # Mappa data ISO -> indice giorno per tradurre le indisponibilita'.
    iso_to_index = {d.isoformat(): i for i, d in enumerate(input_data.PLANNING_DATES)}
    unavailable = {}
    for wid in worker_ids:
        giorni = preferences.get(wid, {}).get("giorni_indisponibilita", [])
        unavailable[wid] = {
            iso_to_index[g] for g in giorni if g in iso_to_index
        }

    return ProblemData(
        case_label=case_label,
        worker_ids=worker_ids,
        worker_names=worker_names,
        standard_ids=standard_ids,
        specialized_ids=specialized_ids,
        preferences=preferences,
        unavailable=unavailable,
        staffing=use_case["staffing"],
    )


# ===========================================================================
# 1. STRUTTURA DI OUTPUT DELLA BOZZA (percorso LLM)
# ===========================================================================
@dataclass
class ScheduleResult:
    """
    Risultato di una bozza di schedulazione.

    schedule[worker_id][day_index] = codice turno ('M'/'P'/'N') oppure None.
    """

    case_label: str
    status_name: str
    feasible: bool
    schedule: Dict[str, Dict[int, Optional[str]]] = field(default_factory=dict)
    objective_value: Optional[float] = None
    satisfaction_per_worker: Dict[str, float] = field(default_factory=dict)
    source: str = "llm"
    # Sorgente cp_model effettivamente eseguito che ha prodotto questa bozza
    # (valorizzato sul percorso LLM). E' l'input testuale della Fase 4.
    generated_code: Optional[str] = None


# ===========================================================================
# 2. PERCORSO LLM (l'LLM SCRIVE il cp_model)
# ===========================================================================
def build_drafting_prompt(data: ProblemData) -> str:
    """
    Costruisce il prompt che istruisce l'LLM a generare lo script
    `ortools.sat.python.cp_model` per la bozza di schedulazione.

    Il codice generato gira nel namespace di esecuzione (safe_execute), dove
    sono GIA' disponibili le seguenti variabili pre-caricate (vedi
    `run_llm_drafting`): non vanno ridefinite, vanno solo usate.
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
            f"    sum(x[(w, d, s)] for w in SPECIALIZED_IDS) >= {min_spec}\n"
            f"    sum(x[(w, d, s)] for w in WORKER_IDS) >= {min_std + min_spec}"
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
- UNAVAILABLE     : dict      -> {{wid: set(indici_giorno) in cui NON puo' lavorare}}
- STANDARD_IDS    : list[str] -> id dei lavoratori standard
- SPECIALIZED_IDS : list[str] -> id dei lavoratori specializzati (vuota nel Caso A)
- MAX_TIME        : float     -> tempo massimo di risoluzione in secondi
- cp_model        : modulo ortools.sat.python.cp_model gia' importato

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
{regola_staffing}

### IMPLEMENTAZIONE CORRETTA DEI VINCOLI - PATTERN OBBLIGATORI

**Vincolo 5 (riposi dopo Notte) - usa ESATTAMENTE questo pattern:**
```python
for w in WORKER_IDS:
    for d in range(NUM_DAYS):
        for k in range(1, 3):          # k=1 e k=2 (i due giorni di riposo)
            if d + k < NUM_DAYS:       # INDISPENSABILE: verifica che il giorno esista
                for s in SHIFT_CODES:
                    model.Add(x[(w, d, 'N')] + x[(w, d + k, s)] <= 1)
```
⚠️ TRAPPOLA COMUNE #1: dimenticare il controllo `if d + k < NUM_DAYS`.
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
        model.Add(sum(SHIFT_HOURS[s] * x[(w, d, s)]
                      for d in finestra for s in SHIFT_CODES) <= 36)
        model.Add(sum(x[(w, d, s)]
                      for d in finestra for s in SHIFT_CODES) <= 6)
```

**Vincolo 3b (Notte->Mattina) - usa ESATTAMENTE questo pattern:**
```python
for w in WORKER_IDS:
    for d in range(NUM_DAYS - 1):      # NUM_DAYS - 1, NON NUM_DAYS
        model.Add(x[(w, d, 'N')] + x[(w, d + 1, 'M')] <= 1)
```
⚠️ TRAPPOLA COMUNE #2: usare `range(NUM_DAYS)` invece di `range(NUM_DAYS - 1)`,
   che creerebbe un accesso fuori bounds a x[(w, NUM_DAYS, 'M')].

### TRAPPOLE COMUNI DA EVITARE
- NON usare `solver.parameters.log_search_progress = True` (produce output indesiderato).
- NON chiamare `print()` in nessuna parte del codice.
- NON usare indici di giorno fuori dall'intervallo [0, {input_data.NUM_DAYS - 1}].
- Inizializza le variabili x PRIMA di usarle in qualsiasi vincolo.
- Popola RESULT_SCHEDULE come dict annidato: {{wid: {{day_index: codice_o_None}}}}.
  Usa None (non stringa vuota, non '-') per i giorni liberi.

### FUNZIONE OBIETTIVO E SODDISFAZIONE
Massimizza la soddisfazione totale calcolata in questo modo:
Per ogni lavoratore la soddisfazione di base e':
   sum( PREFERENCES[w]['satisfaction_weights'][s] * x[w,d,s] ).
A questa va SOTTRATTA una penalita' di 10.0 per ogni turno assegnato in un giorno di indisponibilita' (d in UNAVAILABLE[w]).
Poiche' CP-SAT lavora con interi, scala sia i pesi che la penalita' per {SATISFACTION_SCALE} e arrotonda all'intero (es. penalita_scalata = 10 * {SATISFACTION_SCALE}).

### COSA DEVE PRODURRE IL CODICE
- Crea le variabili booleane x[(w,d,s)].
- Aggiungi TUTTI i vincoli sopra e l'obiettivo.
- Risolvi con cp_model.CpSolver() (imposta max_time_in_seconds = MAX_TIME,
  num_search_workers = 8, log_search_progress = False).
- Poi DEVI popolare nel namespace queste due variabili:
    RESULT_SCHEDULE : dict {{wid: {{day_index: codice_turno_o_None}}}}
    SOLVER_STATUS   : str con il nome dello status (es. solver.StatusName(status))

NON stampare nulla, NON leggere/scrivere file, NON ridefinire le variabili gia'
disponibili. Restituisci SOLO un blocco di codice Python valido (racchiuso tra
```python e ```).
"""
    return prompt


def _build_llm_context(data: ProblemData, max_time: float) -> dict:
    """Pre-popola il namespace di esecuzione con i dati che il codice LLM usera'."""
    shifts = input_data.SHIFTS
    return {
        "WORKER_IDS": list(data.worker_ids),
        "NUM_DAYS": input_data.NUM_DAYS,
        "SHIFT_CODES": list(input_data.SHIFT_CODES),
        "SHIFT_HOURS": {s: shifts[s]["durata_ore"] for s in input_data.SHIFT_CODES},
        "SHIFT_WEIGHT": {s: shifts[s]["peso_turni"] for s in input_data.SHIFT_CODES},
        "PREFERENCES": data.preferences,
        "UNAVAILABLE": {w: set(v) for w, v in data.unavailable.items()},
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
    un'esecuzione cp_model (il dict `RESULT_SCHEDULE` lasciato nel namespace).

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
        pesi = data.preferences[w]["satisfaction_weights"]
        soddisfazione_base = sum(pesi[s] for d in range(input_data.NUM_DAYS)
                                 for s in input_data.SHIFT_CODES if schedule[w].get(d) == s)
        # Sottrai penalita' di 10.0 per ogni turno in un giorno indisponibile
        indisp = data.unavailable.get(w, set())
        turni_indisp = sum(1 for d in range(input_data.NUM_DAYS) 
                           if schedule[w].get(d) is not None and d in indisp)
        result.satisfaction_per_worker[w] = round(soddisfazione_base - (turni_indisp * 10.0), 2)
    result.objective_value = round(sum(result.satisfaction_per_worker.values()), 2)
    return result


def save_generated_code(case_label: str, code: str, path: Optional[str] = None) -> str:
    """
    Salva su .txt il codice cp_model prodotto dall'LLM (deliverable: "il modello
    cp_model parziale generato dall'LLM"). E' anche l'input testuale della Fase 4.
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
    max_time: float = 30.0,
    max_retries: int = 3,
    feedback: Optional[str] = None,
) -> Optional[ScheduleResult]:
    """
    Percorso LLM ibrido: l'LLM scrive il cp_model, il motore della Fase 0 lo
    esegue con auto-correzione. Ritorna ScheduleResult oppure None se fallisce.

    `feedback` (opzionale): traccia d'errore prodotta dal Verification Agent
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

    intestazioni = ["worker_id", "nome"] + [
        d.isoformat() for d in input_data.PLANNING_DATES
    ] + ["tot_turni_pesati", "soddisfazione"]

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
            riga.append(tot_peso)
            riga.append(result.satisfaction_per_worker.get(w, 0.0))
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

    # Distribuzione dei turni per giorno (controllo copertura a colpo d'occhio).
    print("\nConteggio turni assegnati per codice:")
    for s in input_data.SHIFT_CODES:
        tot = sum(
            1 for w in data.worker_ids for d in range(input_data.NUM_DAYS)
            if result.schedule[w][d] == s
        )
        print(f"  {s} ({input_data.SHIFTS[s]['nome']}): {tot}")


# ===========================================================================
# 4. MAIN / CLI
# ===========================================================================
def run_case(case_label: str, max_time: float = 30.0) -> Optional[ScheduleResult]:
    """Esegue la Fase 2 (percorso LLM) per un singolo use case."""
    data = load_problem_data(case_label)

    print(f"\n{'#'*64}")
    print(f"# FASE 2 - DRAFTING AGENT (LLM) | Caso {data.case_label}")
    print(f"# Lavoratori: {len(data.worker_ids)} "
          f"(standard: {len(data.standard_ids)}, specializzati: {len(data.specialized_ids)})")
    print(f"{'#'*64}")

    # Inferenza via Google Gemini 2.5 Flash: richiede GEMINI_API_KEY.
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
        "--max-time", type=float, default=30.0,
        help="Tempo massimo di risoluzione del solver in secondi (default: 30).",
    )
    args = parser.parse_args()

    casi = ["A", "B"] if args.case == "all" else [args.case]
    for case_label in casi:
        run_case(case_label, max_time=args.max_time)


if __name__ == "__main__":
    main()

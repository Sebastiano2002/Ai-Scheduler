"""
drafting_agent.py
=================
Fase 2 - Drafting Agent (Bozza della Schedulazione).

Architettura ESCLUSIVA tramite LLM, come da traccia:
Il Drafting Agent costruisce un prompt dettagliato che istruisce l'LLM a
SCRIVERE il codice `ortools.sat.python.cp_model`. Il codice generato viene
eseguito in sicurezza tramite il motore della Fase 0 (`AgentExecutor.run_with_retry`),
con auto-correzione sugli errori.

Esecuzione:
    python drafting_agent.py --case A
    python drafting_agent.py --case B
"""

import argparse
import csv
import datetime
import importlib
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from ortools.sat.python import cp_model

import input_data

# ===========================================================================
# 0. CARICAMENTO DEI DATI DEL PROBLEMA PER UNO USE CASE
# ===========================================================================
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
# 1. STRUTTURA DI OUTPUT
# ===========================================================================
@dataclass
class ScheduleResult:
    """Risultato di una bozza di schedulazione."""
    case_label: str
    status_name: str
    feasible: bool
    schedule: Dict[str, Dict[int, Optional[str]]] = field(default_factory=dict)
    objective_value: Optional[float] = None
    satisfaction_per_worker: Dict[str, float] = field(default_factory=dict)
    source: str = "llm"

# ===========================================================================
# 2. PERCORSO LLM (l'LLM SCRIVE il cp_model)
# ===========================================================================
def build_drafting_prompt(data: ProblemData) -> str:
    """Costruisce il prompt che istruisce l'LLM a generare lo script CP-SAT."""
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
        regola_staffing = (
            f"- Caso B: per OGNI turno di OGNI giorno almeno "
            f"{data.staffing['min_standard_per_turno']} lavoratori 'standard' "
            f"(STANDARD_IDS) e almeno {data.staffing['min_specializzati_per_turno']} "
            f"'specializzato' (SPECIALIZED_IDS)."
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
- cp_model        : modulo ortools.sat.python.cp_model gia' importato

### VINCOLI HARD DA CODIFICARE (sono LEGGI inviolabili)
1. Max 36 ore settimanali per lavoratore (per ogni finestra di 7 giorni consecutivi).
2. ESATTAMENTE 25 turni/mese per lavoratore, conteggiati col peso (la Notte vale 2):
   sum(SHIFT_WEIGHT[s] * x[w,d,s]) == 25.
3. Max 1 turno al giorno. Divieto ASSOLUTO solo della catena Notte(d)->Mattina(d+1).
   (NON significa "vietato lavorare in due giorni consecutivi": i turni di giorno
   consecutivi sono permessi.)
4. La Notte e' un turno doppio: gia' riflesso in SHIFT_WEIGHT e SHIFT_HOURS.
5. Dopo OGNI turno di Notte: 2 giorni di riposo TOTALE (nessun turno in d+1 e d+2). Fai molta attenzione ai limiti dell'orizzonte: assicurati che se d+1 < NUM_DAYS e d+2 < NUM_DAYS, non ci siano turni. Per evitare TypeError usa una disuguaglianza algebrica: `model.Add(x[w,d,'N'] + x[w,d+1,s] <= 1)` per ogni s, e analogamente per d+2.
6. Almeno 1 giorno di riposo a settimana (per ogni finestra di 7 giorni: <=6 lavorati).
- INDISPONIBILITA': nessun turno nei giorni in UNAVAILABLE[w].
{regola_staffing}

### FUNZIONE OBIETTIVO
Massimizza la soddisfazione totale:
   sum( PREFERENCES[w]['satisfaction_weights'][s] * x[w,d,s] ).
Poiche' CP-SAT lavora con interi, scala i pesi per {SATISFACTION_SCALE} e arrotonda
all'intero (int(round(peso * {SATISFACTION_SCALE}))).

### COSA DEVE PRODURRE IL CODICE
- Crea le variabili booleane x[(w,d,s)].
- Aggiungi TUTTI i vincoli sopra e l'obiettivo.
- Risolvi con cp_model.CpSolver() (imposta max_time_in_seconds = 30).
- Poi DEVI popolare nel namespace queste due variabili:
    RESULT_SCHEDULE : dict {{wid: {{day_index: codice_turno_o_None}}}}
    SOLVER_STATUS   : str con il nome dello status (es. solver.StatusName(status))

ATTENZIONE (MOLTO IMPORTANTE):
- Usa ESCLUSIVAMENTE list comprehensions `[...]` invece di generator expressions `(...)` dentro `sum()`, `AddAtMostOne()`, `AddExactlyOne()`, `AddBoolOr()`, ecc. (es. scrivi `sum([x[(w,d,s)] for s in ...])` e non `sum(x[(w,d,s)] for s in ...)`). Altrimenti il codice fallirà con un NameError dovuto allo scope di `exec()`.

NON stampare nulla, NON leggere/scrivere file, NON ridefinire le variabili gia'
disponibili. Restituisci SOLO un blocco di codice Python valido (racchiuso tra
```python e ```).
"""
    return prompt

def _build_llm_context(data: ProblemData) -> dict:
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
        "cp_model": cp_model,
    }

def run_llm_drafting(executor, data: ProblemData, max_retries: int = 3) -> Optional[ScheduleResult]:
    """
    L'LLM scrive il cp_model, il motore della Fase 0 lo esegue con auto-correzione.
    Ritorna ScheduleResult oppure None se fallisce.
    """
    prompt = build_drafting_prompt(data)
    context_vars = _build_llm_context(data)

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

    schedule: Dict[str, Dict[int, Optional[str]]] = {}
    for w in data.worker_ids:
        giorni = raw_schedule.get(w, {})
        schedule[w] = {int(d): giorni.get(d, giorni.get(int(d))) for d in range(input_data.NUM_DAYS)}

    result = ScheduleResult(
        case_label=data.case_label,
        status_name=str(status_name),
        feasible=any(s for g in schedule.values() for s in g.values()),
        schedule=schedule,
        source="llm",
    )
    for w in data.worker_ids:
        pesi = data.preferences[w]["satisfaction_weights"]
        result.satisfaction_per_worker[w] = round(
            sum(pesi[s] for d in range(input_data.NUM_DAYS)
                for s in input_data.SHIFT_CODES if schedule[w].get(d) == s),
            2,
        )
    result.objective_value = round(sum(result.satisfaction_per_worker.values()), 2)
    return result

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

    print("\nConteggio turni assegnati per codice:")
    for s in input_data.SHIFT_CODES:
        tot = sum(
            1 for w in data.worker_ids for d in range(input_data.NUM_DAYS)
            if result.schedule[w][d] == s
        )
        print(f"  {s} ({input_data.SHIFTS[s]['nome']}): {tot}")

# ===========================================================================
# 3. MAIN / CLI
# ===========================================================================
def run_case(case_label: str) -> Optional[ScheduleResult]:
    """Esegue la Fase 2 per un singolo use case tramite LLM."""
    data = load_problem_data(case_label)

    print(f"\n{'#'*64}")
    print(f"# FASE 2 - DRAFTING AGENT | Caso {data.case_label} | modo: llm")
    print(f"# Lavoratori: {len(data.worker_ids)} "
          f"(standard: {len(data.standard_ids)}, specializzati: {len(data.specialized_ids)})")
    print(f"{'#'*64}")

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise SystemExit(
            "[!] Il Drafting Agent richiede GEMINI_API_KEY.\n"
            "    PowerShell: $env:GEMINI_API_KEY = 'la-tua-chiave'"
        )
    from llm_engine import AgentExecutor
    executor = AgentExecutor(api_key=api_key)
    result = run_llm_drafting(executor, data)
    if result is None:
        return None

    print_summary(data, result)
    return result

def main():
    parser = argparse.ArgumentParser(
        description="SmartScheduler Fase 2 - Drafting Agent (solo LLM)."
    )
    parser.add_argument(
        "--case", choices=["A", "B", "all"], default="all",
        help="Use case da risolvere (default: all).",
    )
    args = parser.parse_args()

    casi = ["A", "B"] if args.case == "all" else [args.case]
    for case_label in casi:
        run_case(case_label)

if __name__ == "__main__":
    main()

"""
drafting_agent.py
=================
Fase 2 - Drafting Agent (Bozza della Schedulazione).

Architettura IBRIDA (Opzione 1), fedele al PROJECT_CONTEXT.md:

  1. *Percorso LLM* (`build_drafting_prompt` + `run_llm_drafting`):
     il Drafting Agent costruisce un prompt dettagliato che istruisce l'LLM a
     SCRIVERE il codice `ortools.sat.python.cp_model`. Il codice generato viene
     eseguito in sicurezza tramite il motore della Fase 0
     (`AgentExecutor.run_with_retry`), con auto-correzione sugli errori.

  2. *Builder deterministico di riferimento/fallback*
     (`build_cp_model` + `solve`):
     un modello CP-SAT scritto a mano in Python che codifica ESATTAMENTE i 6
     vincoli hard del problema piu' la copertura (staffing) dei due use case.
     Serve a:
       - testare la logica di validazione dei vincoli hard (in particolare il
         Caso B con i lavoratori specializzati);
       - verificare l'equita' / la funzione obiettivo;
       - garantire una bozza OR-Tools funzionante ANCHE senza API key.

Entrambi i percorsi producono la stessa struttura dati di output
(`ScheduleResult`), cosi' le Fasi 3-4 (verifica + raffinamento) possono
consumare indifferentemente la bozza dell'LLM o quella deterministica.

Esecuzione:
    # Builder deterministico (NON richiede API key) - default:
    python drafting_agent.py --case A
    python drafting_agent.py --case B

    # Percorso LLM ibrido (richiede GEMINI_API_KEY):
    python drafting_agent.py --case A --mode llm
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
# 1. STRUTTURA DI OUTPUT (condivisa dai due percorsi: LLM e deterministico)
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
    source: str = "deterministic"  # "deterministic" oppure "llm".


# ===========================================================================
# 2. BUILDER DETERMINISTICO CP-SAT (riferimento / fallback)
# ===========================================================================
def build_cp_model(data: ProblemData):
    """
    Costruisce il modello CP-SAT codificando i 6 VINCOLI HARD del problema
    (PROJECT_CONTEXT.md) piu' la copertura per lo use case.

    Variabili decisionali:
        x[(w, d, s)] in {0,1} = 1 se il lavoratore w svolge il turno s nel
        giorno d (indice 0-based sull'orizzonte).

    Ritorna (model, x) cosi' che `solve` possa estrarre la soluzione e il
    percorso LLM possa, se serve, confrontarsi con lo stesso schema.
    """
    model = cp_model.CpModel()
    num_days = input_data.NUM_DAYS
    shift_codes = input_data.SHIFT_CODES  # ["M", "P", "N"]
    shifts = input_data.SHIFTS
    hc = input_data.HARD_CONSTRAINTS

    # Ore e peso (turno doppio) per ciascun codice turno.
    shift_hours = {s: shifts[s]["durata_ore"] for s in shift_codes}      # M/P=6, N=12
    shift_weight = {s: shifts[s]["peso_turni"] for s in shift_codes}     # M/P=1, N=2

    # --- Variabili decisionali ---
    x = {
        (w, d, s): model.NewBoolVar(f"x_{w}_{d}_{s}")
        for w in data.worker_ids
        for d in range(num_days)
        for s in shift_codes
    }

    # -----------------------------------------------------------------------
    # VINCOLO HARD #3 (parte 1): MASSIMO 1 TURNO AL GIORNO per lavoratore.
    # -----------------------------------------------------------------------
    for w in data.worker_ids:
        for d in range(num_days):
            model.Add(sum(x[(w, d, s)] for s in shift_codes) <= hc["max_turni_per_giorno"])

    # -----------------------------------------------------------------------
    # VINCOLO HARD #2: ESATTAMENTE 25 TURNI AL MESE (la Notte vale 2 -> peso).
    # -----------------------------------------------------------------------
    for w in data.worker_ids:
        model.Add(
            sum(shift_weight[s] * x[(w, d, s)] for d in range(num_days) for s in shift_codes)
            == hc["turni_mensili_esatti"]
        )

    # -----------------------------------------------------------------------
    # VINCOLO HARD #1: MAX 36 ORE SETTIMANALI.
    # VINCOLO HARD #6: ALMENO 1 GIORNO DI RIPOSO A SETTIMANA.
    # Interpretazione rigorosa: per OGNI finestra di 7 giorni consecutivi
    #   - somma ore <= 36;
    #   - giorni lavorati <= 6 (=> almeno 1 riposo nella settimana).
    # La finestra mobile e' la lettura piu' forte di "settimanale".
    # -----------------------------------------------------------------------
    week = 7
    for w in data.worker_ids:
        for t in range(0, num_days - week + 1):
            giorni_finestra = range(t, t + week)
            model.Add(
                sum(
                    shift_hours[s] * x[(w, d, s)]
                    for d in giorni_finestra
                    for s in shift_codes
                )
                <= hc["max_ore_settimanali"]
            )
            model.Add(
                sum(x[(w, d, s)] for d in giorni_finestra for s in shift_codes)
                <= week - hc["giorni_riposo_minimi"]
            )

    # -----------------------------------------------------------------------
    # VINCOLO HARD #5: 2 GIORNI LIBERI OBBLIGATORI DOPO OGNI TURNO DI NOTTE.
    # Se w lavora di Notte nel giorno d, allora NON lavora alcun turno in
    # d+1 e d+2 (riposo totale). Questo SUBSUME gia' il divieto Notte->Mattina.
    # -----------------------------------------------------------------------
    riposi = hc["riposi_obbligatori_dopo_notte"]
    for w in data.worker_ids:
        for d in range(num_days):
            for k in range(1, riposi + 1):
                if d + k < num_days:
                    for s in shift_codes:
                        model.Add(x[(w, d, "N")] + x[(w, d + k, s)] <= 1)

    # -----------------------------------------------------------------------
    # VINCOLO HARD #3 (parte 2): DIVIETO ASSOLUTO Notte(d) -> Mattina(d+1).
    # Reso esplicito per una mappatura legge->codice 1:1 (ridondante con #5,
    # ma documenta chiaramente l'unica catena cross-day proibita).
    # -----------------------------------------------------------------------
    for w in data.worker_ids:
        for d in range(num_days - 1):
            model.Add(x[(w, d, "N")] + x[(w, d + 1, "M")] <= 1)

    # -----------------------------------------------------------------------
    # VINCOLO (da preferenze Fase 1): INDISPONIBILITA' = giorni NON lavorabili.
    # "non puo' lavorare" -> hard: nessun turno in quei giorni.
    # -----------------------------------------------------------------------
    for w in data.worker_ids:
        for d in data.unavailable.get(w, set()):
            for s in shift_codes:
                model.Add(x[(w, d, s)] == 0)

    # -----------------------------------------------------------------------
    # COPERTURA / STAFFING (dipende dallo use case).
    #   Caso A: almeno 2 lavoratori per ogni turno.
    #   Caso B: almeno 2 standard + almeno 1 specializzato per ogni turno.
    # -----------------------------------------------------------------------
    if data.case_label == "A":
        min_per_turno = data.staffing["min_lavoratori_per_turno"]
        for d in range(num_days):
            for s in shift_codes:
                model.Add(
                    sum(x[(w, d, s)] for w in data.worker_ids) >= min_per_turno
                )
    else:  # Caso B
        min_std = data.staffing["min_standard_per_turno"]
        min_spec = data.staffing["min_specializzati_per_turno"]
        for d in range(num_days):
            for s in shift_codes:
                # Almeno `min_std` standard per turno.
                model.Add(
                    sum(x[(w, d, s)] for w in data.standard_ids) >= min_std
                )
                # Almeno `min_spec` specializzati per turno.
                model.Add(
                    sum(x[(w, d, s)] for w in data.specialized_ids) >= min_spec
                )

    # -----------------------------------------------------------------------
    # FUNZIONE OBIETTIVO: MASSIMIZZARE LA SODDISFAZIONE COMPLESSIVA.
    # Usa i satisfaction_weights formalizzati nella Fase 1, scalati a interi.
    # -----------------------------------------------------------------------
    obj_terms = []
    for w in data.worker_ids:
        pesi = data.preferences[w]["satisfaction_weights"]
        for d in range(num_days):
            for s in shift_codes:
                coef = int(round(pesi[s] * SATISFACTION_SCALE))
                if coef != 0:
                    obj_terms.append(coef * x[(w, d, s)])
    model.Maximize(sum(obj_terms))

    return model, x


def solve(data: ProblemData, max_time_seconds: float = 30.0,
          log_progress: bool = False) -> ScheduleResult:
    """
    Risolve il modello deterministico e impacchetta la soluzione in
    ScheduleResult. Se il modello e' INFEASIBLE lo segnala esplicitamente
    (senza rilassare i vincoli: i vincoli hard sono leggi).
    """
    model, x = build_cp_model(data)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = max_time_seconds
    solver.parameters.num_search_workers = 8
    solver.parameters.log_search_progress = log_progress

    status = solver.Solve(model)
    status_name = solver.StatusName(status)
    feasible = status in (cp_model.OPTIMAL, cp_model.FEASIBLE)

    result = ScheduleResult(
        case_label=data.case_label,
        status_name=status_name,
        feasible=feasible,
        source="deterministic",
    )

    if not feasible:
        return result

    # --- Estrazione della soluzione ---
    schedule: Dict[str, Dict[int, Optional[str]]] = {}
    for w in data.worker_ids:
        schedule[w] = {}
        for d in range(input_data.NUM_DAYS):
            assegnato = None
            for s in input_data.SHIFT_CODES:
                if solver.Value(x[(w, d, s)]) == 1:
                    assegnato = s
                    break
            schedule[w][d] = assegnato
    result.schedule = schedule

    result.objective_value = solver.ObjectiveValue() / SATISFACTION_SCALE

    # Soddisfazione per lavoratore (pesi reali, non scalati).
    for w in data.worker_ids:
        pesi = data.preferences[w]["satisfaction_weights"]
        tot = sum(
            pesi[s]
            for d in range(input_data.NUM_DAYS)
            for s in input_data.SHIFT_CODES
            if schedule[w][d] == s
        )
        result.satisfaction_per_worker[w] = round(tot, 2)

    return result


# ===========================================================================
# 3. PERCORSO LLM (architettura originale: l'LLM SCRIVE il cp_model)
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
5. Dopo OGNI turno di Notte: 2 giorni di riposo TOTALE (nessun turno in d+1 e d+2).
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
    Percorso LLM ibrido: l'LLM scrive il cp_model, il motore della Fase 0 lo
    esegue con auto-correzione. Ritorna ScheduleResult oppure None se fallisce.
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

    # Normalizza la struttura (chiavi giorno -> int) e calcola la soddisfazione.
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

    # Distribuzione dei turni per giorno (controllo copertura a colpo d'occhio).
    print("\nConteggio turni assegnati per codice:")
    for s in input_data.SHIFT_CODES:
        tot = sum(
            1 for w in data.worker_ids for d in range(input_data.NUM_DAYS)
            if result.schedule[w][d] == s
        )
        print(f"  {s} ({input_data.SHIFTS[s]['nome']}): {tot}")


# ===========================================================================
# 6. MAIN / CLI
# ===========================================================================
def run_case(case_label: str, mode: str, max_time: float = 30.0) -> Optional[ScheduleResult]:
    """Esegue la Fase 2 per un singolo use case nel modo richiesto."""
    data = load_problem_data(case_label)

    print(f"\n{'#'*64}")
    print(f"# FASE 2 - DRAFTING AGENT | Caso {data.case_label} | modo: {mode}")
    print(f"# Lavoratori: {len(data.worker_ids)} "
          f"(standard: {len(data.standard_ids)}, specializzati: {len(data.specialized_ids)})")
    print(f"{'#'*64}")

    if mode == "llm":
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise SystemExit(
                "[!] Modo 'llm' richiede GEMINI_API_KEY.\n"
                "    PowerShell: $env:GEMINI_API_KEY = 'la-tua-chiave'\n"
                "    Oppure usa il builder deterministico: --mode deterministic"
            )
        from llm_engine import AgentExecutor
        executor = AgentExecutor(api_key=api_key)
        result = run_llm_drafting(executor, data)
        if result is None:
            return None
    else:
        result = solve(data, max_time_seconds=max_time)

    print_summary(data, result)
    return result


def main():
    parser = argparse.ArgumentParser(
        description="SmartScheduler Fase 2 - Drafting Agent (approccio ibrido)."
    )
    parser.add_argument(
        "--case", choices=["A", "B", "all"], default="all",
        help="Use case da risolvere (default: all).",
    )
    parser.add_argument(
        "--mode", choices=["deterministic", "llm"], default="deterministic",
        help="'deterministic' (builder CP-SAT, no API key) o 'llm' (l'LLM scrive il cp_model).",
    )
    parser.add_argument(
        "--max-time", type=float, default=30.0,
        help="Tempo massimo di risoluzione del solver in secondi (default: 30).",
    )
    args = parser.parse_args()

    casi = ["A", "B"] if args.case == "all" else [args.case]
    for case_label in casi:
        run_case(case_label, args.mode, max_time=args.max_time)


if __name__ == "__main__":
    main()

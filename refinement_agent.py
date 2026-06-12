"""
refinement_agent.py
===================
Fase 4 - Raffinamento della Schedulazione (Ciclo Iterativo).

Chiude l'architettura multi-agente di SmartScheduler implementando il ciclo
di ottimizzazione dell'equita' descritto nel PROJECT_CONTEXT.md:

  - *Prompt di Feedback*: il Drafting Agent viene reistruito a raffinare la
    bozza CORRENTE con l'obiettivo specifico di migliorare la soddisfazione del
    *lavoratore meno soddisfatto* identificato dalla Fase 3.
  - *Vincolo di ottimizzazione*: il raffinamento NON DEVE far scendere nessun
    altro lavoratore sotto il livello minimo di soddisfazione attuale; tutti i
    vincoli hard restano LEGGI inviolabili.
  - *Terminazione*: il ciclo continua finche' non si raggiunge il limite massimo
    di iterazioni oppure finche' il solutore non restituisce INFEASIBLE/fallisce
    (non e' piu' possibile alzare il livello minimo di equita' -> ottimizzazione
    terminata).

Flusso di un'iterazione:
    1. costruisce un prompt che INCLUDE il codice cp_model della bozza corrente
       e chiede di migliorare il lavoratore peggiore senza danneggiare gli altri;
    2. esegue il nuovo codice con AgentExecutor.run_with_retry (motore Fase 0);
    3. ricostruisce la ScheduleResult e la passa al Verification Agent (Fase 3);
    4. se i vincoli hard sono violati -> SCARTA la bozza;
       se sono rispettati e il nuovo minimo globale e' MIGLIORE -> ACCETTA la
       bozza come nuovo orario di riferimento.

Esecuzione (richiede la variabile d'ambiente GEMINI_API_KEY):
    python refinement_agent.py --case A
    python refinement_agent.py --case B --max-iterations 3
    # Riparte da una bozza gia' salvata (CSV + .txt del codice) senza ri-fase 2:
    python refinement_agent.py --case A --from-draft
"""

import argparse
import os
from dataclasses import dataclass, field
from typing import List, Optional

from dotenv import load_dotenv
load_dotenv()

import input_data
from drafting_agent import (
    ProblemData,
    ScheduleResult,
    SATISFACTION_SCALE,
    _build_llm_context,
    build_schedule_result,
    export_csv,
    load_problem_data,
    run_llm_drafting,
    save_generated_code,
)
from verification_agent import (
    VerificationReport,
    load_schedule_from_csv,
    print_report,
    verify_schedule,
)


# ===========================================================================
# 1. PROMPT DI RAFFINAMENTO (Drafting Agent reistruito)
# ===========================================================================
def build_refinement_prompt(
    data: ProblemData,
    current_code: str,
    worst_worker_id: str,
    worst_worker_name: str,
    current_min: float,
    new_floor_scaled: int,
) -> str:
    """
    Costruisce il prompt che chiede all'LLM di MODIFICARE il codice cp_model
    della bozza corrente per migliorare il lavoratore piu' svantaggiato, senza
    far scendere nessun altro sotto il livello minimo attuale e mantenendo
    intatti tutti i vincoli hard.
    """
    floor_scaled = int(round(current_min * SATISFACTION_SCALE))

    return f"""Sei il "Drafting Agent" di SmartScheduler nella FASE 4 (raffinamento iterativo
dell'equita'). Hai gia' prodotto una bozza VALIDA per lo USE CASE {data.case_label}:
il codice cp_model qui sotto rispetta tutti i vincoli hard. Ora devi MIGLIORARLO.

### CODICE cp_model DELLA BOZZA CORRENTE (da modificare, NON da riscrivere da zero)
```python
{current_code}
```

### OBIETTIVO DEL RAFFINAMENTO
Modifica questo codice per migliorare il punteggio di soddisfazione del lavoratore
{worst_worker_id} ({worst_worker_name}), che e' attualmente il piu' svantaggiato
con un punteggio di {current_min}.

REGOLA FONDAMENTALE (inderogabile):
- DEVI restituire il codice precedente ESATTAMENTE COM'È, riga per riga.
- Aggiungi il nuovo blocco di codice per l'equità e l'hinting ESCLUSIVAMENTE alla fine, subito prima della chiamata a solver.Solve(model).
- È severamente vietato alterare, riassumere o rimuovere i vincoli hard precedenti.
- NON devi far scendere il punteggio di NESSUN altro lavoratore al di sotto di
  {current_min} (il livello minimo di soddisfazione attuale).

### COME IMPORLO NEL MODELLO (pattern obbligatorio)
Nel namespace sono GIA' disponibili, oltre alle variabili della bozza:
- SATISFACTION_SCALE : int -> {SATISFACTION_SCALE} (fattore di scala dei pesi)
- WORST_WORKER_ID    : str -> '{worst_worker_id}' (il lavoratore da migliorare)
- NEW_FLOOR_SCALED   : int -> {new_floor_scaled} (nuovo pavimento di equita', scalato)

La soddisfazione (scalata a interi) di un lavoratore w e':
```python
sat_w = sum(int(round(PREFERENCES[w]['satisfaction_weights'][s] * SATISFACTION_SCALE)) * x[(w, d, s)]
            for d in range(NUM_DAYS) for s in SHIFT_CODES)
penalita_scalata = int(round(10.0 * SATISFACTION_SCALE))
for d in UNAVAILABLE.get(w, []):
    for s in SHIFT_CODES:
        sat_w -= penalita_scalata * x[(w, d, s)]
```
Aggiungi, PRIMA di risolvere, un vincolo di EQUITA' che alza il pavimento per
TUTTI i lavoratori al nuovo livello richiesto (cosi' il minimo globale sale e
nessuno scende sotto quello attuale):
```python
for w in WORKER_IDS:
    sat_w = sum(int(round(PREFERENCES[w]['satisfaction_weights'][s] * SATISFACTION_SCALE)) * x[(w, d, s)]
                for d in range(NUM_DAYS) for s in SHIFT_CODES)
    penalita_scalata = int(round(10.0 * SATISFACTION_SCALE))
    for d in UNAVAILABLE.get(w, []):
        for s in SHIFT_CODES:
            sat_w -= penalita_scalata * x[(w, d, s)]
    model.Add(sat_w >= NEW_FLOOR_SCALED)
```

Aggiungi SEMPRE il Warm-Starting (Hinting) per abbattere i tempi di risoluzione, iniettando la schedulazione precedente subito prima del solver.Solve(model):
```python
for w in WORKER_IDS:
    for d in range(NUM_DAYS):
        for s in SHIFT_CODES:
            if PREVIOUS_SCHEDULE.get(w, {{}}).get(d) == s:
                model.AddHint(x[(w, d, s)], 1)
            else:
                model.AddHint(x[(w, d, s)], 0)
```
Mantieni la funzione obiettivo che massimizza la soddisfazione totale (puoi
lasciarla invariata). Se imporre questo pavimento rende il modello INFEASIBLE,
NON rilassare i vincoli hard: lascia semplicemente che il solver restituisca
INFEASIBLE (significa che l'equita' non e' ulteriormente migliorabile).

### COSA DEVE PRODURRE IL CODICE (invariato rispetto alla bozza)
- Risolvi con cp_model.CpSolver() (max_time_in_seconds = MAX_TIME,
  num_search_workers = 8, log_search_progress = False).
- Popola nel namespace:
    RESULT_SCHEDULE : dict {{wid: {{day_index: codice_turno_o_None}}}}
    SOLVER_STATUS   : str con il nome dello status (solver.StatusName(status))

NON stampare nulla, NON leggere/scrivere file, NON ridefinire le variabili gia'
disponibili. Restituisci SOLO un blocco di codice Python valido (tra ```python e ```).
"""


# ===========================================================================
# 2. ESITO DI UN'ITERAZIONE DEL CICLO
# ===========================================================================
@dataclass
class RefinementStep:
    """Traccia diagnostica di una singola iterazione del ciclo di raffinamento."""

    iteration: int
    status: str            # 'ACCEPTED' | 'REJECTED_HARD' | 'NO_IMPROVEMENT' | 'INFEASIBLE' | 'LLM_FAILED'
    solver_status: str
    worst_before: float
    worst_after: Optional[float] = None
    detail: str = ""


@dataclass
class RefinementOutcome:
    """Risultato complessivo della Fase 4 per uno use case."""

    case_label: str
    iterations_run: int
    initial_worst: float
    final_worst: float
    improved: bool
    best_result: ScheduleResult
    best_report: VerificationReport
    steps: List[RefinementStep] = field(default_factory=list)


# ===========================================================================
# 3. UTILITY
# ===========================================================================
def _is_infeasible(result: ScheduleResult) -> bool:
    """Vero se il solver ha dichiarato il modello INFEASIBLE o non ha prodotto turni."""
    status = (result.status_name or "").upper()
    return (not result.feasible) or ("INFEASIBLE" in status) or ("UNKNOWN" in status and not result.feasible)


def _build_refinement_context(
    data: ProblemData, max_time: float, worst_worker_id: str, new_floor_scaled: int, previous_schedule: dict
) -> dict:
    """Namespace di esecuzione: contesto della bozza + variabili di equita' della Fase 4."""
    ctx = _build_llm_context(data, max_time)
    ctx["SATISFACTION_SCALE"] = SATISFACTION_SCALE
    ctx["WORST_WORKER_ID"] = worst_worker_id
    ctx["NEW_FLOOR_SCALED"] = new_floor_scaled
    ctx["PREVIOUS_SCHEDULE"] = previous_schedule
    return ctx


# ===========================================================================
# 4. CICLO ITERATIVO DI RAFFINAMENTO
# ===========================================================================
def run_refinement_loop(
    executor,
    data: ProblemData,
    initial_result: ScheduleResult,
    initial_report: VerificationReport,
    max_iterations: int = 3,
    max_time: float = 60.0,
    max_retries: int = 3,
) -> RefinementOutcome:
    """
    Esegue il ciclo di raffinamento mirato all'equita' partendo da una bozza
    gia' verificata come valida (initial_report.hard_ok == True).

    Ad ogni iterazione alza il pavimento di equita' di +1 step (scalato) e chiede
    all'LLM di modificare il codice corrente di conseguenza; accetta la nuova
    bozza solo se il Verification Agent la conferma valida E il minimo globale
    migliora. Termina al raggiungimento di max_iterations o su INFEASIBLE/fallimento.
    """
    if not initial_report.hard_ok or initial_report.fairness is None:
        raise ValueError(
            "Il raffinamento (Fase 4) richiede una bozza iniziale gia' VALIDA "
            "(Fase 3 con hard_ok=True)."
        )
    if not initial_result.generated_code:
        raise ValueError(
            "Manca il codice cp_model della bozza iniziale: la Fase 4 lo richiede "
            "come input testuale. Rigenera la Fase 2 o passa --from-draft con il "
            f".txt del codice (draft_code_case_{data.case_label}.txt)."
        )

    # Stato di riferimento: la migliore schedulazione valida finora.
    best_result = initial_result
    best_report = initial_report
    current_code = initial_result.generated_code
    current_min = initial_report.fairness.worst_satisfaction
    initial_worst = current_min

    steps: List[RefinementStep] = []

    print(f"\n{'#'*64}")
    print(f"# FASE 4 - REFINEMENT AGENT | Caso {data.case_label}")
    print(f"# Lavoratore piu' svantaggiato di partenza: "
          f"{initial_report.fairness.worst_worker_id} "
          f"({initial_report.fairness.worst_worker_name}) = {current_min}")
    print(f"# Iterazioni massime: {max_iterations}")
    print(f"{'#'*64}")

    for it in range(1, max_iterations + 1):
        worst_id = best_report.fairness.worst_worker_id
        worst_name = best_report.fairness.worst_worker_name
        # Nuovo pavimento: +1 step scalato sopra il minimo attuale -> forza il
        # minimo globale a salire. Se irraggiungibile, il solver dira' INFEASIBLE.
        new_floor_scaled = int(round(current_min * SATISFACTION_SCALE)) + 1

        print(f"\n{'-'*64}")
        print(f"[Iterazione {it}/{max_iterations}] Pavimento attuale = {current_min} "
              f"-> obiettivo minimo >= {new_floor_scaled / SATISFACTION_SCALE}")
        print(f"  Target lavoratore: {worst_id} ({worst_name})")
        print(f"{'-'*64}")

        prompt = build_refinement_prompt(
            data, current_code, worst_id, worst_name, current_min, new_floor_scaled
        )
        context_vars = _build_refinement_context(
            data, max_time, worst_id, new_floor_scaled, best_result.schedule
        )

        # --- STEP 2: esegui il nuovo codice (motore Fase 0, auto-correzione) ---
        successo, risultato = executor.run_with_retry(
            prompt, context_vars=context_vars, max_retries=max_retries
        )
        if not successo:
            print("[!] L'LLM non ha prodotto codice eseguibile per questa iterazione. "
                  "Fine del ciclo.")
            steps.append(RefinementStep(
                it, "LLM_FAILED", "N/A", current_min,
                detail="run_with_retry ha esaurito i tentativi."))
            break

        raw_schedule = risultato.get("RESULT_SCHEDULE")
        solver_status = str(risultato.get("SOLVER_STATUS", "UNKNOWN"))
        if not isinstance(raw_schedule, dict):
            print(f"[!] Codice valido ma 'RESULT_SCHEDULE' assente (status: "
                  f"{solver_status}). Tratto come ottimizzazione terminata.")
            steps.append(RefinementStep(
                it, "INFEASIBLE", solver_status, current_min,
                detail="RESULT_SCHEDULE non definito."))
            break

        candidate = build_schedule_result(
            data, raw_schedule, solver_status, source="llm-refined",
            generated_code=executor.last_code,
        )

        # --- TERMINAZIONE: INFEASIBLE/fallimento del solutore ---
        if _is_infeasible(candidate):
            print(f"[=] Il solutore ha restituito {solver_status}: non e' piu' "
                  f"possibile alzare il minimo di equita'. Ottimizzazione TERMINATA.")
            steps.append(RefinementStep(
                it, "INFEASIBLE", solver_status, current_min,
                detail="Solver INFEASIBLE forzando un'equita' superiore."))
            break

        # --- STEP 3: Verification Agent (Fase 3) sul nuovo risultato ---
        candidate_report = verify_schedule(data, candidate)

        if not candidate_report.hard_ok:
            # Vincoli hard violati -> SCARTA la bozza, mantieni il riferimento.
            print(f"[X] Bozza SCARTATA: il Verification Agent ha rilevato "
                  f"{len(candidate_report.violations)} violazioni hard.")
            steps.append(RefinementStep(
                it, "REJECTED_HARD", solver_status, current_min,
                detail=f"{len(candidate_report.violations)} violazioni hard."))
            continue

        new_min = candidate_report.fairness.worst_satisfaction

        # --- ACCETTAZIONE: il minimo globale e' migliorato senza danni agli altri ---
        if new_min > current_min:
            print(f"[OK] Bozza ACCETTATA: minimo globale {current_min} -> {new_min} "
                  f"(lavoratore peggiore: {candidate_report.fairness.worst_worker_id}).")
            steps.append(RefinementStep(
                it, "ACCEPTED", solver_status, current_min, worst_after=new_min,
                detail="Nuovo orario di riferimento."))
            best_result = candidate
            best_report = candidate_report
            current_code = candidate.generated_code or current_code
            current_min = new_min
        else:
            # Valido ma senza guadagno di equita': non aggiorna il riferimento.
            print(f"[~] Nessun miglioramento dell'equita' (minimo resta {current_min}, "
                  f"candidato {new_min}). Bozza non adottata.")
            steps.append(RefinementStep(
                it, "NO_IMPROVEMENT", solver_status, current_min, worst_after=new_min,
                detail="Minimo globale non migliorato."))
            # Non c'e' progresso possibile in questa direzione: si esce.
            break

    return RefinementOutcome(
        case_label=data.case_label,
        iterations_run=len(steps),
        initial_worst=initial_worst,
        final_worst=current_min,
        improved=current_min > initial_worst,
        best_result=best_result,
        best_report=best_report,
        steps=steps,
    )


# ===========================================================================
# 5. REPORT FINALE DELLA FASE 4
# ===========================================================================
def print_refinement_summary(data: ProblemData, outcome: RefinementOutcome) -> None:
    print(f"\n{'='*64}")
    print(f"FASE 4 - RAFFINAMENTO COMPLETATO | Caso {outcome.case_label}")
    print(f"{'='*64}")
    print(f"Iterazioni eseguite      : {outcome.iterations_run}")
    print(f"Minimo iniziale (equita'): {outcome.initial_worst}")
    print(f"Minimo finale  (equita') : {outcome.final_worst}")
    print(f"Miglioramento ottenuto   : {'SI' if outcome.improved else 'NO'}")

    print("\nStorico iterazioni:")
    for s in outcome.steps:
        delta = ""
        if s.worst_after is not None:
            delta = f" (minimo {s.worst_before} -> {s.worst_after})"
        print(f"  [{s.iteration}] {s.status:<15} status_solver={s.solver_status}"
              f"{delta} | {s.detail}")

    print("\n--- Orario di riferimento finale (verifica Fase 3) ---")
    print_report(data, outcome.best_report)


# ===========================================================================
# 6. ORCHESTRAZIONE: Fase 2 -> Fase 3 -> Fase 4 per uno use case
# ===========================================================================
def _load_draft_from_disk(data: ProblemData) -> ScheduleResult:
    """
    Ricostruisce la bozza iniziale da disco: schedulazione dal CSV della Fase 2 e
    codice cp_model dal .txt salvato. Evita di richiamare l'LLM per la sola bozza
    iniziale quando e' gia' disponibile su disco.
    """
    csv_path = f"schedule_case_{data.case_label}.csv"
    code_path = f"draft_code_case_{data.case_label}.txt"
    result = load_schedule_from_csv(data, csv_path)
    if os.path.exists(code_path):
        with open(code_path, encoding="utf-8") as f:
            result.generated_code = f.read()
    return result


def run_case(
    case_label: str,
    from_draft: bool = False,
    max_iterations: int = 3,
    max_time: float = 60.0,
) -> Optional[RefinementOutcome]:
    """
    Esegue l'intera pipeline a valle per uno use case:
      Fase 2 (bozza, o caricata da disco) -> Fase 3 (verifica) -> Fase 4 (raffina).
    """
    data = load_problem_data(case_label)

    # Inferenza via Google Gemini 2.5 Flash: richiede GEMINI_API_KEY.
    from llm_engine import AgentExecutor
    executor = AgentExecutor()

    # --- Bozza iniziale (Fase 2) ---
    if from_draft:
        print(f"[*] Carico la bozza iniziale da disco (Caso {case_label})...")
        initial_result = _load_draft_from_disk(data)
        if not initial_result.generated_code:
            raise SystemExit(
                f"[!] Manca 'draft_code_case_{case_label}.txt' (codice cp_model "
                f"della bozza). Rigenera la Fase 2 senza --from-draft."
            )
    else:
        print(f"[*] Genero la bozza iniziale con la Fase 2 (Caso {case_label})...")
        initial_result = run_llm_drafting(executor, data, max_time=max_time)
        if initial_result is None:
            raise SystemExit(
                f"[!] La Fase 2 non ha prodotto una bozza per il Caso {case_label}."
            )

    # --- Verifica iniziale (Fase 3) ---
    print(f"\n[*] Verifica della bozza iniziale (Fase 3)...")
    initial_report = verify_schedule(data, initial_result)
    print_report(data, initial_report)

    if not initial_report.hard_ok:
        print("\n[!] La bozza iniziale viola gia' i vincoli hard: il raffinamento "
              "(Fase 4) parte solo da un piano valido. Interrompo.")
        return None

    # --- Ciclo di raffinamento (Fase 4) ---
    outcome = run_refinement_loop(
        executor, data, initial_result, initial_report,
        max_iterations=max_iterations, max_time=max_time,
    )

    # --- Salvataggio dell'orario di riferimento finale ---
    final_csv = f"schedule_case_{case_label}_final.csv"
    export_csv(data, outcome.best_result, path=final_csv)
    if outcome.best_result.generated_code:
        save_generated_code(
            case_label, outcome.best_result.generated_code,
            path=f"final_code_case_{case_label}.txt",
        )

    print_refinement_summary(data, outcome)
    return outcome


def main():
    parser = argparse.ArgumentParser(
        description="SmartScheduler Fase 4 - Refinement Agent "
                    "(ciclo iterativo di equita')."
    )
    parser.add_argument(
        "--case", choices=["A", "B", "all"], default="all",
        help="Use case da raffinare (default: all).",
    )
    parser.add_argument(
        "--from-draft", action="store_true",
        help="Usa la bozza iniziale gia' salvata su disco (CSV + codice .txt) "
             "invece di rigenerarla con la Fase 2.",
    )
    parser.add_argument(
        "--max-iterations", type=int, default=3,
        help="Numero massimo di iterazioni del ciclo di raffinamento (default: 3).",
    )
    parser.add_argument(
        "--max-time", type=float, default=60.0,
        help="Tempo massimo del solver per iterazione in secondi (default: 60).",
    )
    args = parser.parse_args()

    casi = ["A", "B"] if args.case == "all" else [args.case]
    for case_label in casi:
        run_case(
            case_label,
            from_draft=args.from_draft,
            max_iterations=args.max_iterations,
            max_time=args.max_time,
        )


if __name__ == "__main__":
    main()

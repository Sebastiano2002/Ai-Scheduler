"""
Fase4_refinement_agent.py
===================
Fase 4 - Raffinamento della Schedulazione (Ciclo Iterativo).

Implementa il ciclo di ottimizzazione dell'equità:
  - *Feedback*: Reistruisce il Drafting Agent per migliorare il lavoratore meno soddisfatto.
  - *Vincolo*: Non peggiorare la soddisfazione minima attuale e rispettare i vincoli hard.
  - *Terminazione*: Si ferma quando non è possibile migliorare ulteriormente o si raggiunge `max_iterations`.

Flusso iterativo:
    1. Crea un prompt con il codice corrente per migliorare il lavoratore peggiore.
    2. Esegue il nuovo codice tramite AgentExecutor.
    3. Valida il risultato con il Verification Agent.
    4. Accetta la bozza se i vincoli sono rispettati e il minimo globale migliora; scarta altrimenti.

Esecuzione:
    python Fase4_refinement_agent.py --case A
    python Fase4_refinement_agent.py --case B 
    
"""

import argparse
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from . import input_data
from .Fase2_drafting_agent import (
    ProblemData,
    ScheduleResult,
    SATISFACTION_SCALE,
    _build_llm_context,
    build_schedule_result,
    compute_sat_max,
    export_csv,
    load_problem_data,
    run_llm_drafting,
    save_generated_code,
    worker_satisfaction_pct,
)
from .Fase3_verification_agent import (
    VerificationReport,
    load_schedule_from_csv,
    print_report,
    verify_schedule,
)

# Risoluzione della scala NORMALIZZATA usata dal leximin: la soddisfazione
# normalizzata norm(w)=sat(w)/sat_max(w) e' rappresentata come intero z ~
# Permette di scrivere il vincolo max-min in forma LINEARE intera mantenendo i coefficienti del modello CP contenuti.
NORM_SCALE = 100


# ===========================================================================
# 1. PROMPT DI RAFFINAMENTO (Drafting Agent reistruito)
# ===========================================================================
def build_refinement_prompt(
    data: ProblemData,
    reference_code: str,
    locked_floors: dict,
    free_workers: List[str],
) -> str:
    """
    Costruisce il prompt che chiede all'LLM (Drafting Agent reistruito) di scrivere
    UNA VOLTA un modello cp_model **leximin parametrico**: l'obiettivoè la
    MASSIMIZZAZIONE DEL MINIMO.

    Il modello generato e' parametrico: legge dal namespace `LOCKED_FLOORS`,
    `FREE_WORKERS` e `CURRENT_ASSIGN`, cosi' che il ciclo di raffinamento
    (run_refinement_loop) possa RI-ESEGUIRE lo STESSO codice a ogni livello
    leximin aggiornando solo quelle variabili, senza una nuova chiamata LLM.

    - `LOCKED_FLOORS[w]` : per i lavoratori gia' fissati ai livelli leximin precedenti (la loro soddisfazione non puo' piu' scendere).
    - `FREE_WORKERS`: lavoratori ancora da ottimizzare; il modello ne massimizza il minimo.
    - `CURRENT_ASSIGN`   : la schedule corrente: senza questo hint il modello esatto ==25 e' lentissimo
      a trovare il primo punto fattibile (puo' dare UNKNOWN).
    """
    # Riepilogo compatto delle preferenze (informativo per l'LLM).
    pref_lines = []
    for wid in data.worker_ids:
        p = data.preferences.get(wid, {})
        graditi = ", ".join(p.get("turni_preferiti", [])) or "nessuno"
        sgraditi = ", ".join(p.get("turni_indesiderati", [])) or "nessuno"
        ferie = ", ".join(p.get("giorni_indesiderati", [])) or "nessuna"
        flex = p.get("flexibility_score", 0.5)
        pref_lines.append(
            f"  {wid} ({data.worker_names[wid]}): "
            f"preferiti={graditi}, indesiderati={sgraditi}, "
            f"ferie_richieste={ferie}, flessibilita'={flex}"
        )
    pref_summary = "\n".join(pref_lines)

    return f"""Sei il "Drafting Agent" di SmartScheduler nella FASE 4 (raffinamento iterativo
dell'equita' secondo il criterio MAX-MIN FAIRNESS / leximin). Hai gia' prodotto
una bozza VALIDA per lo USE CASE {data.case_label} (codice di riferimento sotto,
rispetta tutti i vincoli hard). Ora devi scrivere un modello cp_model che, invece
di massimizzare la soddisfazione TOTALE, **massimizza il MINIMO della soddisfazione
NORMALIZZATA** tra i lavoratori. La soddisfazione normalizzata di w e'
norm(w)=sat(w)/sat_max(w): quanto w e' vicino al PROPRIO massimo individuale. Si
normalizza perche' i punteggi assoluti NON sono confrontabili tra lavoratori (chi
predilige la Notte ha un massimo di poche unita', chi predilige Mattina/Pomeriggio
un massimo molto piu' alto): massimizzare il minimo ASSOLUTO premierebbe la
grandezza dei pesi, non l'equita' reale. Massimizzare il minimo NORMALIZZATO
solleva chi e' davvero piu' lontano dal proprio ottimo.

### CODICE cp_model DI RIFERIMENTO (per i VINCOLI HARD: replicali identici)
```python
{reference_code}
```

### PREFERENZE DI TUTTI I LAVORATORI (informativo)
{pref_summary}

### VARIABILI PARAMETRICHE GIA' DISPONIBILI NEL NAMESPACE (NON ridefinirle, NON
### inserire valori letterali al loro posto: il ciclo le aggiorna a ogni livello)
- SATISFACTION_SCALE : int  -> {SATISFACTION_SCALE}
- UNDESIRED_DAYS     : dict -> {{wid: set(indici_giorno) di ferie}}
- UNDESIRED_DAY_PENALTY : float -> penalita' per turno in un giorno indesiderato
- SAT_MAX_SCALED : dict -> {{wid: massimo individuale di soddisfazione SCALATO (intero >=1)}};
  e' il denominatore della normalizzazione (sat_max(w) del lavoratore w).
- NORM_SCALE : int -> risoluzione della scala normalizzata ({NORM_SCALE} = 100%).
- LOCKED_FLOORS : dict -> {{wid: pavimento_normalizzato_intero (in unita' di NORM_SCALE)}}
  dei lavoratori GIA' FISSATI ai livelli leximin precedenti (la loro soddisfazione
  normalizzata NON deve scendere sotto il pavimento).
- FREE_WORKERS  : list -> lavoratori ancora da ottimizzare (di cui massimizzare il minimo NORMALIZZATO).
- CURRENT_ASSIGN: dict -> {{(wid, giorno, turno): 0/1}} schedule corrente (warm-start).

### STRUTTURA OBBLIGATORIA DEL MODELLO leximin
```python
def build_model():
    model = cp_model.CpModel()
    x = {{(w, d, s): model.NewBoolVar(f"x_{{w}}_{{d}}_{{s}}")
          for w in WORKER_IDS for d in range(NUM_DAYS) for s in SHIFT_CODES}}
    # ... REPLICA QUI, IDENTICI, TUTTI i vincoli hard del codice di riferimento ...
    return model, x

model, x = build_model()

# WARM-START dalla schedule corrente: indispensabile, senza l'hint il modello
# esatto ==25 puo' restare UNKNOWN entro il tempo limite.
for key, var in x.items():
    if key in CURRENT_ASSIGN:
        model.AddHint(var, CURRENT_ASSIGN[key])

# Soddisfazione scalata per lavoratore (STESSO modello di soddisfazione: pesi
# turno + penalita' per i giorni indesiderati).
sat = {{}}
for w in WORKER_IDS:
    terms = []
    for d in range(NUM_DAYS):
        for s in SHIFT_CODES:
            peso = int(round(PREFERENCES[w]['satisfaction_weights'][s] * SATISFACTION_SCALE))
            if d in UNDESIRED_DAYS[w]:
                peso -= int(round(UNDESIRED_DAY_PENALTY * SATISFACTION_SCALE))
            terms.append(peso * x[(w, d, s)])
    sat[w] = cp_model.LinearExpr.sum(terms)

# Pavimenti dei lavoratori gia' fissati (leximin), in scala NORMALIZZATA:
# norm(w) >= floor/NORM_SCALE  <=>  sat[w]*NORM_SCALE >= floor*SAT_MAX_SCALED[w].
for w, floor in LOCKED_FLOORS.items():
    model.Add(sat[w] * NORM_SCALE >= floor * SAT_MAX_SCALED[w])

# MAX-MIN NORMALIZZATO: massimizza il minimo della soddisfazione normalizzata tra
# i lavoratori LIBERI. z e' in unita' di NORM_SCALE (z=1000 -> 100%). Il vincolo
# z <= norm(w) = sat[w]/SAT_MAX_SCALED[w] si linearizza moltiplicando in croce
# (SAT_MAX_SCALED[w] e' una costante positiva):
z = model.NewIntVar(-100000, 100000, 'z_min_norm')
for w in FREE_WORKERS:
    model.Add(z * SAT_MAX_SCALED[w] <= sat[w] * NORM_SCALE)
# Tie-breaker (Bounded Leximin): massimizza in subordine la soddisfazione totale.
# Impostando BIG=15 il sistema e' disposto a sacrificare al massimo 15 punti di 
# soddisfazione globale per regalare 1 punto percentuale (1 unita' di z) al 
# lavoratore piu' svantaggiato. Evita i "bagni di sangue" del Leximin puro.
BIG = 15
model.Maximize(z * BIG + cp_model.LinearExpr.sum([sat[w] for w in WORKER_IDS]))

solver = cp_model.CpSolver()
solver.parameters.max_time_in_seconds = MAX_TIME
solver.parameters.num_search_workers = 8
solver.parameters.log_search_progress = False
status = solver.Solve(model)

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

### REGOLE INDEROGABILI
- REPLICA IDENTICI tutti i vincoli hard del codice di riferimento: max 1 turno/giorno,
  25 turni mensili pesati (==25), max 36h/settimana e >=1 riposo settimanale,
  divieto Notte(d)->Mattina(d+1), 2 riposi dopo la Notte, turni vietati
  (FORBIDDEN_SHIFTS) e staffing del caso. NON rimuovere ne' indebolire nulla.
  (I giorni di ferie NON sono hard: restano nella penalita' della soddisfazione.)
- USA le variabili parametriche dal namespace (LOCKED_FLOORS, FREE_WORKERS,
  CURRENT_ASSIGN): NON sostituirle con valori letterali, il ciclo le riaggiorna.
- Se imporre i pavimenti rende il modello INFEASIBLE, NON rilassare i vincoli hard:
  lascia che il solver restituisca INFEASIBLE.
- NON stampare nulla, NON leggere/scrivere file.

Popola nel namespace SOLO `RESULT_SCHEDULE` e `SOLVER_STATUS`. Restituisci SOLO
un blocco di codice Python valido (tra ```python e ```).
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
    # Vettori di soddisfazione per lavoratore PRIMA e DOPO il leximin: servono a
    # mostrare il guadagno reale.
    initial_satisfaction: Dict[str, float] = field(default_factory=dict)
    final_satisfaction: Dict[str, float] = field(default_factory=dict)
    # Stessi vettori in versione NORMALIZZATA: sono la vista corretta per leggere il guadagno di equita'.
    initial_worst_pct: float = 0.0
    final_worst_pct: float = 0.0
    initial_satisfaction_pct: Dict[str, float] = field(default_factory=dict)
    final_satisfaction_pct: Dict[str, float] = field(default_factory=dict)


# ===========================================================================
# 3. UTILITY
# ===========================================================================
def _is_infeasible(result: ScheduleResult) -> bool:
    """Vero se il solver ha dichiarato il modello INFEASIBLE o non ha prodotto turni."""
    status = (result.status_name or "").upper()
    return (not result.feasible) or ("INFEASIBLE" in status) or ("UNKNOWN" in status and not result.feasible)


def _build_refinement_context(
    data: ProblemData,
    max_time: float,
    locked_floors: dict,
    free_workers: List[str],
    current_assign: dict,
) -> dict:
    """
    Namespace di esecuzione del template leximin: contesto (vincoli,
    pesi, ferie) + le variabili PARAMETRICHE che il ciclo aggiorna a ogni livello
    (pavimenti dei lavoratori fissati, lavoratori liberi, warm-start corrente).
    """
    ctx = _build_llm_context(data, max_time)
    ctx["SATISFACTION_SCALE"] = SATISFACTION_SCALE
    ctx["LOCKED_FLOORS"] = dict(locked_floors)
    ctx["FREE_WORKERS"] = list(free_workers)
    ctx["CURRENT_ASSIGN"] = current_assign
    # Scala normalizzata: massimo individuale scalato a intero (>=1).
    ctx["SAT_MAX_SCALED"] = _sat_max_scaled(data)
    ctx["NORM_SCALE"] = NORM_SCALE
    return ctx


def _sat_max_scaled(data: ProblemData) -> Dict[str, int]:
    """Massimo individuale di soddisfazione"""
    sat_max = compute_sat_max(data)
    return {
        w: max(int(round(sat_max.get(w, 0.0) * SATISFACTION_SCALE)), 1)
        for w in data.worker_ids
    }


def _assign_from_result(data: ProblemData, result: ScheduleResult) -> dict:
    """
    Traduce una schedule in hint per il warm-start, indispensabile per non ricadere in UNKNOWN sul modello ==25.
    """
    assign = {}
    for w in data.worker_ids:
        for d in range(input_data.NUM_DAYS):
            code = result.schedule[w].get(d)
            for s in input_data.SHIFT_CODES:
                assign[(w, d, s)] = 1 if code == s else 0
    return assign


def _scaled_satisfaction(result: ScheduleResult, wid: str) -> int:
    """Soddisfazione del lavoratore nella schedule, scalata a intero (come nel CP)."""
    return int(round(result.satisfaction_per_worker[wid] * SATISFACTION_SCALE))


# ===========================================================================
# 4. CICLO ITERATIVO DI RAFFINAMENTO
# ===========================================================================
def run_refinement_loop(
    executor,
    data: ProblemData,
    initial_result: ScheduleResult,
    initial_report: VerificationReport,
    max_iterations: int = 25,
    max_time: float = 60.0,
    max_retries: int = 3,
) -> RefinementOutcome:
    """
    Esegue il ciclo di raffinamento LEXIMIN partendo da una bozza valida.

    Flusso:
    - L'LLM genera UNA VOLTA il modello parametrico che massimizza il minimo per i liberi.
    - Il ciclo determina il minimo, fissa i lavoratori vincolanti a quel livello, 
      aggiorna il warm-start e riesegue il solver (senza chiamate LLM).
    
    Perché leximin: massimizzando il minimo si solleva direttamente la fascia bassa, 
    mentre la somma lascia indietro i peggiori.

    Terminazione: Tutti fissati, solver INFEASIBLE/UNKNOWN, o nessun miglioramento. 
    `max_iterations` è solo una salvaguardia.
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
            f".txt del codice (Output/draft_code_case_{data.case_label}.txt)."
        )

    worker_ids = list(data.worker_ids)
    initial_sat = dict(initial_result.satisfaction_per_worker)
    initial_worst = initial_report.fairness.worst_satisfaction
    # Il leximin ottimizza il MINIMO NORMALIZZATO.
    sat_max_scaled = _sat_max_scaled(data)
    initial_worst_pct = initial_report.fairness.worst_pct
    initial_pct = dict(initial_report.fairness.satisfaction_pct_per_worker)

    # Stato di riferimento (migliore schedule valida finora) e stato del leximin.
    best_result = initial_result
    best_report = initial_report
    reference_code = initial_result.generated_code   # solo per i vincoli hard
    template_code = None                             
    locked_floors: Dict[str, int] = {}               
    free = list(worker_ids)                          # lavoratori ancora da ottimizzare
    current_assign = _assign_from_result(data, initial_result)  
    last_floor_scaled = None                         # pavimento del livello precedente
    current_max_time = max_time                      
    last_objective_score = None                      # tracking stallo (z_star, total_sum)

    steps: List[RefinementStep] = []

    print(f"\n{'#'*64}")
    print(f"# FASE 4 - REFINEMENT AGENT (LEXIMIN) | Caso {data.case_label}")
    print(f"# Minimo di partenza: {initial_report.fairness.worst_worker_id} "
          f"({initial_report.fairness.worst_worker_name}) = {initial_worst}")
    print(f"# Iterazioni massime (salvaguardia): {max_iterations}")
    print(f"{'#'*64}")

    for it in range(1, max_iterations + 1):
        if not free:
            print("[=] Tutti i lavoratori sono fissati: leximin COMPLETO.")
            break

        context_vars = _build_refinement_context(
            data, current_max_time, locked_floors, free, current_assign
        )

        print(f"\n{'-'*64}")
        print(f"[Livello leximin {it}/{max_iterations}] liberi={len(free)} "
              f"fissati={len(locked_floors)}")
        print(f"{'-'*64}")

        # STEP 1+2: ottieni ed esegui il template leximin
        # Prima volta -> genera con l'LLM; volte successive -> ri-esegui lo stesso
        # codice aggiornando solo il namespace.
        if template_code is None:
            prompt = build_refinement_prompt(data, reference_code, locked_floors, free)
            successo, risultato = executor.run_with_retry(
                prompt, context_vars=context_vars, max_retries=max_retries
            )
            if successo:
                template_code = executor.last_code
        else:
            successo, risultato = executor.safe_execute(template_code, context_vars)
            if not successo:
                # Fallback: il template ha fallito su questo livello -> ri-prompt.
                print("[!] Ri-esecuzione del template fallita, ri-prompt del livello...")
                prompt = build_refinement_prompt(
                    data, reference_code, locked_floors, free
                )
                successo, risultato = executor.run_with_retry(
                    prompt, context_vars=context_vars, max_retries=max_retries
                )
                if successo:
                    template_code = executor.last_code

        if not successo:
            print("[!] L'LLM non ha prodotto codice eseguibile. Fine del ciclo.")
            steps.append(RefinementStep(
                it, "LLM_FAILED", "N/A", initial_worst,
                detail="run_with_retry ha esaurito i tentativi."))
            break

        raw_schedule = risultato.get("RESULT_SCHEDULE")
        solver_status = str(risultato.get("SOLVER_STATUS", "UNKNOWN"))
        if not isinstance(raw_schedule, dict):
            steps.append(RefinementStep(
                it, "INFEASIBLE", solver_status, initial_worst,
                detail="RESULT_SCHEDULE non definito."))
            break

        candidate = build_schedule_result(
            data, raw_schedule, solver_status, source="llm-refined",
            generated_code=template_code,
        )

        if _is_infeasible(candidate):
            print(f"[=] Il solutore ha restituito {solver_status}: leximin TERMINATO.")
            steps.append(RefinementStep(
                it, "INFEASIBLE", solver_status, initial_worst,
                detail="Solver INFEASIBLE/UNKNOWN su questo livello."))
            break

        # STEP 3: Verification Agent (Fase 3)
        candidate_report = verify_schedule(data, candidate)
        if not candidate_report.hard_ok:
            print(f"[X] Livello SCARTATO: {len(candidate_report.violations)} "
                  f"violazioni hard.")
            steps.append(RefinementStep(
                it, "REJECTED_HARD", solver_status, initial_worst,
                detail=f"{len(candidate_report.violations)} violazioni hard."))
            break

        # --- BOOKKEEPING LEXIMIN (NORMALIZZATO): minimo % tra i liberi + fissaggio ---
        # norm_scaled[w] = soddisfazione normalizzata in unita' di NORM_SCALE.
        # Si usa la divisione INTERA (floor) cosi' il pavimento norm_scaled[w] e'
        # sempre soddisfatto dal candidato stesso (evita INFEASIBLE da arrotondamento).
        sat_scaled = {w: _scaled_satisfaction(candidate, w) for w in worker_ids}
        norm_scaled = {
            w: (sat_scaled[w] * NORM_SCALE) // sat_max_scaled[w] for w in worker_ids
        }
        z_star = min(norm_scaled[w] for w in free)   # minimo normalizzato (permille)
        total_sum = sum(sat_scaled.values())
        current_score = (z_star, total_sum)

        # Il nuovo candidato e' matematicamente migliore o uguale al warm-start,
        # quindi aggiorniamo sempre il riferimento corrente.
        best_result = candidate
        best_report = candidate_report
        current_assign = _assign_from_result(data, candidate)

        is_optimal = ("OPTIMAL" in solver_status)
        is_stuck = ("FEASIBLE" in solver_status and last_objective_score is not None and current_score == last_objective_score)

        z_star_pct = round(z_star / (NORM_SCALE / 100), 1)   # permille -> percento

        if is_stuck:
            if current_max_time <= max_time:
                # Tentativo approfondito
                current_max_time = max_time * 2
                print(f"[!] Nessun progresso oggettivo rilevato. Raddoppio il tempo a {current_max_time}s per un 'Deep Dive'.")
                steps.append(RefinementStep(
                    it, "STUCK_DOUBLING_TIME", solver_status,
                    worst_before=initial_worst_pct, worst_after=z_star_pct,
                    detail=f"Nessun progresso. Deep dive {current_max_time}s nel prossimo ciclo."))
                last_objective_score = current_score
                continue
            else:
                # Impantanato anche col deep dive
                print(f"[X] Impantanato anche dopo il Deep Dive ({current_max_time}s). Forzo il blocco del livello come se fosse OPTIMAL.")
                is_optimal = True
                steps.append(RefinementStep(
                    it, "FORCED_OPTIMAL", solver_status,
                    worst_before=initial_worst_pct, worst_after=z_star_pct,
                    detail=f"Resa (Early Stop) dopo Deep Dive {current_max_time}s."))

        # Se ci siamo sbloccati o abbiamo migliorato senza esserci arenati prima
        if current_score != last_objective_score and not is_optimal:
            current_max_time = max_time 
            last_objective_score = current_score

        # Blocchiamo il livello SOLO se il solutore ha certificato l'ottimalita' (o forzato)
        if is_optimal:
            # Niente progresso: il minimo NORMALIZZATO dei liberi non supera il pavimento.
            if last_floor_scaled is not None and z_star <= last_floor_scaled:
                print(f"[~] Il minimo normalizzato dei liberi ({z_star_pct}%) non "
                      f"supera il pavimento precedente (OPTIMAL o Forzato): nessun ulteriore miglioramento.")
                steps.append(RefinementStep(
                    it, "NO_IMPROVEMENT", solver_status,
                    round(last_floor_scaled / (NORM_SCALE / 100), 1),
                    worst_after=z_star_pct,
                    detail="Minimo normalizzato dei liberi non migliorato."))
                break

            binding = sorted(w for w in free if norm_scaled[w] == z_star)
            for w in binding:
                locked_floors[w] = z_star
                free.remove(w)

            last_floor_scaled = z_star
            last_objective_score = None 
            current_max_time = max_time

            print(f"[OK] Livello bloccato. Minimo normalizzato dei liberi confermato a {z_star_pct}%; "
                  f"fissati a questo livello: {', '.join(binding)}.")
            if "OPTIMAL" in solver_status:
                steps.append(RefinementStep(
                    it, "ACCEPTED_OPTIMAL", solver_status,
                    worst_before=initial_worst_pct, worst_after=z_star_pct,
                    detail=f"Livello bloccato: {', '.join(binding)} a {z_star_pct}%."))
        else:
            # Soluzione FEASIBLE: migliorata ma non possiamo garantire che sia il tetto massimo.
            # NON blocchiamo e procediamo all'iterazione successiva con il nuovo warm-start.
            print(f"[*] Avanzamento parziale (FEASIBLE). Minimo normalizzato provvisorio a {z_star_pct}%. "
                  f"Nessun lavoratore bloccato, proseguo la ricerca...")
            steps.append(RefinementStep(
                it, "ACCEPTED_FEASIBLE", solver_status,
                worst_before=initial_worst_pct, worst_after=z_star_pct,
                detail=f"Avanzamento parziale, minimo a {z_star_pct}%."))

    final_sat = dict(best_result.satisfaction_per_worker)
    final_pct = dict(best_report.fairness.satisfaction_pct_per_worker)
    # "improved" in senso leximin sulla scala NORMALIZZATA: il vettore delle % di
    # soddisfazione ordinato in modo crescente e' lessicograficamente migliore (il
    # minimo normalizzato sale, o a parita' di minimo sale il secondo peggiore).
    improved = sorted(final_pct.values()) > sorted(initial_pct.values())

    return RefinementOutcome(
        case_label=data.case_label,
        iterations_run=len(steps),
        initial_worst=initial_worst,
        final_worst=best_report.fairness.worst_satisfaction,
        improved=improved,
        best_result=best_result,
        best_report=best_report,
        steps=steps,
        initial_satisfaction=initial_sat,
        final_satisfaction=final_sat,
        initial_worst_pct=initial_worst_pct,
        final_worst_pct=best_report.fairness.worst_pct,
        initial_satisfaction_pct=initial_pct,
        final_satisfaction_pct=final_pct,
    )


# ===========================================================================
# 5. REPORT FINALE DELLA FASE 4
# ===========================================================================
def print_refinement_summary(data: ProblemData, outcome: RefinementOutcome) -> None:
    print(f"\n{'='*64}")
    print(f"FASE 4 - RAFFINAMENTO COMPLETATO | Caso {outcome.case_label}")
    print(f"{'='*64}")
    print(f"Iterazioni eseguite          : {outcome.iterations_run}")
    print(f"Minimo iniziale (assoluto)   : {outcome.initial_worst}")
    print(f"Minimo finale  (assoluto)    : {outcome.final_worst}")
    print(f"Minimo normalizzato iniziale : {outcome.initial_worst_pct}%")
    print(f"Minimo normalizzato finale   : {outcome.final_worst_pct}%  <-- criterio leximin")
    print(f"Miglioramento ottenuto       : {'SI' if outcome.improved else 'NO'}")

    print("\nStorico livelli leximin (pavimenti in % normalizzata):")
    for s in outcome.steps:
        delta = ""
        if s.worst_after is not None:
            delta = f" (pavimento -> {s.worst_after}%)"
        print(f"  [{s.iteration}] {s.status:<18} status_solver={s.solver_status}"
              f"{delta} | {s.detail}")

    # Confronto leximin PRIMA vs DOPO, su ENTRAMBE le scale: assoluta e normalizzata 
    ini = outcome.initial_satisfaction
    fin = outcome.final_satisfaction
    ini_p = outcome.initial_satisfaction_pct
    fin_p = outcome.final_satisfaction_pct
    if ini and fin:
        print("\n--- Soddisfazione PRIMA -> DOPO | ASSOLUTA  e  NORMALIZZATA (%) ---")
        print(f"  {'lavoratore':<24}{'assoluta':>18}{'normalizzata':>20}")
        for w in sorted(ini, key=lambda k: fin_p.get(k, 0.0)):
            mark = "  *" if fin_p.get(w, 0.0) > ini_p.get(w, 0.0) + 1e-9 else ""
            print(f"  {w} {data.worker_names.get(w, ''):<20} "
                  f"{ini.get(w, 0.0):>7.1f} -> {fin.get(w, 0.0):>7.1f}"
                  f"   {ini_p.get(w, 0.0):>6.1f}% -> {fin_p.get(w, 0.0):>6.1f}%{mark}")
        print(f"  {'Minimo':<24}{min(ini.values()):>7.1f} -> {min(fin.values()):>7.1f}"
              f"   {min(ini_p.values()):>6.1f}% -> {min(fin_p.values()):>6.1f}%")
        print(f"  {'Somma':<24}{sum(ini.values()):>7.1f} -> {sum(fin.values()):>7.1f}")

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
    csv_path = f"Output/schedule_case_{data.case_label}.csv"
    code_path = f"Output/draft_code_case_{data.case_label}.txt"
    result = load_schedule_from_csv(data, csv_path)
    if os.path.exists(code_path):
        with open(code_path, encoding="utf-8") as f:
            result.generated_code = f.read()
    return result


def run_case(
    case_label: str,
    from_draft: bool = False,
    max_iterations: int = 25,
    max_time: float = 60.0,
) -> Optional[RefinementOutcome]:
    """
    Esegue l'intera pipeline a valle per uno use case:
      Fase 2 (bozza, o caricata da disco) -> Fase 3 (verifica) -> Fase 4 (raffina).
    """
    data = load_problem_data(case_label)

    
    from .llm_engine import AgentExecutor
    executor = AgentExecutor()

    # Bozza iniziale (Fase 2) 
    if from_draft:
        print(f"[*] Carico la bozza iniziale da disco (Caso {case_label})...")
        initial_result = _load_draft_from_disk(data)
        if not initial_result.generated_code:
            raise SystemExit(
                f"[!] Manca 'Output/draft_code_case_{case_label}.txt' (codice cp_model "
                f"della bozza). Rigenera la Fase 2 senza --from-draft."
            )
    else:
        print(f"[*] Genero la bozza iniziale con la Fase 2 (Caso {case_label})...")
        initial_result = run_llm_drafting(executor, data, max_time=max_time)
        if initial_result is None:
            raise SystemExit(
                f"[!] La Fase 2 non ha prodotto una bozza per il Caso {case_label}."
            )

    # Verifica iniziale (Fase 3) 
    print(f"\n[*] Verifica della bozza iniziale (Fase 3)...")
    initial_report = verify_schedule(data, initial_result)
    print_report(data, initial_report)

    if not initial_report.hard_ok:
        print("\n[!] La bozza iniziale viola gia' i vincoli hard: il raffinamento "
              "(Fase 4) parte solo da un piano valido. Interrompo.")
        return None

    # Ciclo di raffinamento (Fase 4) 
    outcome = run_refinement_loop(
        executor, data, initial_result, initial_report,
        max_iterations=max_iterations, max_time=max_time,
    )

    # Salvataggio dell'orario di riferimento finale 
    final_csv = f"Output/schedule_case_{case_label}_final.csv"
    export_csv(data, outcome.best_result, path=final_csv)
    if outcome.best_result.generated_code:
        save_generated_code(
            case_label, outcome.best_result.generated_code,
            path=f"Output/final_code_case_{case_label}.txt",
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
        "--max-iterations", type=int, default=25,
        help="SALVAGUARDIA: tetto dei livelli leximin (default: 25, sopra il n. di "
             "lavoratori cosi' il leximin puo' completarsi). Solo il 1o livello e' "
             "una chiamata LLM; i successivi ri-eseguono lo stesso template (solo "
             "risoluzione CP). Il ciclo termina di norma quando tutti i lavoratori "
             "sono fissati o non si migliora piu'.",
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

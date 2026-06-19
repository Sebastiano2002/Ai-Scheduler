"""
Fase4_refinement_agent.py
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
  - *Terminazione*: il criterio PRIMARIO (quello del PDF) e' "finche' nessun
    ulteriore miglioramento dell'equita' e' possibile": il ciclo si ferma quando
    il solutore restituisce INFEASIBLE forzando un pavimento piu' alto, oppure
    quando il minimo globale non migliora piu'. Il limite `max_iterations` e'
    invece una SALVAGUARDIA secondaria (anti-loop e anti-costo: ogni iterazione
    e' una chiamata LLM), tarata alta perche' la terminazione avvenga di norma
    in modo naturale e non per taglio del tetto.

Flusso di un'iterazione:
    1. costruisce un prompt che INCLUDE il codice cp_model della bozza corrente
       e chiede di migliorare il lavoratore peggiore senza danneggiare gli altri;
    2. esegue il nuovo codice con AgentExecutor.run_with_retry (motore Fase 0);
    3. ricostruisce la ScheduleResult e la passa al Verification Agent (Fase 3);
    4. se i vincoli hard sono violati -> SCARTA la bozza;
       se sono rispettati e il nuovo minimo globale e' MIGLIORE -> ACCETTA la
       bozza come nuovo orario di riferimento.

Esecuzione (richiede la variabile d'ambiente GEMINI_API_KEY):
    python Fase4_refinement_agent.py --case A
    python Fase4_refinement_agent.py --case B --max-iterations 3
    # Riparte da una bozza gia' salvata (CSV + .txt del codice) senza ri-fase 2:
    python Fase4_refinement_agent.py --case A --from-draft
"""

import argparse
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import input_data
from Fase2_drafting_agent import (
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
from Fase3_verification_agent import (
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
    reference_code: str,
    locked_floors: dict,
    free_workers: List[str],
) -> str:
    """
    Costruisce il prompt che chiede all'LLM (Drafting Agent reistruito) di scrivere
    UNA VOLTA un modello cp_model **leximin parametrico**: l'obiettivo non e' piu'
    la somma utilitaristica (che lasciava il peggiore inchiodato), ma la
    MASSIMIZZAZIONE DEL MINIMO (Max-Min Fairness, Stage 4 del PDF).

    Il modello generato e' parametrico: legge dal namespace `LOCKED_FLOORS`,
    `FREE_WORKERS` e `CURRENT_ASSIGN`, cosi' che il ciclo di raffinamento
    (run_refinement_loop) possa RI-ESEGUIRE lo STESSO codice a ogni livello
    leximin aggiornando solo quelle variabili, senza una nuova chiamata LLM.

    - `LOCKED_FLOORS[w]` : pavimento HARD (scalato) per i lavoratori gia' fissati
      ai livelli leximin precedenti (la loro soddisfazione non puo' piu' scendere).
    - `FREE_WORKERS`     : lavoratori ancora da ottimizzare; il modello ne
      massimizza il minimo.
    - `CURRENT_ASSIGN`   : la schedule corrente (dict {(w,d,s): 0/1}) usata come
      warm-start (AddHint): senza questo hint il modello esatto ==25 e' lentissimo
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
di massimizzare la soddisfazione TOTALE, **massimizza il MINIMO** di soddisfazione
tra i lavoratori: e' questo che solleva il lavoratore piu' svantaggiato (la somma
utilitaristica, al contrario, lo lascia inchiodato per premiare i gia' soddisfatti).

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
- LOCKED_FLOORS : dict -> {{wid: pavimento_scalato_intero}} dei lavoratori GIA'
  FISSATI ai livelli leximin precedenti (la loro soddisfazione NON deve scendere).
- FREE_WORKERS  : list -> lavoratori ancora da ottimizzare (di cui massimizzare il minimo).
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
    sat[w] = sum(terms)

# Pavimenti HARD dei lavoratori gia' fissati ai livelli precedenti (leximin).
for w, floor in LOCKED_FLOORS.items():
    model.Add(sat[w] >= floor)

# MAX-MIN: massimizza il minimo di soddisfazione tra i lavoratori LIBERI.
z = model.NewIntVar(-100000, 100000, 'z_min')
for w in FREE_WORKERS:
    model.Add(z <= sat[w])
# Tie-breaker debole: a parita' di minimo, massimizza anche la somma (cosi' il
# margine in eccesso non viene sprecato). Il peso BIG rende l'ordine lessicografico
# (prima il minimo, poi la somma).
BIG = 100000
model.Maximize(z * BIG + sum(sat[w] for w in WORKER_IDS))

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
    # mostrare il guadagno reale (il leximin solleva l'intera fascia bassa, non
    # solo il minimo assoluto, che puo' essere gia' al suo tetto strutturale).
    initial_satisfaction: Dict[str, float] = field(default_factory=dict)
    final_satisfaction: Dict[str, float] = field(default_factory=dict)


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
    Namespace di esecuzione del template leximin: contesto della bozza (vincoli,
    pesi, ferie) + le variabili PARAMETRICHE che il ciclo aggiorna a ogni livello
    (pavimenti dei lavoratori fissati, lavoratori liberi, warm-start corrente).
    """
    ctx = _build_llm_context(data, max_time)
    ctx["SATISFACTION_SCALE"] = SATISFACTION_SCALE
    ctx["LOCKED_FLOORS"] = dict(locked_floors)
    ctx["FREE_WORKERS"] = list(free_workers)
    ctx["CURRENT_ASSIGN"] = current_assign
    return ctx


def _assign_from_result(data: ProblemData, result: ScheduleResult) -> dict:
    """
    Traduce una schedule in hint per il warm-start: {(wid, giorno, turno): 0/1}.
    E' la schedule da cui il livello leximin successivo riparte (carry del
    warm-start), indispensabile per non ricadere in UNKNOWN sul modello ==25.
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
    Esegue il ciclo di raffinamento LEXIMIN (Max-Min Fairness, Stage 4 del PDF)
    partendo da una bozza gia' verificata valida (initial_report.hard_ok == True).

    Architettura (Opzione 1, "template parametrico"):
    - l'LLM scrive UNA VOLTA il modello leximin parametrico (vedi
      build_refinement_prompt): massimizza il MINIMO di soddisfazione tra i
      lavoratori liberi, con i lavoratori gia' fissati tenuti sopra il loro
      pavimento hard, e warm-start dalla schedule corrente;
    - il ciclo (questa funzione) e' il vero "agente di raffinamento": a ogni
      LIVELLO leximin determina il minimo raggiunto, FISSA i lavoratori vincolanti
      a quel pavimento, aggiorna il warm-start e RI-ESEGUE lo stesso template
      (senza nuova chiamata LLM) per sollevare il livello successivo.

    Perche' leximin e non "somma + pavimento +0.1": massimizzando la somma il
    solver lascia il peggiore inchiodato (premia chi e' gia' soddisfatto); il
    max-min lo solleva direttamente, e fissando i vincolanti si risale tutta la
    fascia bassa (non solo il minimo assoluto, che puo' essere al suo tetto
    strutturale -> in quel caso il livello successivo migliora comunque gli altri).

    Terminazione NATURALE: tutti i lavoratori fissati (leximin completo), oppure
    solver INFEASIBLE/UNKNOWN, oppure il minimo dei liberi non supera il pavimento
    gia' bloccato. `max_iterations` resta solo SALVAGUARDIA (anti-loop/costo LLM).
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

    worker_ids = list(data.worker_ids)
    initial_sat = dict(initial_result.satisfaction_per_worker)
    initial_worst = initial_report.fairness.worst_satisfaction

    # Stato di riferimento (migliore schedule valida finora) e stato del leximin.
    best_result = initial_result
    best_report = initial_report
    reference_code = initial_result.generated_code   # solo per i vincoli hard
    template_code = None                             # codice leximin (1 sola gen LLM)
    locked_floors: Dict[str, int] = {}               # wid -> pavimento scalato
    free = list(worker_ids)                          # lavoratori ancora da ottimizzare
    current_assign = _assign_from_result(data, initial_result)  # warm-start corrente
    last_floor_scaled = None                         # pavimento del livello precedente

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
            data, max_time, locked_floors, free, current_assign
        )

        print(f"\n{'-'*64}")
        print(f"[Livello leximin {it}/{max_iterations}] liberi={len(free)} "
              f"fissati={len(locked_floors)}")
        print(f"{'-'*64}")

        # --- STEP 1+2: ottieni ed esegui il template leximin ---
        # Prima volta -> genera con l'LLM; volte successive -> ri-esegui lo stesso
        # codice aggiornando solo il namespace (nessuna nuova chiamata LLM).
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

        # --- STEP 3: Verification Agent (Fase 3) ---
        candidate_report = verify_schedule(data, candidate)
        if not candidate_report.hard_ok:
            print(f"[X] Livello SCARTATO: {len(candidate_report.violations)} "
                  f"violazioni hard.")
            steps.append(RefinementStep(
                it, "REJECTED_HARD", solver_status, initial_worst,
                detail=f"{len(candidate_report.violations)} violazioni hard."))
            break

        # --- BOOKKEEPING LEXIMIN: minimo raggiunto tra i liberi + fissaggio ---
        sat_scaled = {w: _scaled_satisfaction(candidate, w) for w in worker_ids}
        z_star = min(sat_scaled[w] for w in free)

        # Niente progresso: il minimo dei liberi non supera l'ultimo pavimento.
        if last_floor_scaled is not None and z_star <= last_floor_scaled:
            print(f"[~] Il minimo dei liberi ({z_star / SATISFACTION_SCALE}) non "
                  f"supera il pavimento precedente: nessun ulteriore miglioramento.")
            steps.append(RefinementStep(
                it, "NO_IMPROVEMENT", solver_status,
                last_floor_scaled / SATISFACTION_SCALE,
                worst_after=z_star / SATISFACTION_SCALE,
                detail="Minimo dei liberi non migliorato."))
            break

        binding = sorted(w for w in free if sat_scaled[w] == z_star)

        # Il leximin e' monotono: questa schedule domina la precedente (i fissati
        # restano sopra il loro pavimento, i liberi sono saliti) -> nuovo riferimento.
        best_result = candidate
        best_report = candidate_report
        current_assign = _assign_from_result(data, candidate)
        for w in binding:
            locked_floors[w] = z_star
            free.remove(w)
        last_floor_scaled = z_star

        print(f"[OK] Minimo dei liberi sollevato a {z_star / SATISFACTION_SCALE}; "
              f"fissati a questo livello: {', '.join(binding)}.")
        steps.append(RefinementStep(
            it, "ACCEPTED", solver_status,
            worst_before=initial_worst, worst_after=z_star / SATISFACTION_SCALE,
            detail=f"Livello leximin: bloccati {', '.join(binding)} "
                   f"a {z_star / SATISFACTION_SCALE}."))

    final_sat = dict(best_result.satisfaction_per_worker)
    # "improved" in senso leximin: il vettore ordinato in modo crescente e'
    # lessicograficamente migliore (il minimo sale, o a parita' di minimo sale il
    # secondo peggiore, ecc.). Cattura anche i casi in cui il minimo assoluto e'
    # gia' al tetto ma la fascia bassa migliora.
    improved = sorted(final_sat.values()) > sorted(initial_sat.values())

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

    print("\nStorico livelli leximin:")
    for s in outcome.steps:
        delta = ""
        if s.worst_after is not None:
            delta = f" (pavimento -> {s.worst_after})"
        print(f"  [{s.iteration}] {s.status:<15} status_solver={s.solver_status}"
              f"{delta} | {s.detail}")

    # Confronto leximin: vettore di soddisfazione (crescente) PRIMA vs DOPO. E' la
    # vista che mostra il vero effetto del raffinamento: la fascia bassa si alza
    # anche quando il minimo assoluto e' gia' al suo tetto strutturale.
    if outcome.initial_satisfaction and outcome.final_satisfaction:
        ini = outcome.initial_satisfaction
        fin = outcome.final_satisfaction
        print("\n--- Soddisfazione per lavoratore: PRIMA -> DOPO (leximin) ---")
        for w in sorted(ini, key=lambda k: fin.get(k, 0.0)):
            mark = "  *" if fin.get(w, 0.0) > ini.get(w, 0.0) + 1e-9 else ""
            print(f"  {w} {data.worker_names.get(w, ''):<22} "
                  f"{ini.get(w, 0.0):>8.1f} -> {fin.get(w, 0.0):>8.1f}{mark}")
        print(f"  Minimo : {min(ini.values()):.1f} -> {min(fin.values()):.1f}")
        print(f"  Somma  : {sum(ini.values()):.1f} -> {sum(fin.values()):.1f}")

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
    max_iterations: int = 25,
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

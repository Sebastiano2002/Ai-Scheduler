"""
refinement_agent.py
===================
Fase 4 - Refinement Agent.
Raffina iterativamente l'equità concentrandosi sul lavoratore più svantaggiato,
come identificato dal Verification Agent (Fase 3).
"""

import os
import json
import re
import time
import math
from typing import Dict, Optional
import copy

from ortools.sat.python import cp_model
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

import input_data
from drafting_agent import ProblemData, ScheduleResult, SATISFACTION_SCALE

load_dotenv()

MAX_ITER = 5
SOLVER_TIMEOUT_SECONDS = 60.0
LLM_MAX_RETRIES = 3
LLM_MODEL = "gemini-2.5-flash"

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise EnvironmentError("GEMINI_API_KEY non trovata.")

llm = ChatGoogleGenerativeAI(
    model=LLM_MODEL,
    google_api_key=GEMINI_API_KEY,
)

def build_and_solve_with_floors(
    data: ProblemData,
    min_scores: Dict[str, float],
    max_time_seconds: float = 30.0,
    hints: Optional[Dict[str, Dict[int, Optional[str]]]] = None,
) -> ScheduleResult:
    """
    Ricostruisce il modello CP-SAT da zero (stateless) iniettando le soglie
    minime di soddisfazione come vincoli addizionali.

    Best Practice #1: Il codice è statico. Solo min_scores cambia ad ogni iterazione.
    Best Practice #3: Il modello viene ricreato da zero → zero 'vincoli fantasma'.
    Best Practice #5: max_time_seconds limita il tempo del solver per singola iterazione.

    Parameters
    ----------
    data            : ProblemData  – dati del problema (lavoratori, preferenze, staffing)
    min_scores      : dict         – {worker_id: soglia_minima_soddisfazione}
    max_time_seconds: float        – timeout OR-Tools per singola esecuzione

    Returns
    -------
    ScheduleResult con feasible=True/False e satisfaction_per_worker compilato.
    """
    model = cp_model.CpModel()
    num_days    = input_data.NUM_DAYS
    shift_codes = input_data.SHIFT_CODES   # ["M", "P", "N"]
    shifts      = input_data.SHIFTS
    hc          = input_data.HARD_CONSTRAINTS

    shift_hours  = {s: shifts[s]["durata_ore"]  for s in shift_codes}  # M/P=6, N=12
    shift_weight = {s: shifts[s]["peso_turni"]  for s in shift_codes}  # M/P=1, N=2

    # ------------------------------------------------------------------
    # VARIABILI DECISIONALI: x[(w, d, s)] ∈ {0, 1}
    # ------------------------------------------------------------------
    x = {
        (w, d, s): model.NewBoolVar(f"x_{w}_{d}_{s}")
        for w in data.worker_ids
        for d in range(num_days)
        for s in shift_codes
    }

    # ------------------------------------------------------------------
    # VINCOLI HARD (invariati, copiati 1:1 da drafting_agent.py)
    # ------------------------------------------------------------------

    # HC#3 (parte 1): max 1 turno al giorno
    for w in data.worker_ids:
        for d in range(num_days):
            model.Add(sum(x[(w, d, s)] for s in shift_codes) <= hc["max_turni_per_giorno"])

    # HC#2: esattamente 25 turni al mese (Notte vale 2)
    for w in data.worker_ids:
        model.Add(
            sum(shift_weight[s] * x[(w, d, s)]
                for d in range(num_days) for s in shift_codes)
            == hc["turni_mensili_esatti"]
        )

    # HC#1 + HC#6: max 36 ore settimanali + almeno 1 riposo/settimana
    week = 7
    for w in data.worker_ids:
        for t in range(0, num_days - week + 1):
            giorni_finestra = range(t, t + week)
            model.Add(
                sum(shift_hours[s] * x[(w, d, s)]
                    for d in giorni_finestra for s in shift_codes)
                <= hc["max_ore_settimanali"]
            )
            model.Add(
                sum(x[(w, d, s)] for d in giorni_finestra for s in shift_codes)
                <= week - hc["giorni_riposo_minimi"]
            )

    # HC#5: 2 giorni liberi obbligatori dopo ogni Notte
    riposi = hc["riposi_obbligatori_dopo_notte"]
    for w in data.worker_ids:
        for d in range(num_days):
            for k in range(1, riposi + 1):
                if d + k < num_days:
                    for s in shift_codes:
                        model.Add(x[(w, d, "N")] + x[(w, d + k, s)] <= 1)

    # HC#3 (parte 2): divieto Notte(d) → Mattina(d+1)
    for w in data.worker_ids:
        for d in range(num_days - 1):
            model.Add(x[(w, d, "N")] + x[(w, d + 1, "M")] <= 1)

    # Indisponibilità (da preferenze Fase 1)
    for w in data.worker_ids:
        for d in data.unavailable.get(w, set()):
            for s in shift_codes:
                model.Add(x[(w, d, s)] == 0)

    # Staffing (dipende dal caso)
    if data.case_label == "A":
        min_per_turno = data.staffing["min_lavoratori_per_turno"]
        for d in range(num_days):
            for s in shift_codes:
                model.Add(sum(x[(w, d, s)] for w in data.worker_ids) >= min_per_turno)
    else:
        min_std  = data.staffing["min_standard_per_turno"]
        min_spec = data.staffing["min_specializzati_per_turno"]
        for d in range(num_days):
            for s in shift_codes:
                model.Add(sum(x[(w, d, s)] for w in data.standard_ids)    >= min_std)
                model.Add(sum(x[(w, d, s)] for w in data.specialized_ids) >= min_spec)

    # ------------------------------------------------------------------
    # VINCOLI SOFT FLOOR (FASE 4) – PARAMETRIZZAZIONE A DIZIONARIO
    # Best Practice #1: iniettati come parametri, il codice OR-Tools non cambia mai.
    # ------------------------------------------------------------------
    # Costruiamo variabili IntVar per la soddisfazione per poter applicare
    # i vincoli di soglia in modo deterministico.
    satisfaction_vars: Dict[str, cp_model.IntVar] = {}
    for w in data.worker_ids:
        pesi = data.preferences[w]["satisfaction_weights"]
        # Calcola i limiti teorici per definire il dominio dell'IntVar
        min_pos_coef = sum(
            int(round(pesi[s] * SATISFACTION_SCALE))
            for d in range(num_days) for s in shift_codes
            if int(round(pesi[s] * SATISFACTION_SCALE)) > 0
        )
        max_neg_coef = sum(
            int(round(pesi[s] * SATISFACTION_SCALE))
            for d in range(num_days) for s in shift_codes
            if int(round(pesi[s] * SATISFACTION_SCALE)) < 0
        )
        satisfaction_vars[w] = model.NewIntVar(
            max_neg_coef,
            min_pos_coef,
            f"sat_{w}"
        )
        model.Add(
            satisfaction_vars[w] == sum(
                int(round(pesi[s] * SATISFACTION_SCALE)) * x[(w, d, s)]
                for d in range(num_days)
                for s in shift_codes
            )
        )

        # Applica la soglia minima per questo lavoratore (il cuore della Fase 4)
        raw_floor = min_scores.get(w, float("-inf"))
        if math.isfinite(raw_floor):
            floor_scaled = int(round(raw_floor * SATISFACTION_SCALE))
            if floor_scaled > max_neg_coef:
                model.Add(satisfaction_vars[w] >= floor_scaled)


    # Add Hints if provided
    if hints:
        for w in data.worker_ids:
            for d in range(num_days):
                for s in shift_codes:
                    if hints.get(w, {}).get(d) == s:
                        model.AddHint(x[(w, d, s)], 1)
                    else:
                        model.AddHint(x[(w, d, s)], 0)

    # ------------------------------------------------------------------
    # FUNZIONE OBIETTIVO: massimizza soddisfazione totale

    # ------------------------------------------------------------------
    model.Maximize(sum(satisfaction_vars.values()))

    # ------------------------------------------------------------------
    # RISOLUZIONE
    # ------------------------------------------------------------------
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds   = max_time_seconds  # Best Practice #5
    solver.parameters.num_search_workers    = 8
    solver.parameters.log_search_progress   = False

    status      = solver.Solve(model)
    status_name = solver.StatusName(status)
    feasible    = status in (cp_model.OPTIMAL, cp_model.FEASIBLE)

    result = ScheduleResult(
        case_label   = data.case_label,
        status_name  = status_name,
        feasible     = feasible,
        source       = "phase4_deterministic",
    )

    if not feasible:
        return result

    # Estrae la schedulazione e calcola i punteggi reali (non scalati)
    schedule: Dict[str, Dict[int, Optional[str]]] = {}
    for w in data.worker_ids:
        schedule[w] = {}
        for d in range(num_days):
            assegnato = None
            for s in shift_codes:
                if solver.Value(x[(w, d, s)]) == 1:
                    assegnato = s
                    break
            schedule[w][d] = assegnato

    result.schedule        = schedule
    result.objective_value = solver.ObjectiveValue() / SATISFACTION_SCALE

    for w in data.worker_ids:
        pesi = data.preferences[w]["satisfaction_weights"]
        result.satisfaction_per_worker[w] = round(
            sum(pesi[s] for d in range(num_days)
                for s in shift_codes if schedule[w][d] == s), 2
        )

    return result





def ask_llm_for_new_threshold(
    data: ProblemData,
    current_scores: Dict[str, float],
    current_min_scores: Dict[str, float],
    iteration: int,
    target_worker: str,
    max_retries: int = LLM_MAX_RETRIES,
) -> Optional[dict]:
    """Chiama l'LLM per determinare la nuova soglia per il lavoratore peggiore."""
    worst_score = current_scores[target_worker]
    
    workers_sorted = sorted(current_scores.items(), key=lambda kv: kv[1])
    ranking_str = "\n".join(
        f"  {wid} ({data.worker_names[wid]}): soddisfazione={score:.1f}, "
        f"soglia_minima_attuale={current_min_scores.get(wid, 'nessuna')}"
        for wid, score in workers_sorted
    )
    
    prompt = f"""Sei il "Fairness Optimizer" di SmartScheduler (Fase 4, iterazione {iteration}).

Il tuo compito è analizzare i punteggi di soddisfazione e alzare la soglia minima
del lavoratore meno soddisfatto identificato dalla Fase 3.

### PUNTEGGI ATTUALI
{ranking_str}

### REGOLE DA RISPETTARE
1. Il lavoratore identificato come peggiore è {target_worker} ({data.worker_names[target_worker]}) con score {worst_score:.1f}.
2. Proponi di alzare la sua soglia minima a un valore MAGGIORE del suo PUNTEGGIO ATTUALE.
3. Lo step deve essere conservativo (piccolo incremento).
4. La nuova soglia DEVE superare strettamente il punteggio attuale {worst_score:.1f}.

### OUTPUT RICHIESTO
Rispondi SOLO con un oggetto JSON valido.
Schema:
{{
  "target_worker": "{target_worker}",
  "new_min_score": <float>,
  "reasoning": "<breve spiegazione in italiano, max 2 righe>"
}}
"""
    for attempt in range(1, max_retries + 1):
        try:
            response = llm.invoke([HumanMessage(content=prompt)])
            raw_text = response.content
            if isinstance(raw_text, list):
                raw_text = "".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in raw_text)

            match = re.search(r"```json\s*(.*?)\s*```", raw_text, re.DOTALL)
            if not match:
                match = re.search(r"(\{.*?\})", raw_text, re.DOTALL)
            if not match:
                continue

            payload = json.loads(match.group(1))
            if "target_worker" not in payload or "new_min_score" not in payload:
                continue

            return payload

        except Exception as e:
            err = str(e)
            if "429" in err or "RESOURCE_EXHAUSTED" in err:
                time.sleep(60)
            else:
                time.sleep(2)

    return None

def run_refinement(data: ProblemData, base_result: ScheduleResult, initial_worst_worker: str) -> ScheduleResult:
    """
    Esegue il ciclo di raffinamento dell'equità (Fase 4).
    """
    print("=" * 60)
    print(f"🔄 FASE 4 – REFINEMENT AGENT | Caso {data.case_label}")
    print("=" * 60)
    
    # Inizialmente a -∞ per tutti → nessun vincolo aggiuntivo rispetto alla Fase 2
    min_scores: Dict[str, float] = {wid: float("-inf") for wid in data.worker_ids}
    
    best_result = base_result
    best_scores = dict(base_result.satisfaction_per_worker)
    
    # Usiamo il peggiore identificato dalla verifica
    current_worst_worker = initial_worst_worker

    for iteration in range(1, MAX_ITER + 1):
        print(f"\n🔄 Iterazione {iteration}/{MAX_ITER} - Focus: {current_worst_worker}")
        
        llm_decision = ask_llm_for_new_threshold(data, best_scores, min_scores, iteration, current_worst_worker)
        
        if not llm_decision:
            print("   ❌ LLM non ha prodotto JSON valido. Interruzione refinement.")
            break
            
        target_wid = llm_decision["target_worker"]
        new_min_score = float(llm_decision["new_min_score"])
        
        # Validazione corretta: se la nuova soglia NON migliora, interrompiamo il raffinamento.
        if new_min_score <= best_scores[target_wid]:
            print(f"   ⚠️  La nuova soglia ({new_min_score}) non migliora quella attuale ({best_scores[target_wid]}). Convergenza raggiunta.")
            break
            
        print(f"   🤖 LLM alza soglia di {target_wid} a {new_min_score}")
        print(f"   📝 {llm_decision.get('reasoning', '')}")
        
        new_min_scores = dict(min_scores)
        new_min_scores[target_wid] = new_min_score

        # Regola Robin Hood: nessuno può scendere sotto la soglia di chi sta peggio tra "gli altri"
        altri_scores = [best_scores[w] for w in data.worker_ids if w != target_wid]
        min_other_score = min(altri_scores) if altri_scores else float("-inf")

        for wid in data.worker_ids:
            if wid != target_wid:
                new_min_scores[wid] = max(min_scores.get(wid, float("-inf")), min_other_score)

        print(f"   ⚙️  Esecuzione solver...")
        result = build_and_solve_with_floors(data, new_min_scores, SOLVER_TIMEOUT_SECONDS, best_result.schedule)

        if not result.feasible:
            print(f"\n🏁 STOP: Ottimo di Pareto raggiunto (INFEASIBLE con soglia {new_min_score} per {target_wid}).")
            break
            
        # Aggiornamento stato
        min_scores = new_min_scores
        best_result = result
        best_scores = dict(result.satisfaction_per_worker)
        
        # Aggiorna il lavoratore peggiore per la prossima iterazione
        current_worst_worker = min(best_scores, key=best_scores.get)

    print(f"\n✅ FASE 4 COMPLETATA – Soddisfazione min: {min(best_scores.values()):.1f}")
    return best_result

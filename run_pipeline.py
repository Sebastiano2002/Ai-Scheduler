"""
run_pipeline.py
===============
Orchestratore end-to-end di SmartScheduler (il "Pulsante Unico").

Esegue in sequenza, con un solo comando, l'intero flusso multi-agente del
progetto per lo use case richiesto:

    Fase 1 - Workers Agent      : formalizza le preferenze in linguaggio
                                  naturale nel file 'formalized_preferences_case_X.py'.
    Fase 2 - Drafting Agent     : l'LLM (Gemini 2.5 Flash) SCRIVE il codice
                                  ortools.sat.python.cp_model della bozza;
                                  salva obbligatoriamente il codice generato in
                                  'draft_code_case_X.txt' (deliverable) e la
                                  bozza in 'schedule_case_X.csv'.
    Fase 3 - Verification Agent : verifica i vincoli hard (leggi inviolabili);
                                  se la bozza e' RIFIUTATA, la traccia d'errore
                                  viene rinviata al Drafting Agent per la
                                  revisione; se valida, identifica il
                                  lavoratore meno soddisfatto.
    Fase 4 - Refinement Agent   : ciclo iterativo Max-Min Fairness che alza il
                                  punteggio del lavoratore meno soddisfatto
                                  senza violare i vincoli hard, fino a
                                  INFEASIBLE o al limite di iterazioni;
                                  salva l'orario finale in
                                  'schedule_case_X_final.csv' e il codice in
                                  'final_code_case_X.txt'.

Esecuzione (richiede la variabile d'ambiente GEMINI_API_KEY):
    python run_pipeline.py --case A
    python run_pipeline.py --case B
    python run_pipeline.py --case all
"""

import argparse
import importlib

import Fase1_workers_agent
from llm_engine import AgentExecutor
from Fase2_drafting_agent import (
    export_csv,
    load_problem_data,
    print_summary,
    run_llm_drafting,
    save_generated_code,
)
from Fase3_verification_agent import print_report, verify_schedule
from Fase4_refinement_agent import print_refinement_summary, run_refinement_loop


# ---------------------------------------------------------------------------
# FASE 1 - Estrazione e formalizzazione delle preferenze
# ---------------------------------------------------------------------------
def run_phase_1(executor, case_label):
    """Esegue il Workers Agent e salva 'formalized_preferences_case_X.py'."""
    print(f"\n{'#'*64}")
    print(f"# PIPELINE | FASE 1 - WORKERS AGENT | Caso {case_label}")
    print(f"{'#'*64}")

    preferences_text = Fase1_workers_agent.load_preferences_text()
    out_path = Fase1_workers_agent.formalize_case(executor, case_label, preferences_text)
    if out_path is None:
        raise SystemExit(
            f"[!] Pipeline interrotta: la Fase 1 non ha prodotto le preferenze "
            f"formalizzate per il Caso {case_label}."
        )
    # Il file e' stato (ri)scritto ora: invalida le cache degli import cosi'
    # che load_problem_data() legga la versione appena generata.
    importlib.invalidate_caches()
    return out_path


# ---------------------------------------------------------------------------
# FASE 2 + FASE 3 - Bozza LLM e verifica (con rinvio in caso di rifiuto)
# ---------------------------------------------------------------------------
def run_phase_2_and_3(executor, data, max_time, max_draft_attempts):
    """
    Genera la bozza con il Drafting Agent e la sottopone al Verification Agent.
    Se il piano viene RIFIUTATO (violazioni hard), la traccia d'errore e'
    rinviata al Drafting Agent per la revisione, fino a max_draft_attempts.
    Ritorna (ScheduleResult, VerificationReport) della prima bozza valida.
    """
    feedback = None
    for attempt in range(1, max_draft_attempts + 1):
        print(f"\n{'#'*64}")
        print(f"# PIPELINE | FASE 2 - DRAFTING AGENT | Caso {data.case_label} "
              f"(tentativo {attempt}/{max_draft_attempts})")
        print(f"{'#'*64}")

        result = run_llm_drafting(
            executor, data, max_time=max_time, feedback=feedback
        )
        if result is None:
            print("[!] Il Drafting Agent non ha prodotto codice eseguibile. "
                  "Nuovo tentativo...")
            continue

        print_summary(data, result)

        # Deliverable obbligatori della Fase 2: il codice cp_model generato
        # dall'LLM (.txt) e la bozza di schedulazione (.csv).
        if result.feasible:
            export_csv(data, result)
            if result.generated_code:
                save_generated_code(data.case_label, result.generated_code)

        print(f"\n{'#'*64}")
        print(f"# PIPELINE | FASE 3 - VERIFICATION AGENT | Caso {data.case_label}")
        print(f"{'#'*64}")
        report = verify_schedule(data, result)
        print_report(data, report)

        if report.hard_ok:
            return result, report

        # Piano RIFIUTATO: la traccia d'errore torna al Drafting Agent.
        feedback = report.feedback_drafting
        print("\n[!] Bozza RIFIUTATA dal Verification Agent: la traccia d'errore "
              "viene rinviata al Drafting Agent per la revisione.")

    raise SystemExit(
        f"[!] Pipeline interrotta: nessuna bozza valida per il Caso "
        f"{data.case_label} dopo {max_draft_attempts} tentativi."
    )


# ---------------------------------------------------------------------------
# FASE 4 - Ciclo di raffinamento Max-Min Fairness e output finale
# ---------------------------------------------------------------------------
def run_phase_4(executor, data, result, report, max_iterations, max_time):
    """
    Esegue il ciclo di raffinamento dell'equita' (verifica inclusa a ogni
    iterazione) e salva l'orario finale (.csv) e il relativo codice (.txt).
    """
    outcome = run_refinement_loop(
        executor, data, result, report,
        max_iterations=max_iterations, max_time=max_time,
    )

    final_csv = f"schedule_case_{data.case_label}_final.csv"
    export_csv(data, outcome.best_result, path=final_csv)
    if outcome.best_result.generated_code:
        save_generated_code(
            data.case_label, outcome.best_result.generated_code,
            path=f"final_code_case_{data.case_label}.txt",
        )

    print_refinement_summary(data, outcome)
    return outcome, final_csv


# ---------------------------------------------------------------------------
# PIPELINE COMPLETA PER UNO USE CASE
# ---------------------------------------------------------------------------
def run_pipeline(case_label, max_time, max_iterations, max_draft_attempts):
    """Esegue Fase 1 -> Fase 2 -> Fase 3 -> Fase 4 per un singolo use case."""
    executor = AgentExecutor()

    # Fase 1: preferenze formalizzate (.py).
    run_phase_1(executor, case_label)

    # Dati del problema (anagrafica + preferenze appena formalizzate).
    data = load_problem_data(case_label)

    # Fase 2 + Fase 3: bozza LLM (.txt + .csv) verificata.
    result, report = run_phase_2_and_3(
        executor, data, max_time, max_draft_attempts
    )

    # Fase 4: raffinamento Max-Min Fairness -> orario finale (.csv).
    outcome, final_csv = run_phase_4(
        executor, data, result, report, max_iterations, max_time
    )

    print(f"\n{'='*64}")
    print(f"PIPELINE COMPLETATA | Caso {case_label}")
    print(f"{'='*64}")
    print(f"  Preferenze formalizzate : formalized_preferences_case_{case_label}.py")
    print(f"  Codice bozza (LLM)      : draft_code_case_{case_label}.txt")
    print(f"  Bozza schedulazione     : schedule_case_{case_label}.csv")
    print(f"  Codice finale (LLM)     : final_code_case_{case_label}.txt")
    print(f"  Orario finale           : {final_csv}")
    print(f"  Minimo equita'          : {outcome.initial_worst} -> {outcome.final_worst}")
    return outcome


def main():
    parser = argparse.ArgumentParser(
        description="SmartScheduler - Pipeline end-to-end "
                    "(Fase 1 -> Fase 2 -> Fase 3 -> Fase 4)."
    )
    parser.add_argument(
        "--case", choices=["A", "B", "all"], default="all",
        help="Use case da eseguire (default: all).",
    )
    parser.add_argument(
        "--max-time", type=float, default=60.0,
        help="Tempo massimo del solver per risoluzione in secondi (default: 60).",
    )
    parser.add_argument(
        "--max-iterations", type=int, default=25,
        help="SALVAGUARDIA: tetto dei livelli leximin del raffinamento (default: "
             "25, sopra il n. di lavoratori cosi' il leximin puo' completarsi). "
             "Solo il 1o livello e' una chiamata LLM; i successivi ri-eseguono lo "
             "stesso template (solo risoluzione CP). Il ciclo termina di norma "
             "quando tutti i lavoratori sono fissati o non si migliora piu'.",
    )
    parser.add_argument(
        "--max-draft-attempts", type=int, default=3,
        help="Tentativi massimi di bozza in caso di rifiuto della Fase 3 (default: 3).",
    )
    args = parser.parse_args()

    casi = ["A", "B"] if args.case == "all" else [args.case]
    for case_label in casi:
        run_pipeline(
            case_label,
            max_time=args.max_time,
            max_iterations=args.max_iterations,
            max_draft_attempts=args.max_draft_attempts,
        )


if __name__ == "__main__":
    main()

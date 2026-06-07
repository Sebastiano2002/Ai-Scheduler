"""
run_scheduler.py
================
Orchestratore Master:
1. Controlla e avvia la Fase 1 se le preferenze mancano.
2. Esegue la Fase 2 (Drafting Agent) per generare la bozza.
3. Esegue la Fase 3 (Verification Agent) per validare l'orario ed esportarlo.
"""

import argparse
import importlib
from drafting_agent import run_case, load_problem_data
from verification_agent import verify_schedule

def ensure_preferences():
    """Verifica che i file delle preferenze esistano, altrimenti avvia la Fase 1."""
    try:
        importlib.import_module("formalized_preferences_case_A")
        importlib.import_module("formalized_preferences_case_B")
    except ImportError:
        print("\n[!] Preferenze non trovate. Avvio automatico del Workers Agent (Fase 1)...")
        import workers_agent
        workers_agent.main()
        importlib.invalidate_caches()

def main():
    parser = argparse.ArgumentParser(description="Orchestratore Completo (Fase 1 -> Fase 2 -> Fase 3)")
    parser.add_argument("--case", choices=["A", "B", "all"], default="all", help="Use case da risolvere")
    parser.add_argument("--mode", choices=["deterministic", "llm"], default="deterministic", help="Modo per la Fase 2")
    args = parser.parse_args()

    # Fase 1: Assicuriamoci di avere i dati
    ensure_preferences()

    casi = ["A", "B"] if args.case == "all" else [args.case]
    
    for case_label in casi:
        print(f"\n========================================================")
        print(f"AVVIO FLUSSO SCHEDULAZIONE - CASO {case_label}")
        print(f"========================================================")
        
        # FASE 2: Drafting Agent (Genera la bozza)
        result = run_case(case_label, mode=args.mode)
        
        if result is None or not result.feasible:
            print(f"[!] Impossibile procedere con la Fase 3 per il Caso {case_label}.")
            continue
            
        # FASE 3: Verification Agent (Verifica matematica, equità e salvataggio)
        data = load_problem_data(case_label)
        verify_schedule(data, result)

if __name__ == "__main__":
    main()

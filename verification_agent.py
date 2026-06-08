"""
verification_agent.py
=====================
Fase 3 - Verification Agent.

Questo modulo verifica matematicamente una bozza di schedulazione (ScheduleResult):
1. Controllo Vincoli Hard: verifica che tutti i vincoli legali e ospedalieri siano
   rispettati (ore settimanali, riposi, staffing, ecc.).
2. Valutazione Equità: analizza i punteggi di soddisfazione e individua il
   lavoratore più svantaggiato.
3. Salvataggio: esporta l'orario validato in CSV.
"""

import csv
from typing import List, Optional, Tuple

import input_data
from drafting_agent import ProblemData, ScheduleResult


def verify_hard_constraints(data: ProblemData, result: ScheduleResult) -> List[str]:
    """
    Controllo rigoroso dei Vincoli Hard.
    Restituisce una lista di stringhe descrittive per ogni violazione trovata.
    Se la lista è vuota, l'orario è perfettamente legale.
    """
    violazioni: List[str] = []
    
    if not result.feasible or not result.schedule:
        return ["La schedulazione non è fattibile o è vuota."]

    sched = result.schedule
    num_days = input_data.NUM_DAYS
    codes = input_data.SHIFT_CODES
    weight = {s: input_data.SHIFTS[s]["peso_turni"] for s in codes}
    hours = {s: input_data.SHIFTS[s]["durata_ore"] for s in codes}
    hc = input_data.HARD_CONSTRAINTS

    for w in data.worker_ids:
        giorni = sched[w]
        
        # 1. Esattamente 25 turni pesati al mese
        tot_peso = sum(weight[giorni[d]] for d in range(num_days) if giorni[d])
        if tot_peso != hc["turni_mensili_esatti"]:
            violazioni.append(f"{w}: turni pesati = {tot_peso} (atteso {hc['turni_mensili_esatti']})")

        for d in range(num_days):
            # 2. Riposi obbligatori dopo il turno di notte e divieto Notte->Mattina
            if giorni[d] == "N":
                for k in range(1, hc["riposi_obbligatori_dopo_notte"] + 1):
                    if d + k < num_days and giorni[d + k] is not None:
                        violazioni.append(f"{w}: turno il giorno {d+k} dopo Notte al giorno {d} (manca riposo)")
            
            # 3. Indisponibilità (vincolo soft: se vogliamo segnarlo come warning possiamo farlo qui)
            # if giorni[d] is not None and d in data.unavailable.get(w, set()):
            #    pass

        # 4. Max ore settimanali e Almeno 1 giorno di riposo a settimana (finestra mobile 7 gg)
        for t in range(0, num_days - 7 + 1):
            ore = sum(hours[giorni[d]] for d in range(t, t + 7) if giorni[d])
            lavorati = sum(1 for d in range(t, t + 7) if giorni[d])
            
            if ore > hc["max_ore_settimanali"]:
                violazioni.append(f"{w}: {ore}h nella finestra giorni {t}-{t+6} (> {hc['max_ore_settimanali']}h ammesse)")
            
            if lavorati > 7 - hc["giorni_riposo_minimi"]:
                violazioni.append(f"{w}: {lavorati} giorni lavorati nella finestra {t}-{t+6} (manca giorno di riposo settimanale)")

    # 5. Controllo Copertura / Staffing
    for d in range(num_days):
        for s in codes:
            presenti = [w for w in data.worker_ids if sched[w][d] == s]
            if data.case_label == "A":
                min_req = data.staffing["min_lavoratori_per_turno"]
                if len(presenti) < min_req:
                    violazioni.append(f"Giorno {d} turno {s}: solo {len(presenti)} lavoratori (minimo {min_req})")
            else:
                n_std = sum(1 for w in presenti if w in data.standard_ids)
                n_spec = sum(1 for w in presenti if w in data.specialized_ids)
                if n_std < data.staffing["min_standard_per_turno"]:
                    violazioni.append(f"Giorno {d} turno {s}: solo {n_std} standard (minimo {data.staffing['min_standard_per_turno']})")
                if n_spec < data.staffing["min_specializzati_per_turno"]:
                    violazioni.append(f"Giorno {d} turno {s}: solo {n_spec} specializzati (minimo {data.staffing['min_specializzati_per_turno']})")
                    
    return violazioni


def evaluate_fairness(data: ProblemData, result: ScheduleResult) -> Tuple[Optional[str], Optional[float]]:
    """
    Analizza i punteggi di soddisfazione e individua il lavoratore più penalizzato.
    Restituisce (worker_id_peggiore, punteggio_peggiore).
    """
    sat = result.satisfaction_per_worker
    if not sat:
        return None, None
        
    peggiore_id = min(sat, key=sat.get)
    peggiore_score = sat[peggiore_id]
    
    return peggiore_id, peggiore_score


def export_csv(data: ProblemData, result: ScheduleResult, path: Optional[str] = None) -> str:
    """
    Salva la schedulazione in CSV (righe = lavoratori, colonne = date).
    """
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
                
                # Segnala visualmente se ha lavorato in un giorno di indisponibilità
                if s and d in data.unavailable.get(w, set()):
                    riga.append(f"{s} [FORZATO]")
                else:
                    riga.append(s if s else "-")
                    
                if s:
                    tot_peso += weight[s]
            riga.append(tot_peso)
            riga.append(result.satisfaction_per_worker.get(w, 0.0))
            writer.writerow(riga)

    print(f"[+] Schedulazione validata e salvata in: {path}")
    return path


def verify_schedule(data: ProblemData, result: ScheduleResult) -> Tuple[bool, Optional[str]]:
    """
    Funzione principale del Verification Agent:
    1. Verifica vincoli hard.
    2. Calcola equità.
    3. Se valida, esporta in CSV.
    Restituisce True se l'orario è valido, False altrimenti.
    """
    print(f"\n{'='*64}")
    print(f"FASE 3 - VERIFICATION AGENT | Caso {data.case_label}")
    print(f"{'='*64}")
    
    violazioni = verify_hard_constraints(data, result)
    
    if violazioni:
        print(f"[!] VERIFICA FALLITA: Trovate {len(violazioni)} violazioni ai vincoli Hard!")
        for v in violazioni[:10]:
            print(f"    - {v}")
        if len(violazioni) > 10:
            print(f"    ... e altre {len(violazioni) - 10} violazioni.")
        print("L'orario è stato scartato e non verrà salvato.")
        return False, None
        
    print("[+] VERIFICA SUPERATA: Tutti i vincoli istituzionali sono rispettati.")
    
    # Valutazione Equità
    sat = result.satisfaction_per_worker
    migliore_id = max(sat, key=sat.get)
    peggiore_id, peggiore_score = evaluate_fairness(data, result)
    
    print("\n--- VALUTAZIONE EQUITÀ (Fairness) ---")
    print(f"Soddisfazione totale : {round(sum(sat.values()), 2)}")
    print(f"Piu' soddisfatto     : {migliore_id} ({data.worker_names[migliore_id]}) = {sat[migliore_id]}")
    print(f"Meno soddisfatto     : {peggiore_id} ({data.worker_names[peggiore_id]}) = {peggiore_score}")
    print("-------------------------------------")
    
    # Esporta in CSV come risultato intermedio della Fase 3
    export_csv(data, result, path=f"schedule_case_{data.case_label}_intermedio.csv")
    
    return True, peggiore_id
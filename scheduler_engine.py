from ortools.sat.python import cp_model

def generate_schedule(num_workers, num_days, num_shifts, preferences, min_score_threshold, use_case_b=False, specialized_workers=None):
    if specialized_workers is None:
        specialized_workers = []
        
    model = cp_model.CpModel()
    turni = {}

    # 1. Creazione Variabili Booleane
    for w in range(num_workers):
        for d in range(num_days):
            for s in range(num_shifts):
                turni[(w, d, s)] = model.NewBoolVar(f'w{w}_d{d}_s{s}')

    # 2. Hard Constraints (Vincoli Rigidi)
    
    # Copertura turni
    for d in range(num_days):
        for s in range(num_shifts):
            if use_case_b:
                # Caso B: almeno 2 standard e 1 specializzato
                # Poiché uno specializzato può fare da standard, servono in totale >= 3 lavoratori,
                # di cui almeno 1 DEVE essere specializzato.
                model.Add(sum(turni[(w, d, s)] for w in range(num_workers)) >= 3)
                model.Add(sum(turni[(w, d, s)] for w in specialized_workers) >= 1)
            else:
                # Caso A: Almeno 2 lavoratori qualsiasi per ogni turno
                model.Add(sum(turni[(w, d, s)] for w in range(num_workers)) >= 2)

    for w in range(num_workers):
        # Esattamente 25 turni al mese per lavoratore
        model.Add(sum(turni[(w, d, s)] for d in range(num_days) for s in range(num_shifts)) == 25)
        
        for d in range(num_days):
            # Max 1 turno al giorno
            model.Add(sum(turni[(w, d, s)] for s in range(num_shifts)) <= 1)
            
            # Vincolo Notte (turno 2) -> 2 giorni liberi successivi
            if d <= num_days - 3:
                model.Add(
                    sum(turni[(w, d+1, s)] for s in range(num_shifts)) + 
                    sum(turni[(w, d+2, s)] for s in range(num_shifts)) == 0
                ).OnlyEnforceIf(turni[(w, d, 2)])
            elif d == num_days - 2:
                # Caso limite penultimo giorno
                model.Add(sum(turni[(w, d+1, s)] for s in range(num_shifts)) == 0).OnlyEnforceIf(turni[(w, d, 2)])
                
        # Max 36 ore a settimana (finestra mobile di 7 giorni)
        # Assumendo durate: s=0 (6h), s=1 (6h), s=2 (12h)
        for start_d in range(num_days - 6):
            model.Add(
                sum(
                    turni[(w, d, 0)] * 6 + 
                    turni[(w, d, 1)] * 6 + 
                    turni[(w, d, 2)] * 12 
                    for d in range(start_d, start_d + 7)
                ) <= 36
            )

    # 3. Vincolo di Equità (Soglia Minima)
    # Assicura che ogni lavoratore raggiunga almeno il punteggio soglia
    for w in range(num_workers):
        score_w = sum(turni[(w, d, s)] * preferences[w][d][s] for d in range(num_days) for s in range(num_shifts))
        model.Add(score_w >= min_score_threshold)

    # 4. Funzione Obiettivo (Massimizzazione soddisfazione totale)
    total_satisfaction = sum(
        turni[(w, d, s)] * preferences[w][d][s] 
        for w in range(num_workers) 
        for d in range(num_days) 
        for s in range(num_shifts)
    )
    model.Maximize(total_satisfaction)

    solver = cp_model.CpSolver()
    status = solver.Solve(model)

    if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
        return solver, turni
    return None, None
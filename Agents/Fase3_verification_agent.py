"""
Fase3_verification_agent.py
===========================
Fase 3 - Verifica della Schedulazione.

Implementa i due agenti di verifica:
  1. Hard Constraint Verifier: controlla matematicamente tutti i vincoli hard.
  2. Fairness Verifier: calcola le metriche di equita' e identifica il
     lavoratore piu' svantaggiato.
"""

import argparse
import csv
import os
import statistics
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from . import input_data
from .Fase2_drafting_agent import (
    ProblemData,
    ScheduleResult,
    compute_sat_max,
    load_problem_data,
    run_llm_drafting,
    worker_satisfaction,
    worker_satisfaction_pct,
)



@dataclass
class Violation:
    """Una singola violazione di un vincolo hard."""

    law: str
    descrizione: str
    worker_id: Optional[str] = None
    giorno: Optional[int] = None

    def __str__(self) -> str:
        ctx = []
        if self.worker_id is not None:
            ctx.append(self.worker_id)
        if self.giorno is not None:
            data_iso = input_data.PLANNING_DATES[self.giorno].isoformat()
            ctx.append(f"giorno {self.giorno} ({data_iso})")
        prefisso = f"[{self.law}]"
        suffisso = f" ({', '.join(ctx)})" if ctx else ""
        return f"{prefisso} {self.descrizione}{suffisso}"


@dataclass
class FairnessMetrics:
    """
    Esito del Fairness Verification Agent.
    Contiene le metriche di soddisfazione e identifica il lavoratore piu' svantaggiato.
    """

    satisfaction_per_worker: Dict[str, float]
    # Lavoratore più svantaggiato,
    # identificato sulla soddisfazione NORMALIZZATA.
    worst_worker_id: str
    worst_worker_name: str
    worst_satisfaction: float
    # Lavoratore più avvantaggiato.
    best_worker_id: str
    best_worker_name: str
    best_satisfaction: float
    # Metriche AGGREGATE di equità sulla distribuzione della soddisfazione.
    total_satisfaction: float   # soddisfazione complessiva
    mean_satisfaction: float    # media per lavoratore
    spread: float               # max - min: divario di disuguaglianza
    std_satisfaction: float     # deviazione standard (dispersione)
    
    # --- METRICHE NORMALIZZATE (proportional fairness) ------------------------
    # Soddisfazione in % del massimo individuale di ciascun lavoratore. Sono la
    # base CORRETTA per il confronto tra lavoratori e per il leximin (Fase 4).
    # Affiancate alle assolute per permettere il confronto diretto.
    sat_max_per_worker: Dict[str, float] = field(default_factory=dict)
    satisfaction_pct_per_worker: Dict[str, float] = field(default_factory=dict)
    worst_pct: float = 0.0      # % del più svantaggiato (minimo normalizzato)
    best_pct: float = 0.0       # % del più avvantaggiato (massimo normalizzato)
    mean_pct: float = 0.0       # % media
    spread_pct: float = 0.0     # scarto max-min in %
    std_pct: float = 0.0        # deviazione standard delle %


@dataclass
class VerificationReport:
    """
    Esito completo della Fase 3 per uno use case.

    hard_ok = True  -> il piano e' matematicamente valido (nessuna violazione);
                       in tal caso `fairness` e' valorizzato.
    hard_ok = False -> piano RIFIUTATO; `violations` e `feedback_drafting`
                       contengono la traccia d'errore per il Drafting Agent.
    """

    case_label: str
    schedule_source: str
    hard_ok: bool
    violations: List[Violation] = field(default_factory=list)
    fairness: Optional[FairnessMetrics] = None
    feedback_drafting: Optional[str] = None



class HardConstraintVerifier:
    """
    Ricontrolla, in modo indipendente tutte le leggi hard dal builder della Fase 2.
    """

    def __init__(self, data: ProblemData):
        self.data = data
        self.num_days = input_data.NUM_DAYS
        self.codes = input_data.SHIFT_CODES
        self.weight = {s: input_data.SHIFTS[s]["peso_turni"] for s in self.codes}
        self.hours = {s: input_data.SHIFTS[s]["durata_ore"] for s in self.codes}
        self.hc = input_data.HARD_CONSTRAINTS

    def _check_valid_shift_code(self, sched, viol):
        """Verifica che ogni cella contenga un codice turno ammesso (M/P/N) o None."""
        for w in self.data.worker_ids:
            for d in range(self.num_days):
                code = sched[w].get(d)
                if code is not None and code not in self.codes:
                    viol.append(Violation(
                        "CODE", f"codice turno non valido '{code}'", w, d))

    # -- legge #2: esattamente 25 turni mensili pesati ----------------------
    def _check_monthly_shifts(self, sched, viol):
        atteso = self.hc["turni_mensili_esatti"]
        for w in self.data.worker_ids:
            tot = sum(self.weight[sched[w][d]]
                      for d in range(self.num_days) if sched[w][d])
            if tot != atteso:
                viol.append(Violation(
                    "H2",
                    f"turni mensili pesati = {tot}, attesi esattamente {atteso} "
                    f"(la Notte pesa 2)",
                    w))

    # -- legge #1 (<=36h/sett) e #6 (>=1 riposo/sett) -----------------------
    def _check_weekly_windows(self, sched, viol):
        max_ore = self.hc["max_ore_settimanali"]
        max_lavorati = 7 - self.hc["giorni_riposo_minimi"]
        week = 7
        for w in self.data.worker_ids:
            for t in range(0, self.num_days - week + 1):
                finestra = range(t, t + week)
                ore = sum(self.hours[sched[w][d]] for d in finestra if sched[w][d])
                lavorati = sum(1 for d in finestra if sched[w][d])
                if ore > max_ore:
                    viol.append(Violation(
                        "H1",
                        f"{ore}h nella finestra giorni {t}-{t+6} (max {max_ore}h)",
                        w, t))
                if lavorati > max_lavorati:
                    viol.append(Violation(
                        "H6",
                        f"{lavorati} giorni lavorati nella finestra {t}-{t+6} "
                        f"(serve >=1 riposo, max {max_lavorati} lavorati)",
                        w, t))

    # -- legge #5: 2 riposi totali dopo la Notte ----------------------------
    def _check_rest_after_night(self, sched, viol):
        riposi = self.hc["riposi_obbligatori_dopo_notte"]
        for w in self.data.worker_ids:
            for d in range(self.num_days):
                if sched[w][d] == "N":
                    for k in range(1, riposi + 1):
                        nd = d + k
                        if nd < self.num_days and sched[w][nd] is not None:
                            viol.append(Violation(
                                "H5",
                                f"turno '{sched[w][nd]}' il giorno {nd} dopo la "
                                f"Notte del giorno {d}: servono {riposi} riposi totali",
                                w, nd))

    # -- legge #3 (parte 2): divieto assoluto Notte(d) -> Mattina(d+1) ------
    def _check_night_to_morning(self, sched, viol):
        for w in self.data.worker_ids:
            for d in range(self.num_days - 1):
                if sched[w][d] == "N" and sched[w][d + 1] == "M":
                    viol.append(Violation(
                        "H3b",
                        f"catena VIETATA Notte(giorno {d}) -> Mattina(giorno {d+1})",
                        w, d + 1))

    # -- turni vietati (vincolo HARD per-lavoratore della Fase 1) ------------
    def _check_forbidden_shifts(self, sched, viol):
        for w in self.data.worker_ids:
            vietati = self.data.forbidden.get(w, set())
            for d in range(self.num_days):
                code = sched[w].get(d)
                if code is not None and code in vietati:
                    viol.append(Violation(
                        "FORBIDDEN",
                        f"turno '{code}' assolutamente vietato per il lavoratore",
                        w, d))

    # -- Controlla se ci sono abbastanza lavoratori per ogni turno (minimo) --
    def _check_staffing(self, sched, viol):
        for d in range(self.num_days):
            for s in self.codes:
                presenti = [w for w in self.data.worker_ids if sched[w][d] == s]
                if self.data.case_label == "A":
                    minimo = self.data.staffing["min_lavoratori_per_turno"]
                    if len(presenti) < minimo:
                        viol.append(Violation(
                            "STAFFING",
                            f"turno {s}: {len(presenti)} lavoratori (min {minimo})",
                            giorno=d))
                else:  # Caso B
                    n_std = sum(1 for w in presenti if w in self.data.standard_ids)
                    n_spec = sum(1 for w in presenti if w in self.data.specialized_ids)
                    min_std = self.data.staffing["min_standard_per_turno"]
                    min_spec = self.data.staffing["min_specializzati_per_turno"]
                    if n_spec < min_spec:
                        viol.append(Violation(
                            "STAFFING",
                            f"turno {s}: {n_spec} specializzati (min {min_spec})",
                            giorno=d))
                    if n_std + n_spec < min_std + min_spec:
                        viol.append(Violation(
                            "STAFFING",
                            f"turno {s}: {n_std} standard + {n_spec} specializzati "
                            f"= {n_std + n_spec} totali (min {min_std + min_spec}, "
                            f"di cui almeno {min_spec} specializzati)",
                            giorno=d))

    def verify(self, result: ScheduleResult) -> List[Violation]:
        """Esegue tutti i controlli hard e restituisce la lista di violazioni."""
        viol: List[Violation] = []

        if not result.feasible:
            viol.append(Violation(
                "FEASIBILITY",
                f"il solver non ha prodotto una soluzione fattibile "
                f"(status: {result.status_name})"))
            return viol
        if not result.schedule:
            viol.append(Violation("FEASIBILITY", "schedulazione vuota"))
            return viol

        sched = result.schedule
        self._check_valid_shift_code(sched, viol)
        self._check_monthly_shifts(sched, viol)
        self._check_weekly_windows(sched, viol)
        self._check_rest_after_night(sched, viol)
        self._check_night_to_morning(sched, viol)
        self._check_forbidden_shifts(sched, viol)
        self._check_staffing(sched, viol)
        return viol


class FairnessVerifier:
    """
    Calcola il punteggio di soddisfazione di ogni lavoratore e identifica
    esplicitamente il lavoratore piu' svantaggiato. L'identificazione avviene
    sulla soddisfazione NORMALIZZATA (% del massimo individuale).
    """

    def __init__(self, data: ProblemData):
        self.data = data

    def evaluate(self, result: ScheduleResult) -> FairnessMetrics:
        sat = dict(result.satisfaction_per_worker)
        # Robustezza: assicura una voce per ogni lavoratore.
        for w in self.data.worker_ids:
            sat.setdefault(w, 0.0)

        # Massimo individuale e soddisfazione normalizzata (% del proprio ottimo).
        sat_max = compute_sat_max(self.data)
        pct = {w: worker_satisfaction_pct(sat[w], sat_max.get(w, 0.0))
               for w in self.data.worker_ids}

        # Il piu' svantaggiato e il piu' avvantaggiato sono determinati sulla
        # scala NORMALIZZATA (equa), non sui valori assoluti.
        worst = min(self.data.worker_ids, key=lambda w: pct[w])
        best = max(self.data.worker_ids, key=lambda w: pct[w])

        valori = [sat[w] for w in self.data.worker_ids]
        valori_pct = [pct[w] for w in self.data.worker_ids]
        dev = statistics.pstdev(valori) if len(valori) > 1 else 0.0
        dev_pct = statistics.pstdev(valori_pct) if len(valori_pct) > 1 else 0.0

        return FairnessMetrics(
            satisfaction_per_worker=sat,
            worst_worker_id=worst,
            worst_worker_name=self.data.worker_names[worst],
            worst_satisfaction=round(sat[worst], 2),
            best_worker_id=best,
            best_worker_name=self.data.worker_names[best],
            best_satisfaction=round(sat[best], 2),
            total_satisfaction=round(sum(valori), 2),
            mean_satisfaction=round(statistics.mean(valori), 2),
            spread=round(max(valori) - min(valori), 2),
            std_satisfaction=round(dev, 2),
            sat_max_per_worker=sat_max,
            satisfaction_pct_per_worker=pct,
            worst_pct=round(pct[worst], 1),
            best_pct=round(pct[best], 1),
            mean_pct=round(statistics.mean(valori_pct), 1),
            spread_pct=round(pct[best] - pct[worst], 1),
            std_pct=round(dev_pct, 1),
        )


def build_drafting_feedback(case_label: str, violations: List[Violation]) -> str:
    """
    Costruisce la traccia d'errore strutturata da rimandare al Drafting Agent
    quando il piano viene RIFIUTATO.
    """
    per_legge: Dict[str, List[Violation]] = {}
    for v in violations:
        per_legge.setdefault(v.law, []).append(v)

    righe = [
        f"VERIFICATION AGENT - Caso {case_label}: PIANO RIFIUTATO.",
        f"Rilevate {len(violations)} violazioni dei vincoli hard "
        f"(sono LEGGI inviolabili). Correggi la schedulazione.",
        "",
    ]
    for legge in sorted(per_legge):
        gruppo = per_legge[legge]
        righe.append(f"- Legge {legge}: {len(gruppo)} violazioni. Esempi:")
        for v in gruppo[:5]:
            righe.append(f"    * {v}")
        if len(gruppo) > 5:
            righe.append(f"    * ... e altre {len(gruppo) - 5}.")
    return "\n".join(righe)


def verify_schedule(data: ProblemData, result: ScheduleResult) -> VerificationReport:
    """
    Esegue la Fase 3 completa su una bozza di schedulazione: 
      1. Hard Constraint Verification Agent -> se viola, RIFIUTA + feedback.
      2. Fairness Verification Agent        -> solo se il piano e' valido.
    """
    hard_verifier = HardConstraintVerifier(data)
    violations = hard_verifier.verify(result)

    if violations:
        return VerificationReport(
            case_label=data.case_label,
            schedule_source=result.source,
            hard_ok=False,
            violations=violations,
            feedback_drafting=build_drafting_feedback(data.case_label, violations),
        )

    fairness = FairnessVerifier(data).evaluate(result)
    return VerificationReport(
        case_label=data.case_label,
        schedule_source=result.source,
        hard_ok=True,
        violations=[],
        fairness=fairness,
    )


def load_schedule_from_csv(data: ProblemData, path: str) -> ScheduleResult:
    """Ricostruisce una `ScheduleResult` dal CSV prodotto dalla Fase 2."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"CSV non trovato: {path}. Genera prima la bozza con la Fase 2 "
            f"(python Fase2_drafting_agent.py --case {data.case_label})."
        )

    iso_to_index = {d.isoformat(): i for i, d in enumerate(input_data.PLANNING_DATES)}
    schedule: Dict[str, Dict[int, Optional[str]]] = {
        w: {d: None for d in range(input_data.NUM_DAYS)} for w in data.worker_ids
    }

    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            wid = row.get("worker_id")
            if wid not in schedule:
                continue
            for col, val in row.items():
                if col in iso_to_index and val and val != "-":
                    schedule[wid][iso_to_index[col]] = val.strip()

    result = ScheduleResult(
        case_label=data.case_label,
        status_name="LOADED_FROM_CSV",
        feasible=True,
        schedule=schedule,
        source="csv",
    )
    # Ricalcola la soddisfazione per lavoratore col modello unico della Fase 2
    # (pesi dei turni + penalita' per i giorni indesiderati).
    for w in data.worker_ids:
        result.satisfaction_per_worker[w] = worker_satisfaction(data, schedule, w)
    result.objective_value = round(sum(result.satisfaction_per_worker.values()), 2)
    return result



def print_report(data: ProblemData, report: VerificationReport) -> None:
    print(f"\n{'='*64}")
    print(f"FASE 3 - VERIFICA | Caso {report.case_label} "
          f"(sorgente bozza: {report.schedule_source})")
    print(f"{'='*64}")

    # --- Esito Hard Constraint Verification Agent ---
    if not report.hard_ok:
        print(f"[X] PIANO RIFIUTATO: {len(report.violations)} violazioni hard.")
        print("    Traccia d'errore per il Drafting Agent (Fase 4):\n")
        print(report.feedback_drafting)
        return

    print("[OK] Vincoli hard: TUTTI rispettati (piano matematicamente valido).")

    # --- Esito Fairness Verification Agent ---
    fm = report.fairness
    print(f"\n  >> METRICHE DI EQUITA' (fairness metrics) | ASSOLUTA vs NORMALIZZATA (%):")
    print(f"     {'':<22}{'ASSOLUTA':>12}{'NORM. (%)':>12}")
    print(f"     {'Soddisfazione totale':<22}{fm.total_satisfaction:>12}{'-':>12}")
    print(f"     {'Media per lavoratore':<22}{fm.mean_satisfaction:>12}{fm.mean_pct:>11}%")
    print(f"     {'Minimo (peggiore)':<22}{fm.worst_satisfaction:>12}{fm.worst_pct:>11}%  "
          f"[{fm.worst_worker_id} {fm.worst_worker_name}]")
    print(f"     {'Massimo (migliore)':<22}{fm.best_satisfaction:>12}{fm.best_pct:>11}%  "
          f"[{fm.best_worker_id} {fm.best_worker_name}]")
    print(f"     {'Scarto max-min':<22}{fm.spread:>12}{fm.spread_pct:>11}%  (piu' basso = piu' equo)")
    print(f"     {'Deviazione standard':<22}{fm.std_satisfaction:>12}{fm.std_pct:>11}%")

    print(f"\n  >> LAVORATORE PIU' SVANTAGGIATO (input Fase 4, su scala NORMALIZZATA):")
    print(f"     {fm.worst_worker_id} ({fm.worst_worker_name}) "
          f"= {fm.worst_pct}% del proprio massimo ({fm.worst_satisfaction} su {fm.sat_max_per_worker.get(fm.worst_worker_id, 0.0)})")

    # Classifica completa: ordinata sulla scala NORMALIZZATA, 
    # con accanto la soddisfazione assoluta e il massimo.
    print(f"\n  Classifica per soddisfazione NORMALIZZATA (crescente):")
    print(f"     {'lavoratore':<26}{'norm.%':>8}{'assoluta':>10}{'max':>8}")
    ordinati = sorted(data.worker_ids, key=lambda w: fm.satisfaction_pct_per_worker[w])
    for w in ordinati:
        marcatore = "  <-- meno soddisfatto" if w == fm.worst_worker_id else ""
        nome = f"{w} {data.worker_names[w]}"
        print(f"     {nome:<26}{fm.satisfaction_pct_per_worker[w]:>7}%"
              f"{fm.satisfaction_per_worker[w]:>10}{fm.sat_max_per_worker.get(w, 0.0):>8}{marcatore}")



def run_case(case_label: str, from_csv: Optional[str], max_time: float = 30.0
             ) -> VerificationReport:
    """Esegue la Fase 3 per uno use case (genera o carica la bozza, poi verifica)."""
    data = load_problem_data(case_label)

    print(f"\n{'#'*64}")
    print(f"# FASE 3 - VERIFICATION AGENTS | Caso {data.case_label}")
    print(f"# Lavoratori: {len(data.worker_ids)} "
          f"(standard: {len(data.standard_ids)}, "
          f"specializzati: {len(data.specialized_ids)})")
    print(f"{'#'*64}")

    if from_csv is not None:
        path = from_csv if from_csv else f"Output/schedule_case_{case_label}.csv"
        print(f"[*] Carico la schedulazione da CSV: {path}")
        result = load_schedule_from_csv(data, path)
    else:
        print("[*] Genero la bozza con l'LLM (Fase 2) da verificare...")
        # Inferenza via Google Gemini 2.5 Flash: richiede GEMINI_API_KEY.
        from .llm_engine import AgentExecutor
        executor = AgentExecutor()
        result = run_llm_drafting(executor, data, max_time=max_time)
        if result is None:
            raise SystemExit(
                f"[!] Il Drafting Agent non ha prodotto una bozza per il Caso "
                f"{case_label}."
            )

    report = verify_schedule(data, result)
    print_report(data, report)
    return report


def main():
    parser = argparse.ArgumentParser(
        description="SmartScheduler Fase 3 - Verification Agents "
                    "(vincoli hard + equita')."
    )
    parser.add_argument(
        "--case", choices=["A", "B", "all"], default="all",
        help="Use case da verificare (default: all).",
    )
    parser.add_argument(
        "--from-csv", nargs="?", const="", default=None,
        help="Verifica una schedulazione salvata su CSV invece di rigenerarla. "
             "Opzionalmente passa il path (default: Output/schedule_case_<CASE>.csv).",
    )
    parser.add_argument(
        "--max-time", type=float, default=120.0,
        help="Tempo massimo del solver in secondi quando si rigenera (default: 120; "
             "il Caso B, piu' grande, puo' richiedere oltre 30s per chiudere).",
    )
    args = parser.parse_args()

    casi = ["A", "B"] if args.case == "all" else [args.case]
    for case_label in casi:
        run_case(case_label, args.from_csv, max_time=args.max_time)


if __name__ == "__main__":
    main()

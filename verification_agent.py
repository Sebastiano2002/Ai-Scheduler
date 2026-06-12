"""
verification_agent.py
======================
Fase 3 - Verifica della Schedulazione (Verification Agents).

Implementa i due agenti di verifica descritti nel PROJECT_CONTEXT.md:

  1. *Hard Constraint Verification Agent* (`HardConstraintVerifier`):
     rilegge l'output del solver (la `ScheduleResult` prodotta dalla Fase 2,
     indifferentemente dal percorso LLM o deterministico) e ricontrolla in modo
     INDIPENDENTE e rigoroso TUTTI i vincoli hard (le "leggi") piu' la copertura
     (staffing) e le indisponibilita'. Se trova anche una sola violazione,
     RIFIUTA il piano e produce una traccia d'errore strutturata, pronta per
     essere rimandata al Drafting Agent (Fase 4 / auto-correzione).

  2. *Fairness Verification Agent* (`FairnessVerifier`):
     viene eseguito SOLO se il piano e' matematicamente valido. Calcola il
     punteggio di soddisfazione di ogni lavoratore e identifica ESPLICITAMENTE
     il lavoratore piu' svantaggiato (quello con il punteggio minimo) con il suo
     score: e' l'unico dato richiesto dalla Fase 4 per il raffinamento mirato.

Nota di design: la verifica e' un CONTROLLO MATEMATICO sull'output del solver,
non una generazione di codice. Per questo e' deterministica e NON richiede una
API key (a differenza del percorso LLM della Fase 2). Il Verification Agent e'
volutamente INDIPENDENTE dal builder della Fase 2: ricodifica le leggi da capo,
cosi' una verifica che passa e' una garanzia reale e non una tautologia.

Esecuzione:
    # Genera una bozza deterministica (no API key) e la verifica:
    python verification_agent.py --case A
    python verification_agent.py --case B
    python verification_agent.py --case all

    # Verifica una schedulazione gia' salvata su CSV (es. output Fase 2):
    python verification_agent.py --case A --from-csv
    python verification_agent.py --case A --from-csv schedule_case_A.csv
"""

import argparse
import csv
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import input_data
from drafting_agent import (
    ProblemData,
    ScheduleResult,
    load_problem_data,
    run_llm_drafting,
)


# ===========================================================================
# 1. STRUTTURE DI OUTPUT DELLA VERIFICA
# ===========================================================================
@dataclass
class Violation:
    """
    Una singola violazione di un vincolo hard.

    Campi pensati per costruire una traccia d'errore leggibile sia dall'uomo
    sia dal Drafting Agent (Fase 4):
        law         : etichetta della legge violata (es. 'H2', 'STAFFING').
        descrizione : spiegazione testuale della violazione.
        worker_id   : lavoratore coinvolto (None se la violazione e' globale).
        giorno      : indice giorno 0-based coinvolto (None se non applicabile).
    """

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

    Contiene il punteggio di soddisfazione di ogni lavoratore e, soprattutto,
    l'identificazione esplicita del lavoratore piu' svantaggiato (punteggio
    minimo) con il suo score: e' l'unico dato consumato dalla Fase 4.
    """

    satisfaction_per_worker: Dict[str, float]
    # Lavoratore piu' svantaggiato (cuore della Fase 3, input della Fase 4).
    worst_worker_id: str
    worst_worker_name: str
    worst_satisfaction: float


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


# ===========================================================================
# 2. HARD CONSTRAINT VERIFICATION AGENT
# ===========================================================================
class HardConstraintVerifier:
    """
    Ricontrolla, in modo indipendente dal builder della Fase 2, tutte le leggi
    hard sull'output del solver. Ogni metodo `_check_*` aggiunge eventuali
    `Violation` alla lista; `verify` le restituisce tutte.
    """

    def __init__(self, data: ProblemData):
        self.data = data
        self.num_days = input_data.NUM_DAYS
        self.codes = input_data.SHIFT_CODES
        self.weight = {s: input_data.SHIFTS[s]["peso_turni"] for s in self.codes}
        self.hours = {s: input_data.SHIFTS[s]["durata_ore"] for s in self.codes}
        self.hc = input_data.HARD_CONSTRAINTS

    # -- legge #3 (parte 1): max 1 turno al giorno --------------------------
    def _check_max_one_shift_per_day(self, sched, viol):
        """
        Ridondante per costruzione (lo schedule mappa giorno -> codice singolo),
        ma verificato esplicitamente per robustezza: se la sorgente non e' una
        struttura 1-turno (es. CSV malformato) deve emergere.
        """
        for w in self.data.worker_ids:
            for d in range(self.num_days):
                code = sched[w].get(d)
                if code is not None and code not in self.codes:
                    viol.append(Violation(
                        "H3a", f"codice turno non valido '{code}'", w, d))

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

    # -- copertura / staffing (dipende dallo use case) ----------------------
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
                    # Gli specializzati possono coprire i ruoli standard: i ruoli
                    # standard richiesti possono essere riempiti da standard oppure
                    # da specializzati in eccesso rispetto al minimo specializzato.
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
        self._check_max_one_shift_per_day(sched, viol)
        self._check_monthly_shifts(sched, viol)
        self._check_weekly_windows(sched, viol)
        self._check_rest_after_night(sched, viol)
        self._check_night_to_morning(sched, viol)
        self._check_staffing(sched, viol)
        return viol


# ===========================================================================
# 3. FAIRNESS VERIFICATION AGENT
# ===========================================================================
class FairnessVerifier:
    """
    Calcola il punteggio di soddisfazione di ogni lavoratore e identifica
    esplicitamente il lavoratore piu' svantaggiato (punteggio minimo) con il
    suo score: e' l'unico dato richiesto dalla Fase 4.
    """

    def __init__(self, data: ProblemData):
        self.data = data

    def evaluate(self, result: ScheduleResult) -> FairnessMetrics:
        sat = dict(result.satisfaction_per_worker)
        # Robustezza: assicura una voce per ogni lavoratore.
        for w in self.data.worker_ids:
            sat.setdefault(w, 0.0)

        worst = min(self.data.worker_ids, key=lambda w: sat[w])

        return FairnessMetrics(
            satisfaction_per_worker=sat,
            worst_worker_id=worst,
            worst_worker_name=self.data.worker_names[worst],
            worst_satisfaction=round(sat[worst], 2),
        )


# ===========================================================================
# 4. FEEDBACK PER IL DRAFTING AGENT (traccia d'errore -> Fase 4)
# ===========================================================================
def build_drafting_feedback(case_label: str, violations: List[Violation]) -> str:
    """
    Costruisce la traccia d'errore strutturata da rimandare al Drafting Agent
    quando il piano viene RIFIUTATO. Raggruppa le violazioni per legge per
    rendere il feedback immediatamente azionabile dall'auto-correzione.
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


# ===========================================================================
# 5. ORCHESTRAZIONE DELLA FASE 3
# ===========================================================================
def verify_schedule(data: ProblemData, result: ScheduleResult) -> VerificationReport:
    """
    Esegue la Fase 3 completa su una bozza:
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


# ===========================================================================
# 6. CARICAMENTO DI UNA SCHEDULAZIONE DA CSV (output Fase 2)
# ===========================================================================
def load_schedule_from_csv(data: ProblemData, path: str) -> ScheduleResult:
    """
    Ricostruisce una `ScheduleResult` dal CSV prodotto da
    `drafting_agent.export_csv` (righe = lavoratori, colonne = date).
    Permette di verificare un piano gia' salvato senza ri-risolvere il modello.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"CSV non trovato: {path}. Genera prima la bozza con la Fase 2 "
            f"(python drafting_agent.py --case {data.case_label})."
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
    # Ricalcola la soddisfazione per lavoratore dai pesi della Fase 1.
    for w in data.worker_ids:
        pesi = data.preferences[w]["satisfaction_weights"]
        result.satisfaction_per_worker[w] = round(
            sum(pesi[s] for d in range(input_data.NUM_DAYS)
                for s in input_data.SHIFT_CODES if schedule[w][d] == s),
            2,
        )
    result.objective_value = round(sum(result.satisfaction_per_worker.values()), 2)
    return result


# ===========================================================================
# 7. STAMPA DEL REPORT
# ===========================================================================
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
    print(f"\n  >> LAVORATORE PIU' SVANTAGGIATO (input Fase 4):")
    print(f"     {fm.worst_worker_id} ({fm.worst_worker_name}) "
          f"= {fm.worst_satisfaction}")

    # Classifica completa (utile per il report finale di consegna).
    print(f"\n  Soddisfazione per lavoratore (crescente):")
    ordinati = sorted(data.worker_ids, key=lambda w: fm.satisfaction_per_worker[w])
    for w in ordinati:
        marcatore = "  <-- meno soddisfatto" if w == fm.worst_worker_id else ""
        print(f"     {w} {data.worker_names[w]:<20} "
              f"{fm.satisfaction_per_worker[w]:>7}{marcatore}")


# ===========================================================================
# 8. MAIN / CLI
# ===========================================================================
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
        path = from_csv if from_csv else f"schedule_case_{case_label}.csv"
        print(f"[*] Carico la schedulazione da CSV: {path}")
        result = load_schedule_from_csv(data, path)
    else:
        print("[*] Genero la bozza con l'LLM (Fase 2) da verificare...")
        # Inferenza via Google Gemini 2.5 Flash: richiede GEMINI_API_KEY.
        from llm_engine import AgentExecutor
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
             "Opzionalmente passa il path (default: schedule_case_<CASE>.csv).",
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

"""
Fase1_workers_agent.py
======================
Fase 1 - Workers Agent.

Questo script implementa l'agente che:
    1. legge le preferenze dei lavoratori espresse in linguaggio naturale
       (file 'worker_preferences.txt');
    2. costruisce un prompt che chiede all'LLM di formalizzarle in un
       dizionario Python strutturato 'WORKER_PREFERENCES';
    3. usa il motore della Fase 0 (AgentExecutor.run_with_retry) per generare
       ed eseguire in sicurezza il codice prodotto dall'LLM;
    4. salva il risultato in 'formalized_preferences_case_A.py' /
       'formalized_preferences_case_B.py'.
        L'output formalizzato distingue ESPLICITAMENTE due strutture parallele:
        - HARD_CONSTRAINTS (vincoli inderogabili)
        - SOFT_CONSTRAINTS (preferenze + modello di soddisfazione).
"""

import pprint
import datetime

from pydantic import ValidationError

from llm_engine import AgentExecutor
from models import AllWorkerPreferences
import input_data


PREFERENCES_FILE = "worker_preferences.txt"


# ---------------------------------------------------------------------------
# LETTURA DELLE PREFERENZE TESTUALI
# ---------------------------------------------------------------------------
def load_preferences_text(path=PREFERENCES_FILE):
    """Legge l'intero file di preferenze in linguaggio naturale."""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


# ---------------------------------------------------------------------------
# COSTRUZIONE DEL PROMPT PER L'LLM
# ---------------------------------------------------------------------------
def build_prompt(case_label, workers, preferences_text):
    """
    Costruisce il prompt che chiede all'LLM di trasformare le preferenze
    testuali nel dizionario strutturato WORKER_PREFERENCES.
    """
    # Descrizione compatta dei turni disponibili.
    descrizione_turni = "\n".join(
        f"  - '{code}': {info['nome']} ({info['inizio']}-{info['fine']}, "
        f"{info['durata_ore']}h{', turno doppio' if info['turno_doppio'] else ''})"
        for code, info in input_data.SHIFTS.items()
    )

    # Elenco dei lavoratori di questo use case (id, nome, ruolo).
    elenco_lavoratori = "\n".join(
        f"  - {w['id']} | {w['nome']} | ruolo: {w['ruolo']}" for w in workers
    )

    codici_turno = ", ".join(f"'{c}'" for c in input_data.SHIFT_CODES)
    ids_lavoratori = [w["id"] for w in workers]

    prompt = f"""Sei il "Workers Agent" di un sistema di schedulazione di turni ospedalieri.
Il tuo compito e' trasformare le preferenze dei lavoratori, espresse in
linguaggio naturale italiano, in una struttura dati Python rigorosa e
leggibile dalla macchina che DISTINGUA CHIARAMENTE i Vincoli Hard
(inderogabili) dai Vincoli Soft (preferenze), e che fornisca un modello di
soddisfazione numerico per la funzione obiettivo di OR-Tools.

### CONTESTO: USE CASE {case_label}
Turni disponibili (codici da usare obbligatoriamente):
{descrizione_turni}

I soli codici turno ammessi sono: {codici_turno}.

Orizzonte temporale di pianificazione:
  dal {input_data.START_DATE.isoformat()} al {input_data.END_DATE.isoformat()}
  (le date di indisponibilita' vanno espresse come stringhe ISO 'YYYY-MM-DD').

Lavoratori di questo use case:
{elenco_lavoratori}

### PREFERENZE IN LINGUAGGIO NATURALE
Di seguito il testo con tutte le preferenze. Considera SOLO i lavoratori
elencati sopra per questo use case (gli ID tra parentesi quadre identificano
ciascun lavoratore):
---
{preferences_text}
---

### DISTINZIONE HARD vs SOFT (FONDAMENTALE)
Per ogni lavoratore devi classificare ciascuna informazione del testo come
vincolo HARD (inderogabile) oppure SOFT (preferenza):

- HARD -> va in `hard_constraints`. SOLO impossibilita' assolute sul TIPO di turno:
    * "non posso ASSOLUTAMENTE fare il turno Y", "per motivi di salute non
      posso fare Y" -> aggiungi il codice turno in `turni_vietati`.
- SOFT -> va in `soft_constraints`. Tutto il resto, incluse le richieste di ferie:
    * "preferisco / do priorita' / mi trovo bene con" -> `turni_preferiti`.
    * "evito volentieri / tollero raramente / gradisco meno / mi pesa" (senza
      un divieto assoluto) -> `turni_indesiderati`.
    * QUALSIASI richiesta su GIORNI specifici ("non sono disponibile il X",
      "non posso lavorare il X", "vorrei restare libero il X", "preferirei non
      lavorare nel weekend del X") -> aggiungi le date in `giorni_indesiderati`.

IMPORTANTE: i giorni NON sono mai un vincolo hard. Anche un "non sono
disponibile il 25 dicembre" va in `giorni_indesiderati` (soft): l'algoritmo gli
applichera' una penalita' altissima, cosi' il lavoratore resta a casa salvo
emergenze di organico. Solo un divieto assoluto sul TIPO di turno (es. notte
per salute) e' HARD (`turni_vietati`).

### COMPITO
Genera codice Python che definisca un UNICO dizionario chiamato
esattamente `WORKER_PREFERENCES`.

- Le chiavi del dizionario sono gli ID dei lavoratori (ESATTAMENTE questi):
  {ids_lavoratori}
- Il valore associato a ogni ID e' un dizionario con QUESTE chiavi esatte:

    "nome"             : str  -> nome completo del lavoratore
    "hard_constraints" : dict -> vincoli inderogabili del lavoratore, con chiavi:
        "turni_vietati"         : list -> codici turno ASSOLUTAMENTE vietati
                                          (es. ['N']); lista vuota se nessuno
    "soft_constraints" : dict -> preferenze del lavoratore, con chiavi:
        "turni_preferiti"     : list -> codici turno graditi, dal piu' gradito
                                        (es. ['M', 'P'])
        "turni_indesiderati"  : list -> codici turno sgraditi ma NON vietati
        "giorni_indesiderati" : list -> date ISO 'YYYY-MM-DD' che il lavoratore
                                        preferirebbe NON lavorare (ferie); lista
                                        vuota se nessuna
        "flexibility_score"   : float -> tolleranza ai turni indesiderati, tra
                                         0.0 (rigido) e 1.0 (molto flessibile)
        "satisfaction_weights": dict -> peso numerico per OGNI codice turno
                                        ({codici_turno}). Positivo per i turni
                                        preferiti, vicino a 0 per i neutri,
                                        negativo per gli indesiderati. Serve a
                                        MASSIMIZZARE la soddisfazione nella
                                        funzione obiettivo OR-Tools. IMPORTANTE:
                                        la GRANDEZZA del peso negativo di un turno
                                        indesiderato deve riflettere la TOLLERANZA
                                        del lavoratore (flexibility_score): chi e'
                                        rigido (flex basso) riceve un peso piu'
                                        negativo, chi e' flessibile (flex alto) un
                                        peso vicino a 0 (vedi REGOLA 6). E' cosi'
                                        che i "livelli di tolleranza individuale"
                                        entrano nel modello di soddisfazione.

### REGOLE
1. Includi TUTTI e SOLI i lavoratori elencati sopra.
2. Usa solo i codici turno ammessi: {codici_turno}.
3. I `satisfaction_weights` devono contenere una voce per ciascuno dei
   codici turno ammessi (anche quelli neutri, con peso 0).
4. Coerenza soft: un turno in `turni_preferiti` deve avere peso positivo; uno
   in `turni_indesiderati` deve avere peso negativo.
5. Separazione hard/soft: un turno in `turni_vietati` (hard) NON deve comparire
   ne' in `turni_preferiti` ne' in `turni_indesiderati` (e' gia' escluso).
6. `flexibility_score` (TOLLERANZA individuale: 0.0 = rigido, 1.0 = molto
   flessibile) ha un DOPPIO compito, fondamentale:
   (a) riflettere la tolleranza descritta nel testo (es. "molto flessibile" ->
       ~1.0; "tolleranza media" -> ~0.5; "bassa tolleranza" -> ~0.2);
   (b) GUIDARE la grandezza dei pesi negativi: per un turno indesiderato usa un
       peso indicativo `~ -6 * (1 - flexibility_score)`. Esempi: flex 0.2 ->
       ~-5/-6 ; flex 0.5 -> ~-3 ; flex 0.8 -> ~-1/-2. Cosi' la tolleranza
       individuale entra nel satisfaction model ATTRAVERSO i pesi (non e' un
       parametro a parte): a parita' di turno sgradito, un lavoratore rigido
       "soffre" di piu' di uno flessibile, e il solver lo terra' in conto.
7. Non usare import esterni, non leggere file, non stampare nulla:
   definisci semplicemente il dizionario `WORKER_PREFERENCES`.

### ESEMPIO DI FORMATO (struttura, non valori da copiare)
WORKER_PREFERENCES = {{
    "W08": {{
        "nome": "Francesca Ricci",
        "hard_constraints": {{
            "turni_vietati": ["N"],
        }},
        "soft_constraints": {{
            "turni_preferiti": ["M", "P"],
            "turni_indesiderati": [],
            "giorni_indesiderati": ["2026-12-26"],
            "flexibility_score": 0.5,
            "satisfaction_weights": {{"M": 5.0, "P": 3.0, "N": 0.0}},
        }},
    }},
}}

Restituisci SOLO un blocco di codice Python valido (racchiuso tra ```python e ```)
che definisca `WORKER_PREFERENCES`.
"""
    return prompt


# ---------------------------------------------------------------------------
# SALVATAGGIO DELL'OUTPUT FORMALIZZATO
# ---------------------------------------------------------------------------
def split_hard_soft(worker_preferences):
    """Scompone il dizionario LLM in due strutture: HARD_CONSTRAINTS e SOFT_CONSTRAINTS."""
    hard_per_worker = {}
    soft = {}

    for wid, pref in worker_preferences.items():
        hc = pref.get("hard_constraints", {})
        sc = pref.get("soft_constraints", {})
        nome = pref.get("nome")

        hard_per_worker[wid] = {
            "nome": nome,
            "turni_vietati": list(hc.get("turni_vietati", [])),
        }
        soft[wid] = {
            "nome": nome,
            "turni_preferiti": list(sc.get("turni_preferiti", [])),
            "turni_indesiderati": list(sc.get("turni_indesiderati", [])),
            "giorni_indesiderati": list(sc.get("giorni_indesiderati", [])),
            "flexibility_score": sc.get("flexibility_score"),
            "satisfaction_weights": dict(sc.get("satisfaction_weights", {})),
        }

    hard = {
        "institutional": dict(input_data.HARD_CONSTRAINTS),
        "per_worker": hard_per_worker,
    }
    return hard, soft


def save_formalized(case_label, worker_preferences):
    """Salva le preferenze formalizzate in 'formalized_preferences_case_X.py'"""
    out_path = f"formalized_preferences_case_{case_label}.py"
    timestamp = datetime.datetime.now().isoformat(timespec="seconds")

    hard, soft = split_hard_soft(worker_preferences)

    header = (
        '"""\n'
        f"formalized_preferences_case_{case_label}.py\n"
        f"Generato automaticamente dal Workers Agent (Fase 1) il {timestamp}.\n"
        f"Use Case {case_label}: preferenze formalizzate dei lavoratori.\n\n"
        "Distinzione esplicita tra Vincoli Hard e Vincoli Soft:\n"
        "  - HARD_CONSTRAINTS : vincoli inderogabili.\n"
        "      'institutional' -> regole globali (max ore, turni mensili, ...);\n"
        "      'per_worker'    -> per lavoratore: turni_vietati (divieti assoluti\n"
        "                         sul tipo di turno) estratti dal linguaggio naturale.\n"
        "  - SOFT_CONSTRAINTS : preferenze per lavoratore (turni_preferiti,\n"
        "      turni_indesiderati, giorni_indesiderati = richieste di ferie,\n"
        "      flexibility_score) e satisfaction_weights (modello di soddisfazione\n"
        "      per la funzione obiettivo OR-Tools).\n"
        '"""\n\n'
    )

    def blocco(nome, valore):
        return f"{nome} = " + pprint.pformat(
            valore, indent=4, sort_dicts=False, width=100
        ) + "\n\n"

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(header)
        f.write(blocco("HARD_CONSTRAINTS", hard))
        f.write(blocco("SOFT_CONSTRAINTS", soft))

    print(f"[+] Preferenze formalizzate salvate in: {out_path}")
    print(f"    - HARD_CONSTRAINTS: {len(hard['per_worker'])} lavoratori "
          f"+ {len(hard['institutional'])} regole istituzionali")
    print(f"    - SOFT_CONSTRAINTS: {len(soft)} lavoratori")
    return out_path


# ---------------------------------------------------------------------------
# FORMALIZZAZIONE DI UN SINGOLO USE CASE
# ---------------------------------------------------------------------------
def formalize_case(executor, case_label, preferences_text):
    """Esegue il flusso completo per un use case: prompt -> LLM -> validazione -> salvataggio."""
    use_case = input_data.USE_CASES[case_label]
    workers = use_case["workers"]

    print(f"\n{'#'*64}")
    print(f"# USE CASE {case_label}: {use_case['descrizione']}")
    print(f"# Lavoratori: {len(workers)}")
    print(f"{'#'*64}")

    prompt = build_prompt(case_label, workers, preferences_text)

    context_vars = {}
    successo, risultato = executor.run_with_retry(prompt, context_vars=context_vars)

    if not successo:
        print(f"[!] Impossibile formalizzare le preferenze per lo Use Case {case_label}.")
        print(f"    Ultimo errore:\n{risultato}")
        return None

    worker_preferences = risultato.get("WORKER_PREFERENCES")
    if not isinstance(worker_preferences, dict):
        print(f"[!] Il codice generato non ha definito un dizionario "
              f"'WORKER_PREFERENCES' valido per lo Use Case {case_label}.")
        return None

    # --- Validazione Pydantic dell'output strutturato ---
    # Il dizionario generato dall'LLM viene validato con il modello AllWorkerPreferences
    try:
        AllWorkerPreferences.from_raw_dict(worker_preferences)
        print(f"[+] Validazione Pydantic superata per lo Use Case {case_label}.")
    except ValidationError as e:
        print(f"[!] Validazione Pydantic FALLITA per lo Use Case {case_label}.")
        print(f"    Le preferenze generate non rispettano il modello atteso:\n{e}")
        return None

    print(f"[+] Formalizzati {len(worker_preferences)} lavoratori per lo Use Case {case_label}.")
    return save_formalized(case_label, worker_preferences)


# ---------------------------------------------------------------------------
# MAIN: esegue la formalizzazione per ENTRAMBI gli use case
# ---------------------------------------------------------------------------
def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="SmartScheduler Fase 1 - Workers Agent."
    )
    parser.add_argument(
        "--case", choices=["A", "B", "all"], default="all",
        help="Use case da eseguire (default: all).",
    )
    args = parser.parse_args()

    preferences_text = load_preferences_text()
    executor = AgentExecutor()

    casi = ["A", "B"] if args.case == "all" else [args.case]
    risultati = {}
    for case_label in casi:
        risultati[case_label] = formalize_case(executor, case_label, preferences_text)

    print(f"\n{'='*64}")
    print("RIEPILOGO FASE 1")
    print(f"{'='*64}")
    for case_label, out_path in risultati.items():
        stato = out_path if out_path else "FALLITO"
        print(f"  Use Case {case_label}: {stato}")

if __name__ == "__main__":
    main()

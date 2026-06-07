"""
workers_agent.py
================
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

Il dizionario formalizzato distingue vincoli soft (preferenze) e associa a
ogni lavoratore un modello di soddisfazione (satisfaction_weights) utilizzabile
direttamente nella funzione obiettivo del modello OR-Tools della Fase 2.

Esecuzione:
    Impostare la chiave API di Gemini nella variabile d'ambiente GEMINI_API_KEY
    e lanciare:  python workers_agent.py
"""

import os
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
leggibile dalla macchina, che separi i vincoli soft (preferenze) e fornisca
un modello di soddisfazione numerico per la funzione obiettivo di OR-Tools.

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

### COMPITO
Genera codice Python che definisca un UNICO dizionario chiamato
esattamente `WORKER_PREFERENCES`.

- Le chiavi del dizionario sono gli ID dei lavoratori (ESATTAMENTE questi):
  {ids_lavoratori}
- Il valore associato a ogni ID e' un dizionario con QUESTE chiavi esatte:

    "nome"                 : str  -> nome completo del lavoratore
    "turni_preferiti"      : list -> codici turno graditi, ordinati dal piu'
                                     gradito (es. ['M', 'P'])
    "turni_indesiderati"   : list -> codici turno sgraditi (es. ['N'])
    "giorni_indisponibilita": list -> date ISO 'YYYY-MM-DD' in cui il
                                      lavoratore NON puo' lavorare
    "flexibility_score"    : float -> tolleranza ai turni indesiderati,
                                      tra 0.0 (rigido) e 1.0 (molto flessibile)
    "satisfaction_weights" : dict -> peso numerico per OGNI codice turno
                                     ({codici_turno}). Usa valori positivi per
                                     i turni preferiti, vicini a 0 per quelli
                                     neutri e negativi per quelli indesiderati.
                                     Questi pesi servono per MASSIMIZZARE la
                                     soddisfazione nella funzione obiettivo
                                     OR-Tools.

### REGOLE
1. Includi TUTTI e SOLI i lavoratori elencati sopra.
2. Usa solo i codici turno ammessi: {codici_turno}.
3. I `satisfaction_weights` devono contenere una voce per ciascuno dei
   codici turno ammessi (anche quelli neutri, con peso 0).
4. Coerenza: un turno in `turni_preferiti` deve avere peso positivo; uno in
   `turni_indesiderati` deve avere peso negativo.
5. `flexibility_score` deve riflettere la tolleranza descritta nel testo
   (es. "molto flessibile" -> vicino a 1.0; "bassa tolleranza" -> vicino a 0.2).
6. Non usare import esterni, non leggere file, non stampare nulla:
   definisci semplicemente il dizionario `WORKER_PREFERENCES`.

Restituisci SOLO un blocco di codice Python valido (racchiuso tra ```python e ```)
che definisca `WORKER_PREFERENCES`.
"""
    return prompt


# ---------------------------------------------------------------------------
# SALVATAGGIO DELL'OUTPUT FORMALIZZATO
# ---------------------------------------------------------------------------
def save_formalized(case_label, worker_preferences):
    """
    Serializza il dizionario WORKER_PREFERENCES in un file Python importabile
    'formalized_preferences_case_X.py'.
    """
    out_path = f"formalized_preferences_case_{case_label}.py"
    timestamp = datetime.datetime.now().isoformat(timespec="seconds")

    header = (
        '"""\n'
        f"formalized_preferences_case_{case_label}.py\n"
        f"Generato automaticamente dal Workers Agent (Fase 1) il {timestamp}.\n"
        f"Use Case {case_label}: preferenze formalizzate dei lavoratori.\n\n"
        "Struttura per ogni lavoratore:\n"
        "    turni_preferiti, turni_indesiderati, giorni_indisponibilita,\n"
        "    flexibility_score (0-1) e satisfaction_weights (pesi per OR-Tools).\n"
        '"""\n\n'
    )

    corpo = "WORKER_PREFERENCES = " + pprint.pformat(
        worker_preferences, indent=4, sort_dicts=False, width=100
    ) + "\n"

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(header)
        f.write(corpo)

    print(f"[+] Preferenze formalizzate salvate in: {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# FORMALIZZAZIONE DI UN SINGOLO USE CASE
# ---------------------------------------------------------------------------
def formalize_case(executor, case_label, preferences_text):
    """
    Esegue l'intero flusso per un use case: prompt -> LLM -> esecuzione sicura
    -> salvataggio.
    """
    use_case = input_data.USE_CASES[case_label]
    workers = use_case["workers"]

    print(f"\n{'#'*64}")
    print(f"# USE CASE {case_label}: {use_case['descrizione']}")
    print(f"# Lavoratori: {len(workers)}")
    print(f"{'#'*64}")

    prompt = build_prompt(case_label, workers, preferences_text)

    # Il codice generato dovra' popolare 'WORKER_PREFERENCES' in questo dizionario.
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
    # Il dizionario generato dall'LLM viene validato con il modello
    # AllWorkerPreferences (codici turno ammessi, flexibility_score in [0, 1],
    # coerenza tra preferenze e pesi di soddisfazione).
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
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise SystemExit(
            "[!] Variabile d'ambiente GEMINI_API_KEY non impostata.\n"
            "    Impostala con la tua chiave API di Gemini prima di eseguire.\n"
            "    PowerShell:  $env:GEMINI_API_KEY = 'la-tua-chiave'\n"
            "    Bash:        export GEMINI_API_KEY='la-tua-chiave'"
        )

    preferences_text = load_preferences_text()
    executor = AgentExecutor(api_key=api_key)

    risultati = {}
    for case_label in ("A", "B"):
        risultati[case_label] = formalize_case(executor, case_label, preferences_text)

    print(f"\n{'='*64}")
    print("RIEPILOGO FASE 1")
    print(f"{'='*64}")
    for case_label, out_path in risultati.items():
        stato = out_path if out_path else "FALLITO"
        print(f"  Use Case {case_label}: {stato}")


if __name__ == "__main__":
    main()

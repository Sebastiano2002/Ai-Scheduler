"""
llm_engine.py
=============
Fase 0 - Infrastruttura di esecuzione e orchestrazione (Ponte di Comando).

Questo modulo implementa l'`AgentExecutor`, lo strato che:
    1. dialoga con Google Gemini 2.5 Flash tramite l'SDK `google-genai`
       (`google.genai.Client`);
    2. estrae il codice Python generato;
    3. lo esegue in modo sicuro (`safe_execute`) catturando eventuali errori;
    4. reinvia gli errori all'LLM per l'auto-correzione, con un numero massimo
       di tentativi (`run_with_retry`).

Prerequisito: impostare la variabile d'ambiente GEMINI_API_KEY (oppure passare
la chiave direttamente al costruttore tramite il parametro `api_key`).
"""

import os
import re
import time
import traceback

from google import genai
from google.genai import types


class AgentExecutor:
    def __init__(self, api_key=None, model_name="gemini-2.5-flash"):
        """
        Inizializza il client Google Gemini tramite l'SDK google-genai.

        Usa google.genai.Client come backend: l'inferenza avviene sul modello
        Gemini 2.5 Flash via API remota. La chiave API puo' essere passata
        direttamente oppure letta dalla variabile d'ambiente GEMINI_API_KEY.
        """
        self.model_name = model_name
        # Ultimo codice Python eseguito con successo da run_with_retry. Permette
        # alle fasi a valle (Fase 4 - raffinamento) di recuperare il sorgente
        # cp_model effettivamente prodotto dall'LLM, non solo il suo risultato.
        self.last_code = None

        resolved_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not resolved_key:
            raise ValueError(
                "Chiave API Gemini mancante. Impostare la variabile d'ambiente "
                "GEMINI_API_KEY oppure passare api_key al costruttore."
            )

        self.client = genai.Client(api_key=resolved_key)
        print(f"[*] Motore LLM Inizializzato (Google Gemini) con il modello: {model_name}")

    def generate_and_extract(self, prompt, _rate_limit_retries=3):
        """
        Invia il prompt a Google Gemini ed estrae solo il codice Python generato.

        Il parametro `_rate_limit_retries` gestisce i tentativi in caso di errore
        di quota (HTTP 429) o errori temporanei dell'API remota.
        """
        for attempt in range(1, _rate_limit_retries + 1):
            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=0.1,
                        thinking_config=types.ThinkingConfig(
                            thinking_budget=0,
                        ),
                    ),
                )
                testo_risposta = response.text

                # `text` puo' essere None se il modello non ha prodotto testo
                if not testo_risposta:
                    print("[-] Attenzione: l'LLM non ha restituito testo.")
                    return None

                # Estrae il testo contenuto tra ```python e ```
                match = re.search(r"```python(.*?)```", testo_risposta, re.DOTALL)

                if match:
                    return match.group(1).strip()
                else:
                    print("[-] Attenzione: l'LLM non ha generato un blocco di codice valido.")
                    return None

            except Exception as e:
                err_str = str(e)
                # Gestione rate limit (HTTP 429): attesa esponenziale.
                if "429" in err_str or "quota" in err_str.lower():
                    wait = 30 * attempt
                    print(f"[-] Rate limit raggiunto (tentativo {attempt}/{_rate_limit_retries}). "
                          f"Attendo {wait}s prima di riprovare...")
                    time.sleep(wait)
                    if attempt == _rate_limit_retries:
                        print(f"[-] Errore di comunicazione con l'API Gemini: {e}")
                        return None
                else:
                    print(f"[-] Errore di comunicazione con l'API Gemini: {e}")
                    return None

    def safe_execute(self, code_str, context_vars):
        """
        Esegue il codice Python estratto in modo sicuro.
        Se il codice OR-tools ha errori, non fa crashare il programma ma cattura l'errore.

        NOTA TECNICA: si usa un namespace unico (globals + context_vars fusi) invece
        di passare globals() e context_vars separatamente a exec(). Con due namespace
        distinti, Python 3 non rende le variabili locali visibili all'interno di
        generator expression / list comprehension annidate (che creano il proprio
        scope e cercano i nomi solo nei globals). Fondendo tutto in un unico dict si
        elimina il NameError "x is not defined" che si verificava quando il codice
        generato usava `x` dentro una comprehension dopo averla definita nel corpo.
        """
        try:
            # Fondiamo globals() e context_vars in un unico namespace.
            # Le variabili pre-caricate (WORKER_IDS, cp_model, ecc.) sono già in
            # context_vars; le nuove variabili create dal codice (x, model, solver,
            # RESULT_SCHEDULE, …) finiscono nello stesso dizionario e sono quindi
            # visibili anche nelle scope annidate (comprehension, generator, lambda).
            namespace = {**globals(), **context_vars}
            exec(code_str, namespace)
            # Propaga le eventuali nuove variabili definite dal codice in context_vars
            # (es. RESULT_SCHEDULE, SOLVER_STATUS) così il chiamante le trova lì.
            context_vars.update(namespace)
            return True, context_vars

        except Exception:
            # Se l'LLM ha scritto codice non valido (es. KeyError, SyntaxError)
            errore_dettagliato = traceback.format_exc()
            return False, errore_dettagliato

    def run_with_retry(self, prompt, context_vars=None, max_retries=3):
        """
        Ciclo di auto-correzione completo (Fase 0 - Ponte di Comando).
        1. Genera il codice OR-Tools tramite LLM.
        2. Lo esegue in modo sicuro.
        3. Se fallisce, reinvia l'errore all'LLM chiedendo la correzione.
        4. Ripete fino a max_retries tentativi.
        """
        if context_vars is None:
            context_vars = {}

        codice_corrente = None
        errore = "Generazione codice fallita a causa di un errore API."

        for tentativo in range(1, max_retries + 1):
            print(f"\n{'='*60}")
            print(f"[*] Tentativo {tentativo}/{max_retries}")
            print(f"{'='*60}")

            # --- STEP 1: Genera (o ri-genera) il codice ---
            if codice_corrente is None:
                # Prima esecuzione: usa il prompt originale
                print("[*] Invio del prompt originale all'LLM...")
                codice_corrente = self.generate_and_extract(prompt)
            else:
                # Tentativi successivi: il prompt è già stato aggiornato con l'errore
                print("[*] Invio del prompt di correzione all'LLM...")
                codice_corrente = self.generate_and_extract(prompt_correzione)

            if codice_corrente is None:
                print("[-] L'LLM non ha generato codice valido. Riprovo...")
                # Prepara un prompt di correzione generico per il prossimo giro
                prompt_correzione = (
                    f"Il prompt originale era:\n---\n{prompt}\n---\n\n"
                    f"Non hai generato un blocco di codice Python valido "
                    f"(racchiuso tra ```python e ```). "
                    f"Per favore, riprova generando SOLO un blocco di codice Python valido."
                )
                continue

            print(f"[+] Codice generato ({len(codice_corrente)} caratteri). Esecuzione...")

            # --- STEP 2: Esegui il codice in modo sicuro ---
            successo, risultato = self.safe_execute(codice_corrente, context_vars)

            if successo:
                print(f"[+] Codice eseguito con successo al tentativo {tentativo}!")
                # Conserva il sorgente eseguito: la Fase 4 lo rilegge per costruire
                # il prompt di raffinamento a partire dalla bozza corrente.
                self.last_code = codice_corrente
                return True, risultato

            # --- STEP 3: Prepara il prompt di correzione con l'errore ---
            errore = risultato
            print(f"[-] Errore durante l'esecuzione:\n{errore}")

            prompt_correzione = (
                f"Il prompt originale era:\n---\n{prompt}\n---\n\n"
                f"Il codice Python che hai generato:\n"
                f"```python\n{codice_corrente}\n```\n\n"
                f"Ha prodotto il seguente errore durante l'esecuzione:\n"
                f"```\n{errore}\n```\n\n"
                f"Analizza l'errore e genera una versione CORRETTA del codice. "
                f"Restituisci SOLO il blocco di codice Python corretto."
            )

            # Reset per forzare la ri-generazione al prossimo ciclo
            codice_corrente = None

        # Se siamo qui, tutti i tentativi sono falliti
        print(f"\n[!] FALLITO: Il codice non è stato corretto dopo {max_retries} tentativi.")
        return False, errore

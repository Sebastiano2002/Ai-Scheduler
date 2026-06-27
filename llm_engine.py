"""
llm_engine.py
=============
Fase 0 - Infrastruttura di esecuzione e orchestrazione.

Questo modulo implementa l'`AgentExecutor`, lo strato che:
    1. dialoga con Google Gemini 2.5 Flash tramite l'SDK 'google-genai';
    2. estrae il codice Python generato;
    3. lo esegue in modo sicuro (`safe_execute`) catturando eventuali errori;
    4. reinvia gli errori all'LLM per l'auto-correzione, con un numero massimo
       di tentativi (`run_with_retry`).

Prerequisito: impostare la variabile d'ambiente GEMINI_API_KEY.
"""

import os
import re
import time
import traceback

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from google import genai
from google.genai import types


class AgentExecutor:
    def __init__(self, api_key=None, model_name="gemini-2.5-flash"):
        
        self.model_name = model_name
        # Ultimo codice Python eseguito con successo da run_with_retry. Permette
        # alla Fase 4 (raffinamento) di recuperare il sorgente cp_model effettivamente prodotto dall'LLM, non solo il suo risultato.
        self.last_code = None

        resolved_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not resolved_key:
            raise ValueError(
                "Chiave API Gemini mancante. Impostare la variabile d'ambiente "
            )

        self.client = genai.Client(api_key=resolved_key)
        print(f"[*] Motore LLM Inizializzato (Google Gemini) con il modello: {model_name}")

    def generate_and_extract(self, prompt, _rate_limit_retries=3):
        """
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

                # se il modello non ha prodotto testo:
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
                # Gestione rate limit (HTTP 429) con attesa esponenziale.
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
        Se il codice OR-tools ha errori li cattura.
        """
        try:
            # Fondiamo globals() e context_vars in un unico namespace.
            namespace = {**globals(), **context_vars}
            exec(code_str, namespace)
            # Propaga le eventuali nuove variabili definite dal codice in context_vars
            context_vars.update(namespace)
            return True, context_vars

        except Exception:
            # Se l'LLM ha scritto codice non valido (es. SyntaxError)
            errore_dettagliato = traceback.format_exc()
            return False, errore_dettagliato

    def run_with_retry(self, prompt, context_vars=None, max_retries=3):
        """
        Ciclo di auto-correzione completo 
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
                # Prima esecuzione:
                print("[*] Invio del prompt originale all'LLM...")
                codice_corrente = self.generate_and_extract(prompt)
            else:
                # Tentativi successivi: 
                print("[*] Invio del prompt di correzione all'LLM...")
                codice_corrente = self.generate_and_extract(prompt_correzione)

            if codice_corrente is None:
                print("[-] L'LLM non ha generato codice valido. Riprovo...")
                # Prepara un prompt di correzione per la successiva iterazione..
                prompt_correzione = (
                    f"Il prompt originale era:\n---\n{prompt}\n---\n\n"
                    f"Non hai generato un blocco di codice Python valido "
                    f"(racchiuso tra ```python e ```). "
                    f"Per favore, riprova generando SOLO un blocco di codice Python valido."
                )
                continue

            print(f"[+] Codice generato ({len(codice_corrente)} caratteri). Esecuzione...")

            successo, risultato = self.safe_execute(codice_corrente, context_vars)

            if successo:
                print(f"[+] Codice eseguito con successo al tentativo {tentativo}!")
                # Conserva il sorgente eseguito: la Fase 4 rilegge il prompt di raffinamento a partire dalla bozza corrente.
                self.last_code = codice_corrente
                return True, risultato

            # STEP 3: Prepara il prompt di correzione.
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

           
            codice_corrente = None

        # Se tutti i tentativi sono falliti.
        print(f"\n[!] FALLITO: Il codice non è stato corretto dopo {max_retries} tentativi.")
        return False, errore

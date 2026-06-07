"""
llm_engine.py
=============
Fase 0 - Infrastruttura di esecuzione e orchestrazione (Ponte di Comando).

Questo modulo implementa l'`AgentExecutor`, lo strato che:
    1. dialoga con l'LLM (Google Gemini) tramite LangChain
       (`ChatGoogleGenerativeAI` di `langchain_google_genai`);
    2. estrae il codice Python generato;
    3. lo esegue in modo sicuro (`safe_execute`) catturando eventuali errori;
    4. reinvia gli errori all'LLM per l'auto-correzione, con un numero massimo
       di tentativi (`run_with_retry`).

L'interfaccia pubblica (generate_and_extract, safe_execute, run_with_retry)
e' invariata rispetto alla versione basata su google.genai: cambia solo il
backend di comunicazione, ora costruito su LangChain.
"""

import re
import time
import traceback

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage


class AgentExecutor:
    def __init__(self, api_key, model_name="gemini-2.5-flash"):
        """
        Inizializza il client dell'LLM tramite LangChain.

        Usa ChatGoogleGenerativeAI (langchain_google_genai) come backend al
        posto delle chiamate dirette a google.genai. Il modello di default
        resta 'gemini-2.5-flash' e la chiave API viene passata esplicitamente
        (tipicamente letta da GEMINI_API_KEY a monte).
        """
        self.model_name = model_name
        # ChatGoogleGenerativeAI gestisce la comunicazione con l'API Gemini.
        # google_api_key accetta direttamente la chiave fornita.
        self.llm = ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=api_key,
        )
        print(f"[*] Motore LLM Inizializzato (LangChain) con il modello: {model_name}")

    def generate_and_extract(self, prompt, _rate_limit_retries=3):
        """
        Invia il prompt all'LLM ed estrae solo il codice Python generato.

        Se l'API restituisce 429 RESOURCE_EXHAUSTED (quota temporaneamente
        esaurita), attende il tempo suggerito e riprova automaticamente
        fino a `_rate_limit_retries` volte, senza consumare i tentativi
        del ciclo run_with_retry.
        """
        for rate_attempt in range(_rate_limit_retries + 1):
            try:
                # LangChain restituisce un AIMessage; il testo e' in `.content`.
                response = self.llm.invoke([HumanMessage(content=prompt)])
                testo_risposta = response.content

                # `content` puo' essere una stringa oppure una lista di blocchi
                # (a seconda della risposta): normalizziamo a stringa.
                if isinstance(testo_risposta, list):
                    testo_risposta = "".join(
                        blocco.get("text", "") if isinstance(blocco, dict) else str(blocco)
                        for blocco in testo_risposta
                    )

                # Estrae il testo contenuto tra ```python e ```
                match = re.search(r"```python(.*?)```", testo_risposta, re.DOTALL)

                if match:
                    return match.group(1).strip()
                else:
                    print("[-] Attenzione: l'LLM non ha generato un blocco di codice valido.")
                    return None

            except Exception as e:
                err_str = str(e)
                # Gestione specifica per 429 RESOURCE_EXHAUSTED:
                # attende il tempo suggerito e riprova senza sprecare tentativi.
                if "429" in err_str and "RESOURCE_EXHAUSTED" in err_str:
                    # Estrae il tempo di attesa suggerito dal messaggio di errore.
                    wait_match = re.search(r"retry.*?(\d+)", err_str, re.IGNORECASE)
                    wait_secs = int(wait_match.group(1)) + 5 if wait_match else 60
                    if rate_attempt < _rate_limit_retries:
                        print(f"[!] Quota API esaurita. Attendo {wait_secs}s prima di riprovare "
                              f"(tentativo rate-limit {rate_attempt+1}/{_rate_limit_retries})...")
                        time.sleep(wait_secs)
                        continue
                    else:
                        print(f"[-] Quota API ancora esaurita dopo {_rate_limit_retries} attese.")
                        return None
                else:
                    print(f"[-] Errore di comunicazione con l'LLM: {e}")
                    return None
        return None

    def safe_execute(self, code_str, context_vars):
        """
        Esegue il codice Python estratto in modo sicuro.
        Se il codice OR-tools ha errori, non fa crashare il programma ma cattura l'errore.
        """
        try:
            # Esegue il codice generato dall'LLM, passandogli le variabili di contesto
            exec(code_str, globals(), context_vars)
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

"""
input_data.py
=============
Fase 1 - Definizione strutturata dei dati del problema SmartScheduler.

Questo modulo raccoglie, sotto forma di costanti Python, tutte le informazioni
descritte nel PROJECT_CONTEXT.md:
    - i turni giornalieri (Mattina / Pomeriggio / Notte);
    - i vincoli obbligatori (hard constraints);
    - l'orizzonte temporale di pianificazione;
    - l'elenco dei lavoratori per i due Use Case (A e B).

Configurazioni lavoratori:
    - Use Case A: 13 lavoratori OMOGENEI (tutti standard, W01-W13).
    - Use Case B: 20 lavoratori (W01-W13 standard + W14-W20 specializzati).

Tutti gli altri moduli delle fasi successive importano da qui, in modo da avere
un'unica fonte di verita' per i dati del problema.
"""

from datetime import date, timedelta

# ---------------------------------------------------------------------------
# TURNI GIORNALIERI
# ---------------------------------------------------------------------------
# Tre turni al giorno. Il turno di Notte ha durata 12h ed e' considerato un
# "turno doppio" perche' copre l'intervallo 20:00 -> 08:00 del giorno seguente.
SHIFTS = {
    "M": {
        "nome": "Mattina",
        "inizio": "08:00",
        "fine": "14:00",
        "durata_ore": 6,
        "turno_doppio": False,
        # Conteggio ai fini del limite mensile (25 turni): la Notte vale 2.
        "peso_turni": 1,
    },
    "P": {
        "nome": "Pomeriggio",
        "inizio": "14:00",
        "fine": "20:00",
        "durata_ore": 6,
        "turno_doppio": False,
        "peso_turni": 1,
    },
    "N": {
        "nome": "Notte",
        "inizio": "20:00",
        "fine": "08:00",
        "durata_ore": 12,
        "turno_doppio": True,
        "peso_turni": 2,
    },
}

# Comodita': lista ordinata dei codici turno.
SHIFT_CODES = list(SHIFTS.keys())  # ["M", "P", "N"]

# ---------------------------------------------------------------------------
# VINCOLI OBBLIGATORI (HARD CONSTRAINTS)
# ---------------------------------------------------------------------------
# Regole istituzionali inderogabili che la schedulazione deve sempre rispettare.
HARD_CONSTRAINTS = {
    # Massimo monte ore di lavoro per dipendente in una settimana.
    "max_ore_settimanali": 36,
    # Numero esatto di turni che ogni lavoratore deve svolgere nel mese.
    "turni_mensili_esatti": 25,
    # Al massimo un turno per giorno per lavoratore.
    "max_turni_per_giorno": 1,
    # Divieto assoluto di lavorare in due giorni di calendario consecutivi.
    "turni_consecutivi_vietati": True,
    # Il turno di notte e' un turno doppio (12h).
    "notte_turno_doppio": True,
    # Giorni di riposo obbligatori subito dopo un turno di notte.
    "riposi_obbligatori_dopo_notte": 2,
    # Garantire almeno questo numero di giorni di riposo (valutando preferenze).
    "giorni_riposo_minimi": 1,
}

# ---------------------------------------------------------------------------
# ORIZZONTE TEMPORALE
# ---------------------------------------------------------------------------
# Periodo di pianificazione di un mese: dal 7 Dicembre 2026 al 6 Gennaio 2027.
START_DATE = date(2026, 12, 7)
END_DATE = date(2027, 1, 6)
NUM_DAYS = (END_DATE - START_DATE).days + 1  # 31 giorni

# Elenco esplicito di tutte le date dell'orizzonte (utile per i modelli OR-Tools).
PLANNING_DATES = [START_DATE + timedelta(days=i) for i in range(NUM_DAYS)]

# Giorni festivi compresi nell'orizzonte (rilevanti per indisponibilita').
HOLIDAYS = {
    date(2026, 12, 8): "Immacolata Concezione",
    date(2026, 12, 25): "Natale",
    date(2026, 12, 26): "Santo Stefano",
    date(2027, 1, 1): "Capodanno",
    date(2027, 1, 6): "Epifania",
}

# ---------------------------------------------------------------------------
# LAVORATORI
# ---------------------------------------------------------------------------
# Configurazione dei lavoratori per i due use case:
#   - Use Case A: 13 lavoratori OMOGENEI (tutti 'standard', W01-W13).
#   - Use Case B: 20 lavoratori totali:
#       * W01-W13 restano tutti 'standard' (13 persone);
#       * W14-W20 sono nuovi lavoratori 'specializzati' (7 persone).

# Anagrafica dei 13 lavoratori standard (comuni ad entrambi gli use case).
_ANAGRAFICA_STANDARD = [
    ("W01", "Marco Rossi"),
    ("W02", "Giulia Bianchi"),
    ("W03", "Luca Ferrari"),
    ("W04", "Sara Russo"),
    ("W05", "Andrea Esposito"),
    ("W06", "Chiara Romano"),
    ("W07", "Matteo Colombo"),
    ("W08", "Francesca Ricci"),
    ("W09", "Davide Greco"),
    ("W10", "Elena Marino"),
    ("W11", "Alessandro Conti"),
    ("W12", "Valentina Bruno"),
    ("W13", "Simone Gallo"),
]

# Anagrafica dei 7 lavoratori specializzati (solo Use Case B).
_ANAGRAFICA_SPECIALIZZATI_B = [
    ("W14", "Roberto Ferrara"),
    ("W15", "Monica Cattaneo"),
    ("W16", "Stefano Leone"),
    ("W17", "Laura Mancini"),
    ("W18", "Gianni Serra"),
    ("W19", "Paola Costa"),
    ("W20", "Fabio Martini"),
]

# Use Case A: 13 lavoratori omogenei (tutti 'standard').
WORKERS_CASE_A = [
    {"id": wid, "nome": nome, "ruolo": "standard"}
    for wid, nome in _ANAGRAFICA_STANDARD
]

# Use Case B: 20 lavoratori = 13 standard (W01-W13) + 7 specializzati (W14-W20).
WORKERS_CASE_B = [
    {"id": wid, "nome": nome, "ruolo": "standard"}
    for wid, nome in _ANAGRAFICA_STANDARD
] + [
    {"id": wid, "nome": nome, "ruolo": "specializzato"}
    for wid, nome in _ANAGRAFICA_SPECIALIZZATI_B
]

# ID degli specializzati nel Use Case B (comodo per lookup rapidi).
SPECIALIZED_IDS_CASE_B = {wid for wid, _ in _ANAGRAFICA_SPECIALIZZATI_B}

# ---------------------------------------------------------------------------
# REQUISITI DI COPERTURA (STAFFING) PER TURNO
# ---------------------------------------------------------------------------
# Use Case A: almeno 2 lavoratori per ogni turno.
STAFFING_CASE_A = {
    "min_lavoratori_per_turno": 2,
}

# Use Case B: minimo 2 standard + 1 specializzato per turno.
# Gli specializzati possono coprire i ruoli standard quando necessario.
STAFFING_CASE_B = {
    "min_standard_per_turno": 2,
    "min_specializzati_per_turno": 1,
    "specializzati_coprono_standard": True,
}

# ---------------------------------------------------------------------------
# REGISTRO USE CASE
# ---------------------------------------------------------------------------
# Mappa comoda usata dagli agenti delle fasi successive per iterare sui casi.
USE_CASES = {
    "A": {
        "descrizione": "13 lavoratori omogenei (tutti standard), minimo 2 per turno.",
        "workers": WORKERS_CASE_A,
        "staffing": STAFFING_CASE_A,
    },
    "B": {
        "descrizione": "20 lavoratori: 13 standard (W01-W13) + 7 specializzati (W14-W20), minimo 2 std + 1 spec per turno.",
        "workers": WORKERS_CASE_B,
        "staffing": STAFFING_CASE_B,
    },
}


if __name__ == "__main__":
    # Piccola stampa di verifica dei dati caricati.
    print("=== SmartScheduler - Dati del problema (Fase 1) ===")
    print(f"Turni definiti      : {', '.join(SHIFT_CODES)}")
    print(f"Orizzonte temporale : {START_DATE} -> {END_DATE} ({NUM_DAYS} giorni)")
    n_std_b = sum(1 for w in WORKERS_CASE_B if w["ruolo"] == "standard")
    n_spec_b = sum(1 for w in WORKERS_CASE_B if w["ruolo"] == "specializzato")
    print(f"Use Case A          : {len(WORKERS_CASE_A)} lavoratori (tutti standard, W01-W13)")
    print(f"Use Case B          : {len(WORKERS_CASE_B)} lavoratori "
          f"({n_std_b} standard W01-W13 + {n_spec_b} specializzati W14-W20)")
    print(f"Vincoli hard        : {len(HARD_CONSTRAINTS)} regole")

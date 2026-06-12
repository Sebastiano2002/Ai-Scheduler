# 🔍 Analisi Aggiornata — Distinzione Hard vs Soft Constraints

Il PDF del progetto richiede che il sistema distingua **esplicitamente** tra:

```
Hard constraints:
  - Legal requirements (requisiti legali)
  - Minimum staffing levels (livelli minimi di personale)
  - Maximum working hours (ore massime di lavoro)
  - Mandatory rest periods (periodi di riposo obbligatori)

Soft constraints:
  - Personal preferences (preferenze personali)
  - Shift desirability (desiderabilità dei turni)
  - Individual tolerance levels (livelli di tolleranza individuali)
```

---

## Stato attuale nel codice

### Hard Constraints — dove sono?

In [input_data.py:63-78](file:///c:/Users/Utente/Desktop/SmartScheduler_Progetto/input_data.py#L63-L78) esiste `HARD_CONSTRAINTS`:

```python
HARD_CONSTRAINTS = {
    "max_ore_settimanali": 36,          # → Maximum working hours ✅
    "turni_mensili_esatti": 25,         # → Legal requirements ✅
    "max_turni_per_giorno": 1,          # → Legal requirements ✅
    "turni_consecutivi_vietati": True,  # → Legal requirements ✅
    "notte_turno_doppio": True,         # → Legal requirements ✅
    "riposi_obbligatori_dopo_notte": 2, # → Mandatory rest periods ✅
    "giorni_riposo_minimi": 1,          # → Mandatory rest periods ✅
}
```

Lo staffing è separato in `STAFFING_CASE_A` / `STAFFING_CASE_B` ([input_data.py:159-169](file:///c:/Users/Utente/Desktop/SmartScheduler_Progetto/input_data.py#L159-L169)) → **Minimum staffing levels ✅**

> [!NOTE]
> I Hard constraints sono **tutti presenti e funzionanti**, ma non sono sotto-categorizzati come richiede il PDF (Legal requirements / Minimum staffing / Maximum working hours / Mandatory rest periods). È tutto in un unico dizionario piatto.

### Soft Constraints — dove sono?

I soft constraints sono **distribuiti** nella struttura `WorkerPreference` in [models.py:26-102](file:///c:/Users/Utente/Desktop/SmartScheduler_Progetto/models.py#L26-L102):

| Requisito PDF | Campo nel codice | Usato nel modello CP-SAT? |
|---|---|---|
| Personal preferences | `turni_preferiti`, `turni_indesiderati` | ⚠️ Solo **indirettamente** via `satisfaction_weights` |
| Shift desirability | `satisfaction_weights` | ✅ Sì, nella funzione obiettivo |
| Individual tolerance levels | `flexibility_score` | ❌ **NON usato** nel Drafting Agent |

---

## ❌ Problemi trovati

### Problema 1: `flexibility_score` NON È USATO nel modello CP-SAT

> [!WARNING]
> Il campo `flexibility_score` è:
> - ✅ Definito nel modello Pydantic ([models.py:42-47](file:///c:/Users/Utente/Desktop/SmartScheduler_Progetto/models.py#L42-L47))
> - ✅ Estratto dall'LLM nella Fase 1
> - ✅ Validato (range 0.0-1.0)
> - ✅ Menzionato nel prompt della Fase 4 di raffinamento ([refinement_agent.py:93](file:///c:/Users/Utente/Desktop/SmartScheduler_Progetto/refinement_agent.py#L93))
> - ❌ **MAI passato** al Drafting Agent nella Fase 2
> 
> Nel prompt del Drafting Agent ([drafting_agent.py:186](file:///c:/Users/Utente/Desktop/SmartScheduler_Progetto/drafting_agent.py#L186)), la variabile `PREFERENCES` contiene solo `satisfaction_weights`. Il `flexibility_score` non è né nel prompt né nel namespace di esecuzione.
> 
> **Il PDF chiede esplicitamente "Individual tolerance levels" come soft constraint.** Il `flexibility_score` è il campo che lo rappresenta, ma non viene integrato nella generazione della schedulazione.

**Cosa fare:**
- Nel prompt della Fase 2 (`build_drafting_prompt`), aggiungere `flexibility_score` alla descrizione di `PREFERENCES` e istruire l'LLM a usarlo come **peso modulatore** nella funzione obiettivo. Esempio: pesare i turni indesiderati anche in base alla tolleranza del lavoratore (`satisfaction_weights[s] * (1 - flexibility_score)` per i turni negativi).

---

### Problema 2: Manca una struttura dati esplicita `SOFT_CONSTRAINTS`

> [!WARNING]
> Il progetto richiede di distinguere **chiaramente** Hard vs Soft. Nel codice:
> - `HARD_CONSTRAINTS` esiste come dizionario esplicito in [input_data.py:63](file:///c:/Users/Utente/Desktop/SmartScheduler_Progetto/input_data.py#L63)
> - **Non esiste** un corrispondente `SOFT_CONSTRAINTS` o una categorizzazione equivalente
> 
> I soft constraints sono impliciti nella struttura `WorkerPreference` ma non c'è nessuna struttura che li elenchi e li categorizzi come fa `HARD_CONSTRAINTS`.

**Cosa fare:**
- In [input_data.py](file:///c:/Users/Utente/Desktop/SmartScheduler_Progetto/input_data.py), aggiungere un dizionario/struttura `SOFT_CONSTRAINTS` che formalizzi le tre sotto-categorie:

```python
SOFT_CONSTRAINTS = {
    "personal_preferences": {
        "descrizione": "Turni preferiti e indesiderati espressi dal lavoratore.",
        "campi": ["turni_preferiti", "turni_indesiderati"],
    },
    "shift_desirability": {
        "descrizione": "Peso numerico di soddisfazione per ogni turno (funzione obiettivo).",
        "campi": ["satisfaction_weights"],
    },
    "individual_tolerance": {
        "descrizione": "Livello di tolleranza ai turni indesiderati (0=rigido, 1=flessibile).",
        "campi": ["flexibility_score"],
    },
}
```

---

### Problema 3: I Hard Constraints non sono sotto-categorizzati

> [!NOTE]
> Problema **minore** ma vale la pena correggerlo. Il PDF elenca 4 sotto-categorie di Hard constraints:
> - Legal requirements
> - Minimum staffing levels
> - Maximum working hours
> - Mandatory rest periods
>
> Il dizionario `HARD_CONSTRAINTS` è piatto — non raggruppa per categoria.

**Cosa fare:**
- Ristrutturare `HARD_CONSTRAINTS` in [input_data.py](file:///c:/Users/Utente/Desktop/SmartScheduler_Progetto/input_data.py) con sotto-categorie, oppure aggiungere commenti/documentazione che mappino ogni vincolo alla sua categoria. Esempio:

```python
HARD_CONSTRAINTS = {
    # --- Legal requirements (requisiti legali) ---
    "turni_mensili_esatti": 25,
    "max_turni_per_giorno": 1,
    "turni_consecutivi_vietati": True,
    "notte_turno_doppio": True,
    # --- Maximum working hours ---
    "max_ore_settimanali": 36,
    # --- Mandatory rest periods ---
    "riposi_obbligatori_dopo_notte": 2,
    "giorni_riposo_minimi": 1,
}
# I Minimum staffing levels sono in STAFFING_CASE_A / STAFFING_CASE_B.
```

---

## 📊 Riepilogo

| Sotto-categoria | Stato | Priorità |
|---|---|---|
| **Hard → Legal requirements** | ✅ Funzionante, documentazione migliorabile | 🟢 Bassa |
| **Hard → Minimum staffing levels** | ✅ Funzionante (STAFFING_CASE_A/B) | 🟢 OK |
| **Hard → Maximum working hours** | ✅ Funzionante | 🟢 OK |
| **Hard → Mandatory rest periods** | ✅ Funzionante | 🟢 OK |
| **Soft → Personal preferences** | ✅ Funzionante (turni_preferiti/indesiderati) | 🟢 OK |
| **Soft → Shift desirability** | ✅ Funzionante (satisfaction_weights nella funzione obiettivo) | 🟢 OK |
| **Soft → Individual tolerance levels** | ⚠️ `flexibility_score` **non integrato** nel modello CP della Fase 2 | 🔴 Alta |
| **Struttura esplicita SOFT_CONSTRAINTS** | ❌ Mancante | 🟡 Media |
| **Sotto-categorie nei HARD_CONSTRAINTS** | ⚠️ Piatto, non categorizzato | 🟢 Bassa |

### Azioni correttive ordinate per priorità

1. 🔴 **Integrare `flexibility_score`** nel prompt del Drafting Agent (Fase 2) come modulatore nella funzione obiettivo
2. 🟡 **Aggiungere `SOFT_CONSTRAINTS`** in `input_data.py` come struttura esplicita parallela a `HARD_CONSTRAINTS`
3. 🟢 **Aggiungere commenti di sotto-categorizzazione** in `HARD_CONSTRAINTS` per allinearsi al PDF

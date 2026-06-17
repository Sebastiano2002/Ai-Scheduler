"""
models.py
=========
Fase 1 - Modelli Pydantic per l'output strutturato del Workers Agent.

Questo modulo definisce i modelli che descrivono e validano le preferenze
formalizzate dei lavoratori prodotte dall'LLM, con una distinzione ESPLICITA
tra Vincoli Hard e Vincoli Soft (requisito del progetto):

    - WorkerHardConstraints : vincoli HARD del singolo lavoratore (inderogabili);
    - WorkerSoftConstraints : vincoli SOFT (preferenze) + modello di soddisfazione;
    - WorkerPreference       : unisce hard + soft per un singolo lavoratore;
    - AllWorkerPreferences   : wrapper sul dizionario {worker_id: WorkerPreference}.

Distinzione Hard vs Soft a livello di singolo lavoratore:
    * HARD (devono SEMPRE valere):
        - turni_vietati          : turni assolutamente vietati (es. divieto medico).
    * SOFT (preferenze, guidano la funzione obiettivo):
        - turni_preferiti, turni_indesiderati, flexibility_score;
        - giorni_indesiderati    : richieste di ferie / giorni che il lavoratore
          preferirebbe NON lavorare (penalizzazione alta nell'obiettivo, non un
          divieto: in emergenza il solver puo' comunque assegnarli);
        - satisfaction_weights (il "modello di soddisfazione").

I validatori garantiscono che i dati generati dall'LLM rispettino le regole
del problema (codici turno ammessi, date ISO valide, flexibility_score nel
range [0, 1], coerenza tra preferenze e pesi di soddisfazione, separazione
tra vincoli hard e soft) prima di essere salvati e riutilizzati dalle fasi
successive (modello OR-Tools).
"""

from datetime import date
from typing import Dict, List

from pydantic import BaseModel, Field, field_validator, model_validator

# Codici turno ammessi dal problema (allineati a input_data.SHIFT_CODES).
ALLOWED_SHIFT_CODES = {"M", "P", "N"}


# ===========================================================================
# VINCOLI HARD DEL SINGOLO LAVORATORE
# ===========================================================================
class WorkerHardConstraints(BaseModel):
    """
    Vincoli HARD (inderogabili) specifici di un lavoratore, estratti dal
    linguaggio naturale. Devono sempre essere rispettati dalla schedulazione.

    Nota: i giorni di ferie/indisponibilita' NON sono qui: sono modellati come
    vincolo SOFT ad alta penalita' (vedi WorkerSoftConstraints.giorni_indesiderati),
    per evitare schedulazioni INFEASIBLE in caso di emergenza di organico.
    """

    turni_vietati: List[str] = Field(
        default_factory=list,
        description=(
            "Codici turno ASSOLUTAMENTE vietati per il lavoratore "
            "(es. divieto per motivi di salute)."
        ),
    )

    @field_validator("turni_vietati")
    @classmethod
    def _check_shift_codes(cls, value: List[str]) -> List[str]:
        """I codici turno vietati devono appartenere ai soli ammessi (M/P/N)."""
        invalidi = [c for c in value if c not in ALLOWED_SHIFT_CODES]
        if invalidi:
            raise ValueError(
                f"Codici turno non ammessi {invalidi}; ammessi: {sorted(ALLOWED_SHIFT_CODES)}."
            )
        return value


# ===========================================================================
# VINCOLI SOFT DEL SINGOLO LAVORATORE (PREFERENZE + SODDISFAZIONE)
# ===========================================================================
class WorkerSoftConstraints(BaseModel):
    """
    Vincoli SOFT (preferenze) di un lavoratore e relativo modello di
    soddisfazione. Non sono obbligatori: guidano la funzione obiettivo che
    massimizza la soddisfazione complessiva.
    """

    turni_preferiti: List[str] = Field(
        default_factory=list,
        description="Codici turno graditi, ordinati dal piu' gradito.",
    )
    turni_indesiderati: List[str] = Field(
        default_factory=list,
        description="Codici turno sgraditi (ma non vietati).",
    )
    giorni_indesiderati: List[str] = Field(
        default_factory=list,
        description=(
            "Date ISO 'YYYY-MM-DD' che il lavoratore preferirebbe NON lavorare "
            "(richieste di ferie). Non sono un divieto: subiscono una penalita' "
            "alta nella funzione obiettivo, ma in emergenza restano assegnabili."
        ),
    )
    flexibility_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description=(
            "Tolleranza ai turni indesiderati, tra 0.0 (rigido) e 1.0 (flessibile). "
            "E' la categoria soft 'Individual tolerance levels' del progetto. Per "
            "DESIGN non entra come parametro separato nel modello OR-Tools (sarebbe "
            "un doppio conteggio): la tolleranza e' gia' incorporata nella grandezza "
            "dei satisfaction_weights negativi, che la Fase 1 fissa in funzione di "
            "questo score (~ -6*(1-flexibility_score) per un turno indesiderato)."
        ),
    )
    satisfaction_weights: Dict[str, float] = Field(
        ...,
        description="Modello di soddisfazione: peso numerico per ciascun codice turno.",
    )

    @field_validator("turni_preferiti", "turni_indesiderati")
    @classmethod
    def _check_shift_codes(cls, value: List[str]) -> List[str]:
        """I codici turno devono appartenere ai soli ammessi (M/P/N)."""
        invalidi = [c for c in value if c not in ALLOWED_SHIFT_CODES]
        if invalidi:
            raise ValueError(
                f"Codici turno non ammessi {invalidi}; ammessi: {sorted(ALLOWED_SHIFT_CODES)}."
            )
        return value

    @field_validator("giorni_indesiderati")
    @classmethod
    def _check_iso_dates(cls, value: List[str]) -> List[str]:
        """I giorni indesiderati devono essere date ISO valide 'YYYY-MM-DD'."""
        for d in value:
            try:
                date.fromisoformat(d)
            except (ValueError, TypeError):
                raise ValueError(
                    f"Giorno indesiderato '{d}' non valido: usa il formato ISO 'YYYY-MM-DD'."
                )
        return value

    @field_validator("satisfaction_weights")
    @classmethod
    def _check_weight_keys(cls, value: Dict[str, float]) -> Dict[str, float]:
        """I pesi devono coprire tutti e soli i codici turno ammessi (M/P/N)."""
        chiavi = set(value.keys())
        if chiavi != ALLOWED_SHIFT_CODES:
            mancanti = ALLOWED_SHIFT_CODES - chiavi
            in_eccesso = chiavi - ALLOWED_SHIFT_CODES
            dettagli = []
            if mancanti:
                dettagli.append(f"mancanti: {sorted(mancanti)}")
            if in_eccesso:
                dettagli.append(f"non ammessi: {sorted(in_eccesso)}")
            raise ValueError(
                "satisfaction_weights deve contenere esattamente i codici "
                f"{sorted(ALLOWED_SHIFT_CODES)} ({'; '.join(dettagli)})."
            )
        return value

    @model_validator(mode="after")
    def _check_coherence(self) -> "WorkerSoftConstraints":
        """
        Coerenza tra preferenze e pesi:
            - un turno preferito deve avere peso positivo;
            - un turno indesiderato deve avere peso negativo.
        """
        for code in self.turni_preferiti:
            if self.satisfaction_weights.get(code, 0.0) <= 0:
                raise ValueError(
                    f"Il turno preferito '{code}' deve avere un satisfaction_weight "
                    f"positivo (trovato {self.satisfaction_weights.get(code)})."
                )
        for code in self.turni_indesiderati:
            if self.satisfaction_weights.get(code, 0.0) >= 0:
                raise ValueError(
                    f"Il turno indesiderato '{code}' deve avere un satisfaction_weight "
                    f"negativo (trovato {self.satisfaction_weights.get(code)})."
                )
        return self


# ===========================================================================
# PREFERENZE COMPLETE DI UN LAVORATORE (HARD + SOFT)
# ===========================================================================
class WorkerPreference(BaseModel):
    """
    Preferenze formalizzate di un singolo lavoratore, con distinzione esplicita
    tra vincoli hard (inderogabili) e soft (preferenze).
    """

    nome: str = Field(..., description="Nome completo del lavoratore.")
    hard_constraints: WorkerHardConstraints = Field(
        ..., description="Vincoli HARD specifici del lavoratore."
    )
    soft_constraints: WorkerSoftConstraints = Field(
        ..., description="Vincoli SOFT (preferenze) e modello di soddisfazione."
    )

    @model_validator(mode="after")
    def _check_hard_soft_separation(self) -> "WorkerPreference":
        """
        Un turno vietato (hard) non puo' comparire tra le preferenze soft:
        sarebbe una contraddizione (e' assolutamente escluso, non solo sgradito).
        """
        vietati = set(self.hard_constraints.turni_vietati)
        in_preferiti = vietati & set(self.soft_constraints.turni_preferiti)
        if in_preferiti:
            raise ValueError(
                f"I turni vietati {sorted(in_preferiti)} non possono essere anche "
                f"'turni_preferiti' (contraddizione hard/soft)."
            )
        in_indesiderati = vietati & set(self.soft_constraints.turni_indesiderati)
        if in_indesiderati:
            raise ValueError(
                f"I turni vietati {sorted(in_indesiderati)} sono gia' un vincolo HARD: "
                f"non vanno ripetuti tra i 'turni_indesiderati' soft."
            )
        return self


class AllWorkerPreferences(BaseModel):
    """
    Wrapper sull'intero dizionario delle preferenze: {worker_id: WorkerPreference}.

    Espone la chiave 'preferences' che mappa ogni ID lavoratore al relativo
    modello WorkerPreference validato.
    """

    preferences: Dict[str, WorkerPreference] = Field(
        ...,
        description="Mappa {worker_id: WorkerPreference} con le preferenze formalizzate.",
    )

    @field_validator("preferences")
    @classmethod
    def _non_vuoto(cls, value: Dict[str, WorkerPreference]) -> Dict[str, WorkerPreference]:
        """Deve esserci almeno un lavoratore."""
        if not value:
            raise ValueError("Il dizionario delle preferenze non puo' essere vuoto.")
        return value

    @classmethod
    def from_raw_dict(cls, raw: Dict[str, dict]) -> "AllWorkerPreferences":
        """
        Costruisce e valida il modello a partire dal dizionario grezzo
        WORKER_PREFERENCES prodotto dall'LLM ({worker_id: {...}}).
        """
        return cls(preferences=raw)

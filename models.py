"""
models.py
=========
Fase 1 - Modelli Pydantic per l'output strutturato del Workers Agent.

Questo modulo definisce i modelli che descrivono e validano le preferenze
formalizzate dei lavoratori prodotte dall'LLM:

    - WorkerPreference      : preferenze di un singolo lavoratore;
    - AllWorkerPreferences  : wrapper sul dizionario {worker_id: WorkerPreference}.

I validatori garantiscono che i dati generati dall'LLM rispettino le regole
del problema (codici turno ammessi, flexibility_score nel range [0, 1],
coerenza tra preferenze e pesi di soddisfazione) prima di essere salvati e
riutilizzati dalle fasi successive (modello OR-Tools).
"""

from typing import Dict, List

from pydantic import BaseModel, Field, field_validator, model_validator

# Codici turno ammessi dal problema (allineati a input_data.SHIFT_CODES).
ALLOWED_SHIFT_CODES = {"M", "P", "N"}


class WorkerPreference(BaseModel):
    """Preferenze formalizzate di un singolo lavoratore."""

    nome: str = Field(..., description="Nome completo del lavoratore.")
    turni_preferiti: List[str] = Field(
        default_factory=list,
        description="Codici turno graditi, ordinati dal piu' gradito.",
    )
    turni_indesiderati: List[str] = Field(
        default_factory=list,
        description="Codici turno sgraditi.",
    )
    giorni_indisponibilita: List[str] = Field(
        default_factory=list,
        description="Date ISO 'YYYY-MM-DD' in cui il lavoratore NON puo' lavorare.",
    )
    flexibility_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Tolleranza ai turni indesiderati, tra 0.0 (rigido) e 1.0 (flessibile).",
    )
    satisfaction_weights: Dict[str, float] = Field(
        ...,
        description="Peso numerico di soddisfazione per ciascun codice turno.",
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
    def _check_coherence(self) -> "WorkerPreference":
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

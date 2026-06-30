"""
models.py - Modelli Pydantic per le preferenze formalizzate dei lavoratori.

Definisce la struttura e la validazione delle preferenze prodotte dall'LLM,
con distinzione tra vincoli Hard (inderogabili) e Soft (preferenze).
"""

from datetime import date
from typing import Dict, List

from pydantic import BaseModel, Field, field_validator, model_validator

# Codici turno ammessi (M = Mattina, P = Pomeriggio, N = Notte).
ALLOWED_SHIFT_CODES = {"M", "P", "N"}


class WorkerHardConstraints(BaseModel):
    """Vincoli HARD (inderogabili) di un lavoratore."""

    turni_vietati: List[str] = Field(
        default_factory=list,
        description="Codici turno assolutamente vietati per il lavoratore.",
    )

    @field_validator("turni_vietati")
    @classmethod
    def _check_shift_codes(cls, value: List[str]) -> List[str]:
        """Verifica che i codici turno appartengano ai soli ammessi (M/P/N)."""
        invalidi = [c for c in value if c not in ALLOWED_SHIFT_CODES]
        if invalidi:
            raise ValueError(
                f"Codici turno non ammessi {invalidi}; ammessi: {sorted(ALLOWED_SHIFT_CODES)}."
            )
        return value


class WorkerSoftConstraints(BaseModel):
    """Vincoli SOFT (preferenze) di un lavoratore e modello di soddisfazione."""

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
        description="Date ISO 'YYYY-MM-DD' che il lavoratore preferirebbe non lavorare.",
    )
    flexibility_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Tolleranza ai turni indesiderati, tra 0.0 (rigido) e 1.0 (flessibile).",
    )
    satisfaction_weights: Dict[str, float] = Field(
        ...,
        description="Peso numerico per ciascun codice turno (modello di soddisfazione).",
    )

    @field_validator("turni_preferiti", "turni_indesiderati")
    @classmethod
    def _check_shift_codes(cls, value: List[str]) -> List[str]:
        """Verifica che i codici turno appartengano ai soli ammessi (M/P/N)."""
        invalidi = [c for c in value if c not in ALLOWED_SHIFT_CODES]
        if invalidi:
            raise ValueError(
                f"Codici turno non ammessi {invalidi}; ammessi: {sorted(ALLOWED_SHIFT_CODES)}."
            )
        return value

    @field_validator("giorni_indesiderati")
    @classmethod
    def _check_iso_dates(cls, value: List[str]) -> List[str]:
        """Verifica che le date siano in formato ISO valido 'YYYY-MM-DD'."""
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
        """Verifica che i pesi coprano tutti e soli i codici turno ammessi."""
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
        """Verifica coerenza: turno preferito -> peso positivo, indesiderato -> negativo."""
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


class WorkerPreference(BaseModel):
    """Preferenze complete di un lavoratore: vincoli hard + soft."""

    nome: str = Field(..., description="Nome completo del lavoratore.")
    hard_constraints: WorkerHardConstraints = Field(
        ..., description="Vincoli HARD del lavoratore."
    )
    soft_constraints: WorkerSoftConstraints = Field(
        ..., description="Vincoli SOFT e modello di soddisfazione."
    )

    @model_validator(mode="after")
    def _check_hard_soft_separation(self) -> "WorkerPreference":
        """Un turno vietato (hard) non puo' comparire tra le preferenze soft."""
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
    """Dizionario completo delle preferenze: {worker_id: WorkerPreference}."""

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
        """Costruisce e valida il modello dal dizionario grezzo prodotto dall'LLM."""
        return cls(preferences=raw)

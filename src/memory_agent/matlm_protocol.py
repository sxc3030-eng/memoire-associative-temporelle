"""Trames de controle du processus interactif MAT-LM.

Ces messages sont distincts du contrat de reponse natif. Ils permettent au
serveur de savoir que le modele et son adaptateur sont réellement charges sans
placer une fausse reponse dans la file des questions.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


INTERACTIVE_CONTROL_SCHEMA_VERSION = "matlm-interactive-control-v1"
INTERACTIVE_READY_EVENT = "ready"


def interactive_ready_frame() -> dict[str, Any]:
    """Construit la trame emise une fois le modele interactif charge."""

    return {
        "schema_version": INTERACTIVE_CONTROL_SCHEMA_VERSION,
        "event": INTERACTIVE_READY_EVENT,
        "ok": True,
    }


def is_interactive_ready_frame(value: Any) -> bool:
    """Reconnaît uniquement la trame de disponibilite exacte."""

    return (
        isinstance(value, Mapping)
        and set(value) == {"schema_version", "event", "ok"}
        and value.get("schema_version") == INTERACTIVE_CONTROL_SCHEMA_VERSION
        and value.get("event") == INTERACTIVE_READY_EVENT
        and value.get("ok") is True
    )

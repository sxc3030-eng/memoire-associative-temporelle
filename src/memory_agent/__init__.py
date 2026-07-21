"""Moteur local de memoire associative temporelle.

L'API publique tient volontairement dans une seule classe afin que le moteur
puisse etre utilise aussi bien depuis un terminal que depuis un agent.
"""

from .memory import MemoryEngine, MemoryIdempotencyConflictError
from .pipeline import (
    BackgroundConsolidator,
    DurableInjectionQueue,
    IdempotencyConflictError,
    MemoryPipeline,
    QueueStateError,
)

__all__ = [
    "BackgroundConsolidator",
    "DurableInjectionQueue",
    "IdempotencyConflictError",
    "MemoryEngine",
    "MemoryIdempotencyConflictError",
    "MemoryPipeline",
    "QueueStateError",
]
__version__ = "0.3.0"

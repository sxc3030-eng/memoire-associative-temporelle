"""Memoire associative, pipeline durable et calculateur local borne.

Les composants publics restent separables afin qu'un agent puisse calculer
sans ecrire dans la memoire, ou utiliser la memoire sans calculateur.
"""

from .memory import MemoryEngine, MemoryIdempotencyConflictError
from .math_engine import MathEngine, MathEngineError
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
    "MathEngine",
    "MathEngineError",
    "MemoryEngine",
    "MemoryIdempotencyConflictError",
    "MemoryPipeline",
    "QueueStateError",
]
__version__ = "0.4.0"

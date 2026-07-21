"""Memoire associative, pipeline durable et laboratoires locaux bornes.

Les composants publics restent separables afin qu'un agent puisse calculer
sans ecrire dans la memoire, ou utiliser la memoire sans calculateur.
"""

from .memory import MemoryEngine, MemoryIdempotencyConflictError
from .math_engine import MathEngine, MathEngineError
from .history_stress_lab import (
    HistoryStressConfig,
    generate_history_scenario,
    history_stress_catalog,
    run_history_stress,
)
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
    "HistoryStressConfig",
    "MathEngine",
    "MathEngineError",
    "MemoryEngine",
    "MemoryIdempotencyConflictError",
    "MemoryPipeline",
    "QueueStateError",
    "generate_history_scenario",
    "history_stress_catalog",
    "run_history_stress",
]
__version__ = "0.5.0"

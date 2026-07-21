"""Memoire associative, pipeline durable et laboratoires locaux bornes.

Les composants publics restent separables afin qu'un agent puisse calculer
sans ecrire dans la memoire, ou utiliser la memoire sans calculateur.
"""

from .memory import MemoryEngine, MemoryIdempotencyConflictError
from .math_engine import MathEngine, MathEngineError
from .memory_hub import (
    GeneratedObservationError,
    MemoryAccessError,
    MemoryHub,
    SpacePolicy,
)
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
from .science_curriculum import (
    ScienceCurriculumError,
    build_science_capsules,
    import_science_reference,
    load_science_dataset,
)

__all__ = [
    "BackgroundConsolidator",
    "DurableInjectionQueue",
    "IdempotencyConflictError",
    "HistoryStressConfig",
    "MathEngine",
    "MathEngineError",
    "GeneratedObservationError",
    "MemoryAccessError",
    "MemoryEngine",
    "MemoryHub",
    "MemoryIdempotencyConflictError",
    "MemoryPipeline",
    "QueueStateError",
    "ScienceCurriculumError",
    "SpacePolicy",
    "build_science_capsules",
    "generate_history_scenario",
    "history_stress_catalog",
    "import_science_reference",
    "load_science_dataset",
    "run_history_stress",
]
__version__ = "0.6.0"

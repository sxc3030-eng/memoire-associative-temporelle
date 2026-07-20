"""Moteur local de memoire associative temporelle.

L'API publique tient volontairement dans une seule classe afin que le moteur
puisse etre utilise aussi bien depuis un terminal que depuis un agent.
"""

from .memory import MemoryEngine

__all__ = ["MemoryEngine"]
__version__ = "0.1.0"

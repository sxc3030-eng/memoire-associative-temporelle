"""Hub multi-agent neutre au modèle, au-dessus de mémoires séparées."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Sequence

from .memory import MemoryEngine


_POLICIES = frozenset({"private", "shared", "reference"})
_WRITABLE_SOURCES = frozenset({"observed", "executed", "user_confirmed"})
_RESERVED_AGENT = "_memory_hub_agent_id"
_RESERVED_SPACE = "_memory_hub_space"


class MemoryAccessError(PermissionError):
    """L'agent n'a pas le droit demandé sur un espace."""


class GeneratedObservationError(PermissionError):
    """Une sortie générée ne peut pas devenir une observation via le hub."""


def _agent_id(value: Any) -> str:
    if not isinstance(value, str):
        raise TypeError("agent_id doit être une chaîne stable")
    clean = value.strip()
    if not clean or clean != value or len(clean) > 128 or any(ord(c) < 32 for c in clean):
        raise ValueError("agent_id doit être stable, non vide et limité à 128 caractères")
    return clean


def _principals(values: Iterable[str] | str) -> frozenset[str]:
    raw = [values] if isinstance(values, str) else list(values)
    return frozenset("*" if value == "*" else _agent_id(value) for value in raw)


@dataclass(frozen=True, slots=True)
class SpacePolicy:
    """Politique d'un espace privé, partagé ou de référence."""

    kind: str
    owner_agent_id: str | None = None
    readers: frozenset[str] = field(default_factory=frozenset)
    writers: frozenset[str] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        kind = str(self.kind).strip().casefold()
        if kind not in _POLICIES:
            raise ValueError("kind doit valoir private, shared ou reference")
        owner = None if self.owner_agent_id is None else _agent_id(self.owner_agent_id)
        readers = _principals(self.readers)
        writers = _principals(self.writers)
        if kind == "private" and owner is None:
            raise ValueError("un espace private exige owner_agent_id")
        if kind == "reference" and writers:
            raise ValueError("un espace reference est toujours en lecture seule")
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "owner_agent_id", owner)
        object.__setattr__(self, "readers", readers)
        object.__setattr__(self, "writers", writers)

    @classmethod
    def private(cls, owner_agent_id: str) -> "SpacePolicy":
        return cls("private", owner_agent_id=owner_agent_id)

    @classmethod
    def shared(
        cls,
        *,
        readers: Iterable[str] = (),
        writers: Iterable[str] = (),
        owner_agent_id: str | None = None,
    ) -> "SpacePolicy":
        return cls("shared", owner_agent_id, frozenset(readers), frozenset(writers))

    @classmethod
    def reference(cls, *, readers: Iterable[str] = ("*",)) -> "SpacePolicy":
        return cls("reference", readers=frozenset(readers))

    def can_read(self, agent_id: str) -> bool:
        if self.kind == "private":
            return agent_id == self.owner_agent_id
        if self.kind == "reference" and not self.readers:
            return True
        return agent_id == self.owner_agent_id or agent_id in self.readers or "*" in self.readers

    def can_write(self, agent_id: str) -> bool:
        if self.kind == "private":
            return agent_id == self.owner_agent_id
        if self.kind == "reference":
            return False
        return agent_id == self.owner_agent_id or agent_id in self.writers or "*" in self.writers


def _json_copy(value: Any) -> Any:
    try:
        return json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True))
    except (TypeError, ValueError) as error:
        raise TypeError("la capsule exige des données JSON neutres") from error


def _encode(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _semantic_context(value: Any) -> Any:
    if not isinstance(value, Mapping):
        return value
    return {
        key: _semantic_context(item)
        for key, item in value.items()
        if key not in {_RESERVED_AGENT, _RESERVED_SPACE}
    }


def _fingerprint(result: Mapping[str, Any]) -> str:
    events = result.get("events", [])
    stable_events = []
    if isinstance(events, Sequence) and not isinstance(events, (str, bytes)):
        for event in events:
            if isinstance(event, Mapping):
                stable_events.append(
                    {
                        "text": event.get("text"),
                        "source": event.get("source"),
                        "context": _semantic_context(event.get("context", {})),
                    }
                )
    stable = {
        "text": result.get("text"),
        "context": _semantic_context(result.get("context", {})),
        "events": stable_events,
    }
    return hashlib.sha256(_encode(stable).encode("utf-8")).hexdigest()


class MemoryHub:
    """Applique les droits puis compose des capsules à partir de plusieurs espaces."""

    def __init__(
        self,
        spaces: Mapping[str, MemoryEngine],
        policies: Mapping[str, SpacePolicy],
    ) -> None:
        if not isinstance(spaces, Mapping) or not spaces:
            raise ValueError("spaces doit contenir au moins un MemoryEngine")
        clean_spaces: dict[str, MemoryEngine] = {}
        identities: set[int] = set()
        disk_paths: set[str] = set()
        for name, engine in spaces.items():
            if not isinstance(name, str) or not name.strip() or name != name.strip():
                raise ValueError("chaque nom d'espace doit être une chaîne stable")
            if not isinstance(engine, MemoryEngine):
                raise TypeError("chaque espace doit contenir un MemoryEngine")
            if id(engine) in identities:
                raise ValueError("deux espaces ne peuvent pas partager le même MemoryEngine")
            identities.add(id(engine))
            if engine.db_path != ":memory:":
                path = os.path.normcase(str(Path(engine.db_path).expanduser().resolve()))
                if path in disk_paths:
                    raise ValueError("deux espaces ne peuvent pas partager le même fichier SQLite")
                disk_paths.add(path)
            clean_spaces[name] = engine
        if set(clean_spaces) != set(policies):
            raise ValueError("une politique est requise pour chaque espace, sans entrée supplémentaire")
        if not all(isinstance(policy, SpacePolicy) for policy in policies.values()):
            raise TypeError("les politiques doivent être des SpacePolicy")
        self.spaces = MappingProxyType(dict(sorted(clean_spaces.items())))
        self.policies = MappingProxyType(
            {name: policies[name] for name in sorted(clean_spaces)}
        )

    def _space(self, name: str) -> tuple[MemoryEngine, SpacePolicy]:
        if name not in self.spaces:
            raise KeyError(f"espace inconnu: {name}")
        return self.spaces[name], self.policies[name]

    def observe(
        self,
        agent_id: str,
        space_name: str,
        text: str,
        *,
        episode_id: str | None = None,
        context: Mapping[str, Any] | None = None,
        source: str | Mapping[str, Any] = "observed",
        idempotency_key: str | None = None,
    ) -> dict[str, Any]:
        agent = _agent_id(agent_id)
        engine, policy = self._space(space_name)
        if not policy.can_write(agent):
            raise MemoryAccessError(f"{agent} ne peut pas écrire dans {space_name}")
        source_data = {"type": source} if isinstance(source, str) else dict(_json_copy(source))
        source_type = str(source_data.get("type", "")).strip().casefold()
        generated_signal = source_data.get("generated") is True or any(
            str(source_data.get(key, "")).strip().casefold()
            in {"generated", "model_generated", "assistant_generated", "llm_generated"}
            for key in ("type", "origin", "kind")
        )
        if generated_signal or source_type not in _WRITABLE_SOURCES:
            raise GeneratedObservationError(
                "seules les observations externes confirmées peuvent être écrites"
            )
        source_data.update({"type": source_type, "agent_id": agent, "space": space_name})
        clean_context = dict(_json_copy(context or {}))
        clean_context[_RESERVED_AGENT] = agent
        clean_context[_RESERVED_SPACE] = space_name
        stable_key = None
        if idempotency_key is not None:
            raw = f"{agent}\0{space_name}\0{idempotency_key}"
            stable_key = "memory-hub:" + hashlib.sha256(raw.encode("utf-8")).hexdigest()
        return engine.observe(
            text,
            episode_id=episode_id,
            context=clean_context,
            source=source_data,
            idempotency_key=stable_key,
        )

    def recall_capsule(
        self,
        agent_id: str,
        query: str | Mapping[str, Any],
        *,
        space_names: Sequence[str] | None = None,
        top_k: int = 5,
        character_budget: int = 16_384,
    ) -> dict[str, Any]:
        agent = _agent_id(agent_id)
        if isinstance(character_budget, bool) or not isinstance(character_budget, int):
            raise TypeError("character_budget doit être un entier")
        if not 512 <= character_budget <= 1_000_000:
            raise ValueError("character_budget doit être compris entre 512 et 1 000 000")
        if space_names is None:
            selected_spaces = [
                name for name in self.spaces if self.policies[name].can_read(agent)
            ]
        else:
            if isinstance(space_names, (str, bytes)):
                raise TypeError("space_names doit être une séquence de noms")
            selected_spaces = sorted(set(space_names))
            for name in selected_spaces:
                _, policy = self._space(name)
                if not policy.can_read(agent):
                    raise MemoryAccessError(f"{agent} ne peut pas lire {name}")

        grouped: dict[str, dict[str, Any]] = {}
        candidate_count = 0
        for name in selected_spaces:
            engine, policy = self._space(name)
            for rank, raw_result in enumerate(engine.recall(query, top_k=top_k)):
                candidate_count += 1
                result = _json_copy(raw_result)
                key = _fingerprint(result)
                origin = {
                    "space": name,
                    "policy": policy.kind,
                    "episode_id": result.get("episode_id"),
                    "rank": rank,
                }
                if key in grouped:
                    grouped[key]["origins"].append(origin)
                    continue
                item = dict(result)
                item.update(
                    {
                        "space": name,
                        "space_policy": policy.kind,
                        "origins": [origin],
                        "deduplication_key": key,
                    }
                )
                grouped[key] = item

        candidates = list(grouped.values())
        for item in candidates:
            item["origins"].sort(key=lambda value: (value["space"], value["rank"]))
        candidates.sort(
            key=lambda item: (
                -float(item.get("score", 0.0)),
                item["deduplication_key"],
            )
        )
        query_hash = hashlib.sha256(_encode(_json_copy(query)).encode("utf-8")).hexdigest()

        def capsule(items: list[dict[str, Any]]) -> dict[str, Any]:
            value = {
                "schema_version": "memory-hub-capsule-v1",
                "agent_id": agent,
                "query_sha256": query_hash,
                "items": items,
                "retrieval": {
                    "spaces_consulted": len(selected_spaces),
                    "candidates": candidate_count,
                    "duplicates_removed": candidate_count - len(candidates),
                    "returned": len(items),
                },
                "budget": {
                    "character_limit": character_budget,
                    "characters_used": 0,
                    "truncated": len(items) < len(candidates),
                    "measurement": "compact_json_characters",
                },
            }
            for _ in range(4):
                used = len(_encode(value))
                if value["budget"]["characters_used"] == used:
                    break
                value["budget"]["characters_used"] = used
            return value

        kept: list[dict[str, Any]] = []
        for item in candidates:
            proposed = capsule([*kept, item])
            if len(_encode(proposed)) <= character_budget:
                kept.append(item)
        result = capsule(kept)
        if len(_encode(result)) > character_budget:
            raise ValueError("character_budget trop petit pour l'enveloppe JSON")
        return result

    build_capsule = recall_capsule

    @staticmethod
    def capsule_json(capsule: Mapping[str, Any]) -> str:
        return _encode(capsule)


__all__ = [
    "GeneratedObservationError",
    "MemoryAccessError",
    "MemoryHub",
    "SpacePolicy",
]

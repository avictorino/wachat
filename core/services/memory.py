from __future__ import annotations

from typing import Iterable

from django.db.models import Q

from core.models import FriendMemory, VirtualFriend

DEFAULT_MEMORY_KEYS = [
    "favorite_verse",
    "main_struggle",
    "prayer_style",
    "family_context",
]


def get_relevant_memories(
    friend: VirtualFriend, keys: Iterable[str] = DEFAULT_MEMORY_KEYS
) -> list[FriendMemory]:
    qs = (
        FriendMemory.objects.filter(friend=friend, is_active=True)
        .filter(Q(key__in=list(keys)) | Q(kind__in=["prayer", "verse"]))
        .order_by("-updated_at")[:25]
    )
    return list(qs)


def upsert_memory(
    friend: VirtualFriend,
    *,
    kind: str,
    key: str,
    value: str,
    confidence: float = 0.80,
    source: dict | None = None,
) -> FriendMemory:
    obj, _ = FriendMemory.objects.update_or_create(
        friend=friend,
        kind=kind,
        key=key,
        defaults={
            "value": value,
            "confidence": confidence,
            "source": source or {},
            "is_active": True,
        },
    )
    return obj

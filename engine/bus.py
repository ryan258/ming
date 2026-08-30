from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from .contracts import Event


class EventBus:
    """Small fan-out event bus for run observers."""

    def __init__(self) -> None:
        self._subscribers: set[asyncio.Queue[Event]] = set()
        self._history: list[Event] = []
        self._lock = asyncio.Lock()

    @property
    def history(self) -> tuple[Event, ...]:
        return tuple(self._history)

    async def publish(self, event: Event) -> None:
        async with self._lock:
            self._history.append(event)
            subscribers = tuple(self._subscribers)
        for queue in subscribers:
            try:
                queue.put_nowait(event)
            except asyncio.QueueFull:
                continue

    @asynccontextmanager
    async def subscribe(self, maxsize: int = 100) -> AsyncIterator[asyncio.Queue[Event]]:
        queue: asyncio.Queue[Event] = asyncio.Queue(maxsize=maxsize)
        async with self._lock:
            self._subscribers.add(queue)
        try:
            yield queue
        finally:
            async with self._lock:
                self._subscribers.discard(queue)

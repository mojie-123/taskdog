"""Small keyed executor for room-level SysNav VLM requests.

Workers only evaluate immutable request snapshots.  The search runner remains
the sole owner of SceneMemory and applies completed results on its own thread.
"""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class AsyncVLMResult:
    kind: str
    key: tuple[Any, ...]
    context: Any
    value: Any = None
    error: str = ""


@dataclass(frozen=True)
class _AsyncVLMTask:
    kind: str
    key: tuple[Any, ...]
    context: Any


class AsyncRoomVLMManager:
    """Run at most one room-level VLM request for each semantic key."""

    def __init__(self, *, max_workers: int = 3):
        if int(max_workers) <= 0:
            raise ValueError("max_workers must be positive")
        self._executor = ThreadPoolExecutor(
            max_workers=int(max_workers),
            thread_name_prefix="sysnav-room-vlm",
        )
        self._futures: dict[Future, _AsyncVLMTask] = {}
        self._active_keys: set[tuple[Any, ...]] = set()
        self._closed = False

    @property
    def has_pending(self) -> bool:
        return bool(self._futures)

    def is_active(self, key: tuple[Any, ...]) -> bool:
        return tuple(key) in self._active_keys

    def submit(
        self,
        kind: str,
        key: tuple[Any, ...],
        context: Any,
        function: Callable[..., Any],
        /,
        *args,
        **kwargs,
    ) -> bool:
        if self._closed:
            return False
        normalized_key = tuple(key)
        if normalized_key in self._active_keys:
            return False
        future = self._executor.submit(function, *args, **kwargs)
        self._futures[future] = _AsyncVLMTask(
            kind=str(kind), key=normalized_key, context=context
        )
        self._active_keys.add(normalized_key)
        return True

    def _collect(self, futures: tuple[Future, ...]) -> list[AsyncVLMResult]:
        completed: list[AsyncVLMResult] = []
        for future in futures:
            task = self._futures.pop(future)
            self._active_keys.discard(task.key)
            try:
                value = future.result()
            except Exception as exc:
                completed.append(
                    AsyncVLMResult(
                        kind=task.kind,
                        key=task.key,
                        context=task.context,
                        error=f"{type(exc).__name__}: {exc}",
                    )
                )
            else:
                completed.append(
                    AsyncVLMResult(
                        kind=task.kind,
                        key=task.key,
                        context=task.context,
                        value=value,
                    )
                )
        return completed

    def poll(self) -> list[AsyncVLMResult]:
        return self._collect(
            tuple(future for future in self._futures if future.done())
        )

    def wait(self) -> list[AsyncVLMResult]:
        return self._collect(tuple(self._futures))

    def close(self, *, wait: bool = True) -> None:
        if self._closed:
            return
        self._closed = True
        self._executor.shutdown(wait=bool(wait), cancel_futures=True)
        self._futures.clear()
        self._active_keys.clear()

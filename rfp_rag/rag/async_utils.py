"""Jupyter 호환 비동기 실행기.

Jupyter 커널에는 이미 이벤트 루프가 돌고 있어 asyncio.run() 이 'cannot be called from a running event loop'
로 실패한다. 또한 호출마다 asyncio.run() 으로 새 루프를 만들면 LangChain/OpenAI 의 httpx.AsyncClient 가
닫힌 루프에 묶여 'Event loop is closed' 가 간헐적으로 난다.
→ 전용 백그라운드 스레드에 '영구' 이벤트 루프 하나를 두고 모든 코루틴을 거기로 보낸다 (CLI/Jupyter/Gradio 공통).
"""
from __future__ import annotations

import asyncio
import threading

_loop: asyncio.AbstractEventLoop | None = None
_lock = threading.Lock()


def _get_loop() -> asyncio.AbstractEventLoop:
    global _loop
    with _lock:
        if _loop is None or _loop.is_closed():
            _loop = asyncio.new_event_loop()
            threading.Thread(target=_loop.run_forever, name="rag-async-loop", daemon=True).start()
        return _loop


def run_coro(coro, timeout: float | None = None):
    """어떤 스레드/환경에서든 코루틴을 실행하고 결과를 동기적으로 반환"""
    return asyncio.run_coroutine_threadsafe(coro, _get_loop()).result(timeout)

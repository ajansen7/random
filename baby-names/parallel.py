"""Run one System One request per record, many at a time.

Both stages of the funnel have the same shape: one independent request per
name, with nothing shared between them. At corpus scale that is 3,726
requests each way, and the work is almost entirely waiting on the network -
sequentially it takes about twenty minutes a stage, concurrently about one.

The concurrency limit is the only knob. Requests are cheap and short (a few
hundred output tokens, a fraction of a second), so the ceiling is the API's
rate limit rather than anything local; TypeSafeRateLimitError is left to the
SDK's own RetryPolicy, which honours Retry-After.

Results come back in the input order regardless of completion order, so a
rerun of the same corpus produces the same file.
"""

from __future__ import annotations

import asyncio
import sys
from collections.abc import Callable, Sequence
from typing import Any

from typesafe_sdk import (
    AsyncTypeSafeClient,
    RetryPolicy,
    TypeSafeAPIConnectionError,
    TypeSafeAPIError,
    TypeSafeError,
)

API_ERRORS = (TypeSafeAPIConnectionError, TypeSafeAPIError, TypeSafeError)


def run_all(
    records: Sequence[dict],
    state_for: Callable[[dict], Any],
    questions_for: Callable[[dict], dict],
    *,
    concurrency: int = 16,
    model: str | None = None,
    label: Callable[[dict], str] = lambda r: r.get("formal", "?"),
    progress_every: int = 100,
) -> tuple[list[tuple[dict, Any]], list[dict]]:
    """Ask one question set per record. Returns (answered, failures).

    `answered` pairs each record with its response, in input order. A record
    whose request failed after retries appears in `failures` instead, so a
    rerun can fill in the gaps rather than starting over.
    """
    return asyncio.run(
        _run_all(
            records,
            state_for,
            questions_for,
            concurrency=concurrency,
            model=model,
            label=label,
            progress_every=progress_every,
        )
    )


async def _run_all(
    records: Sequence[dict],
    state_for: Callable[[dict], Any],
    questions_for: Callable[[dict], dict],
    *,
    concurrency: int,
    model: str | None,
    label: Callable[[dict], str],
    progress_every: int,
) -> tuple[list[tuple[dict, Any]], list[dict]]:
    results: list[Any] = [None] * len(records)
    failures: list[dict] = []
    gate = asyncio.Semaphore(concurrency)
    done = 0

    retry = RetryPolicy(max_retries=4, backoff_initial=0.5, backoff_max=8.0)

    async with AsyncTypeSafeClient(model=model, retry=retry) as client:

        async def one(i: int, rec: dict) -> None:
            nonlocal done
            async with gate:
                try:
                    results[i] = await client.system_one(
                        state=state_for(rec), questions=questions_for(rec)
                    )
                except API_ERRORS as exc:
                    failures.append(
                        {"id": rec.get("id"), "error": f"{type(exc).__name__}: {exc}"}
                    )
                    print(f"  {label(rec)}: {exc}", file=sys.stderr)
                done += 1
                if progress_every and done % progress_every == 0:
                    print(f"  [{done}/{len(records)}]", flush=True)

        await asyncio.gather(*(one(i, r) for i, r in enumerate(records)))

    answered = [(rec, res) for rec, res in zip(records, results) if res is not None]
    return answered, failures

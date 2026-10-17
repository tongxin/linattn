"""Synthetic agentic traces: shared prompt, tool outputs, idle gaps.

Real serving traces are rarely published, and the shape that matters for the KV
layer is simple enough to generate honestly: a long shared prefix, turns that append
tool output and a short generation, and *gaps* between turns. The gaps are the point
-- a prefix cache survives them only if the idle time is below the provider's TTL,
which is exactly the mechanism behind the "17.5% of recompute comes from requests
arriving after the TTL" measurement in the public agentic-cache study.

The generator is deliberately parameterised by the things the cost model consumes
(block counts, gaps, TTL) rather than by a particular product.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Optional, Sequence

__all__ = ["Request", "agentic_trace", "trace_summary"]


@dataclass(frozen=True)
class Request:
    """One model call in a session."""

    at: float                       # arrival time, seconds
    blocks: Sequence[int]           # prefix-inclusive block ids, in order
    session: int = 0
    kind: str = "turn"              # system | turn | tool | retry

    @property
    def n_blocks(self) -> int:
        return len(self.blocks)


def agentic_trace(
    *,
    n_sessions: int = 8,
    turns_per_session: int = 6,
    shared_blocks: int = 32,
    tool_blocks: int = 8,
    answer_blocks: int = 2,
    gap_mean_s: float = 60.0,
    reset_prob: float = 0.08,
    seed: int = 0,
) -> List[Request]:
    """Build a trace of agentic sessions.

    Each session starts with a shared system prefix (``shared_blocks``, identical
    across sessions -- the prefix cache's bread and butter), then alternates
    tool-output blocks with a short generated answer. With probability
    ``reset_prob`` a session is abandoned and a new one starts (no reuse across
    them beyond the system prefix).
    """
    rng = random.Random(seed)
    requests: List[Request] = []
    t = 0.0
    # one global id space for shared and per-session blocks
    next_id = 1
    shared = list(range(1, shared_blocks + 1))
    next_id = shared_blocks + 1
    for session in range(n_sessions):
        session_blocks = list(shared)                  # starts as the shared prefix
        for _turn in range(turns_per_session):
            t += rng.expovariate(1.0 / gap_mean_s)
            if rng.random() < reset_prob:
                break
            tool = list(range(next_id, next_id + tool_blocks))
            next_id += tool_blocks
            requests.append(Request(at=t, blocks=tuple(session_blocks), session=session, kind="turn"))
            session_blocks = session_blocks + tool
            answer = list(range(next_id, next_id + answer_blocks))
            next_id += answer_blocks
            session_blocks = session_blocks + answer
    return requests


def trace_summary(requests: Sequence[Request]) -> dict:
    """Cheap facts about a trace, useful in report headers."""
    if not requests:
        return {"requests": 0}
    t0 = min(r.at for r in requests)
    t1 = max(r.at for r in requests)
    blocks = [r.n_blocks for r in requests]
    return {
        "requests": len(requests),
        "span_s": t1 - t0,
        "mean_blocks": sum(blocks) / len(blocks),
        "max_blocks": max(blocks),
        "sessions": len({r.session for r in requests}),
    }

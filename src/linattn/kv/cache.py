"""Block-granular prefix cache with eviction policies, and a trace simulator.

This is the part of the KV layer that decides the **hit rate**, and the hit rate is
what turns "recompute the prompt" into "read the prefix back". On agentic workloads
that one number can outweigh every attention-kernel improvement, which is why the
cost model downstream takes it as an explicit parameter and this simulator exists to
ground it in a realistic trace.

Model
-----

* the cache holds *blocks* (16 tokens by default), keyed by a content hash that
  already includes the whole prefix, so equal ids mean equal content at equal
  position (the radix-tree invariant, flattened into a set);
* ``lookup`` walks a request's blocks and returns how many leading blocks are
  cached, touching them (LRU order and TTL refresh, except under ``fifo``);
* ``insert`` adds the blocks the request had to compute;
* capacity is in blocks; policies are ``lru``, ``ttl_lru`` (expire after ``ttl_s``
  idle), ``fifo`` and ``random``. ``oracle_hits`` gives the block-level Belady
  upper bound for a fully known trace.
"""

from __future__ import annotations

import math
import random
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

__all__ = ["CacheConfig", "BlockCache", "SimResult", "simulate", "oracle_hits"]


@dataclass
class CacheConfig:
    capacity_blocks: int = 4096
    block_size: int = 16
    policy: str = "lru"          # lru | ttl_lru | fifo | random
    ttl_s: Optional[float] = None
    seed: int = 0

    def __post_init__(self) -> None:
        if self.policy not in ("lru", "ttl_lru", "fifo", "random"):
            raise ValueError(f"unknown policy {self.policy!r}")
        if self.capacity_blocks < 1:
            raise ValueError("capacity_blocks must be >= 1")

    @property
    def capacity_tokens(self) -> int:
        return self.capacity_blocks * self.block_size


class BlockCache:
    """Prefix cache keyed by prefix-inclusive block ids."""

    def __init__(self, cfg: CacheConfig):
        self.cfg = cfg
        self._blocks: "OrderedDict[int, float]" = OrderedDict()   # id -> last touch
        self._rng = random.Random(cfg.seed)
        self.evictions = 0
        self.expirations = 0
        self.hits = 0
        self.misses = 0

    # -- internals ---------------------------------------------------------------
    def _expire(self, now: float) -> None:
        if self.cfg.ttl_s is None:
            return
        stale = [b for b, t in self._blocks.items() if now - t > self.cfg.ttl_s]
        for b in stale:
            del self._blocks[b]
        self.expirations += len(stale)

    def _evict_one(self) -> None:
        if not self._blocks:
            return
        if self.cfg.policy == "random":
            victim = self._rng.choice(list(self._blocks))
            del self._blocks[victim]
        else:                                    # lru / ttl_lru / fifo: oldest first
            self._blocks.popitem(last=False)
        self.evictions += 1

    # -- public API --------------------------------------------------------------
    def __len__(self) -> int:
        return len(self._blocks)

    def __contains__(self, block: int) -> bool:
        return block in self._blocks

    def lookup(self, blocks: Sequence[int], now: float = 0.0) -> int:
        """Number of leading blocks already cached; touches them."""
        self._expire(now)
        matched = 0
        refresh = self.cfg.policy != "fifo"      # FIFO never reorders on a hit
        for b in blocks:
            if b not in self._blocks:
                break
            matched += 1
            self._blocks[b] = now
            if refresh:
                self._blocks.move_to_end(b)
        self.hits += matched
        self.misses += len(blocks) - matched
        return matched

    def insert(self, blocks: Sequence[int], now: float = 0.0) -> None:
        for b in blocks:
            if b in self._blocks:
                self._blocks[b] = now
                if self.cfg.policy != "fifo":
                    self._blocks.move_to_end(b)
                continue
            while len(self._blocks) >= self.cfg.capacity_blocks:
                self._evict_one()
            self._blocks[b] = now

    def stats(self) -> Dict[str, float]:
        total = self.hits + self.misses
        return {
            "hit_blocks": float(self.hits),
            "miss_blocks": float(self.misses),
            "hit_rate": self.hits / total if total else 0.0,
            "evictions": float(self.evictions),
            "expirations": float(self.expirations),
            "occupancy_blocks": float(len(self._blocks)),
            "occupancy_tokens": float(len(self._blocks) * self.cfg.block_size),
        }


@dataclass
class SimResult:
    policy: str
    capacity_blocks: int
    ttl_s: Optional[float]
    requests: int
    total_blocks: int
    hit_blocks: int
    miss_blocks: int
    evictions: int
    expirations: int
    hit_rate: float
    recompute_tokens: int
    peak_occupancy_tokens: int
    extra: Dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, object]:
        out = dict(self.__dict__)
        out["extra"] = dict(self.extra)
        return out


def simulate(requests: Iterable, cfg: CacheConfig) -> SimResult:
    """Run a trace (objects with ``blocks`` and ``at``) through the cache."""
    cache = BlockCache(cfg)
    total = hits = misses = 0
    peak = 0
    n_requests = 0
    for req in requests:
        n_requests += 1
        blocks = tuple(req.blocks)
        matched = cache.lookup(blocks, now=req.at)
        new_blocks = blocks[matched:]
        cache.insert(new_blocks, now=req.at)
        total += len(blocks)
        hits += matched
        misses += len(new_blocks)
        peak = max(peak, len(cache) * cfg.block_size)
    hit_rate = hits / total if total else 0.0
    return SimResult(
        policy=cfg.policy,
        capacity_blocks=cfg.capacity_blocks,
        ttl_s=cfg.ttl_s,
        requests=n_requests,
        total_blocks=total,
        hit_blocks=hits,
        miss_blocks=misses,
        evictions=cache.evictions,
        expirations=cache.expirations,
        hit_rate=hit_rate,
        recompute_tokens=misses * cfg.block_size,
        peak_occupancy_tokens=peak,
    )


def oracle_hits(requests: Sequence, capacity_blocks: int) -> Tuple[int, int]:
    """Block-level Belady upper bound over a fully known trace: ``(hits, total)``.

    Belady is defined per *block*; a real prefix cache also has to keep the chain
    intact, so this is an optimistic bound, which is the right direction for an
    upper bound.
    """
    future: Dict[int, List[int]] = {}
    for i, req in enumerate(requests):
        for b in req.blocks:
            future.setdefault(b, []).append(i)
    nxt: Dict[int, int] = {}
    cache: Dict[int, None] = {}
    hits = total = 0
    for i, req in enumerate(requests):
        blocks = list(dict.fromkeys(req.blocks))
        total += len(blocks)
        for b in blocks:
            uses = future[b]
            while uses and uses[0] <= i:
                uses.pop(0)
            nxt[b] = uses[0] if uses else math.inf
            if b in cache:
                hits += 1
                continue
            if len(cache) >= capacity_blocks:
                victim = max(cache, key=lambda k: nxt[k])
                del cache[victim]
            cache[b] = None
    return hits, total

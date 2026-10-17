"""The KV storage and data-management layer.

Attention research asks *how many* KV entries a kernel reads. Serving asks a larger
question: **which bytes are in which tier, and how much work does it take to get
them there.** The two are coupled:

- a compressed format (MLA latent, FP4, CSA groups) lowers bytes/token, which
  changes attention time *and* the capacity of every cache tier;
- a prefix/radix cache converts "recompute the prompt" into "read it back", and on
  agentic workloads that recompute term dominates everything else;
- the eviction/admission policy decides the hit rate, which is the one parameter
  that binds storage, transfer and compute together.

This package is the instrument kit:

``format``  bytes-per-token per representation (dense fp16/fp8, MLA, CSA2, and
            fixed-size recurrent state, which is independent of sequence length)
``cache``   block-granular prefix cache with LRU / TTL-LRU / FIFO / random
            eviction, a Belady oracle, and a trace simulator
``trace``   synthetic agentic sessions (shared prompt, tool outputs, idle gaps)
``cost``    roofline cost model with a per-component breakdown and the dominant
            term surfaced

The goal is not to predict one deployment; it is to say *which term dominates* for
given (context, format, hit rate, tier bandwidth) so that an algorithmic change is
aimed at the term that matters.
"""

from linattn.kv.cache import BlockCache, CacheConfig, SimResult, oracle_hits, simulate
from linattn.kv.cost import CostBreakdown, Hardware, ModelShape, Workload, breakdown, sweep_hit_rate
from linattn.kv.format import PRESETS, KVFormat, format_by_name
from linattn.kv.trace import Request, agentic_trace, trace_summary

__all__ = [
    "BlockCache",
    "CacheConfig",
    "SimResult",
    "oracle_hits",
    "simulate",
    "CostBreakdown",
    "Hardware",
    "ModelShape",
    "Workload",
    "breakdown",
    "sweep_hit_rate",
    "PRESETS",
    "KVFormat",
    "format_by_name",
    "Request",
    "agentic_trace",
    "trace_summary",
]

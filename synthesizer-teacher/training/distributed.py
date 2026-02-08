"""Distributed Data Parallel (DDP) utilities.

Single source of truth for all distributed training state.  When launched
via ``torchrun``, the standard ``RANK``/``LOCAL_RANK``/``WORLD_SIZE`` env
vars are present and we initialize NCCL.  In plain ``python`` mode every
helper falls back to single-GPU defaults (rank 0, world size 1).
"""

from __future__ import annotations

import os

import torch
import torch.distributed as dist


def is_torchrun() -> bool:
    """Return True when the process was launched by ``torchrun``."""
    return "RANK" in os.environ and "WORLD_SIZE" in os.environ


def setup_distributed() -> bool:
    """Initialize the NCCL process group if running under ``torchrun``.

    Returns True if DDP was initialized, False otherwise.
    """
    if not is_torchrun():
        return False

    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return True


def cleanup_distributed() -> None:
    """Tear down the process group (no-op when not initialized)."""
    if dist.is_initialized():
        dist.destroy_process_group()


def get_rank() -> int:
    return int(os.environ.get("RANK", 0))


def get_local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", 0))


def get_world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", 1))


def is_main_process() -> bool:
    return get_rank() == 0


def barrier() -> None:
    """Synchronize all processes (no-op in single-GPU mode)."""
    if dist.is_initialized():
        dist.barrier()

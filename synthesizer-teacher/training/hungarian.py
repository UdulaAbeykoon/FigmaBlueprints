"""Hungarian matching for DETR-style set prediction.

Computes cost matrices between predicted and ground truth modulation
connections, then solves the optimal assignment using the Hungarian
algorithm (scipy.optimize.linear_sum_assignment).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


def compute_cost_matrix(
    pred_source_logits: torch.Tensor,
    pred_dest_logits: torch.Tensor,
    pred_amount: torch.Tensor,
    gt_source_idx: torch.Tensor,
    gt_dest_idx: torch.Tensor,
    gt_amount: torch.Tensor,
    w_source: float = 1.0,
    w_dest: float = 1.0,
    w_amount: float = 1.0,
) -> torch.Tensor:
    """Compute cost matrix for Hungarian matching.

    For each (pred_query, gt_connection) pair, the cost is:
        w_source * CE(pred_source_logits, gt_source_idx)
        + w_dest * CE(pred_dest_logits, gt_dest_idx)
        + w_amount * |pred_amount - gt_amount|

    Args:
        pred_source_logits: (Q, n_sources) source classification logits.
        pred_dest_logits: (Q, n_dest) destination classification logits.
        pred_amount: (Q,) predicted amount values.
        gt_source_idx: (M,) ground truth source indices (long).
        gt_dest_idx: (M,) ground truth destination indices (long).
        gt_amount: (M,) ground truth amount values.
        w_source: Weight for source classification cost.
        w_dest: Weight for destination classification cost.
        w_amount: Weight for amount L1 cost.

    Returns:
        (Q, M) cost matrix where entry [i, j] is the cost of assigning
        query i to ground truth connection j.
    """
    Q = pred_source_logits.shape[0]
    M = gt_source_idx.shape[0]

    # Source CE cost: (Q, M)
    # For each query i and gt j, compute CE(pred_source[i], gt_source[j])
    src_log_probs = F.log_softmax(pred_source_logits, dim=-1)  # (Q, n_src)
    src_cost = -src_log_probs[:, gt_source_idx]  # (Q, M)

    # Dest CE cost: (Q, M)
    dst_log_probs = F.log_softmax(pred_dest_logits, dim=-1)  # (Q, n_dst)
    dst_cost = -dst_log_probs[:, gt_dest_idx]  # (Q, M)

    # Amount L1 cost: (Q, M)
    amount_cost = (pred_amount.unsqueeze(1) - gt_amount.unsqueeze(0)).abs()  # (Q, M)

    cost = w_source * src_cost + w_dest * dst_cost + w_amount * amount_cost
    return cost


@torch.no_grad()
def hungarian_match(
    cost_matrix: torch.Tensor,
) -> tuple[list[int], list[int]]:
    """Solve optimal assignment using the Hungarian algorithm.

    Args:
        cost_matrix: (Q, M) cost matrix (Q queries, M ground truth).

    Returns:
        (pred_indices, gt_indices) -- paired assignment indices.
        pred_indices[k] is matched to gt_indices[k].
    """
    cost_np = cost_matrix.detach().cpu().numpy()
    row_ind, col_ind = linear_sum_assignment(cost_np)
    return row_ind.tolist(), col_ind.tolist()


def batch_hungarian_match(
    pred_source_logits: torch.Tensor,
    pred_dest_logits: torch.Tensor,
    pred_amount: torch.Tensor,
    pred_exists: torch.Tensor,
    gt_sparse: torch.Tensor,
    gt_n_mods: torch.Tensor,
    w_source: float = 1.0,
    w_dest: float = 1.0,
    w_amount: float = 1.0,
) -> list[tuple[list[int], list[int], int]]:
    """Run Hungarian matching for a batch.

    Args:
        pred_source_logits: (B, Q, n_sources) source logits.
        pred_dest_logits: (B, Q, n_dest) destination logits.
        pred_amount: (B, Q) predicted amounts.
        pred_exists: (B, Q) predicted existence probabilities (unused in cost,
                     returned for loss computation).
        gt_sparse: (B, max_mods, 7) ground truth sparse connections.
            Columns: [exists, src_idx, dst_idx, amount, bipolar, power, stereo]
        gt_n_mods: (B,) number of active connections per sample.
        w_source: Weight for source cost.
        w_dest: Weight for destination cost.
        w_amount: Weight for amount cost.

    Returns:
        List of (pred_indices, gt_indices, n_gt) per batch element.
    """
    B = pred_source_logits.shape[0]
    matches = []

    for b in range(B):
        n_gt = int(gt_n_mods[b].item())
        if n_gt == 0:
            matches.append(([], [], 0))
            continue

        gt_b = gt_sparse[b, :n_gt]  # (n_gt, 7)
        gt_src = gt_b[:, 1].long()  # source indices
        gt_dst = gt_b[:, 2].long()  # dest indices
        gt_amt = gt_b[:, 3]  # amounts

        cost = compute_cost_matrix(
            pred_source_logits[b],  # (Q, n_src)
            pred_dest_logits[b],  # (Q, n_dst)
            pred_amount[b],  # (Q,)
            gt_src,
            gt_dst,
            gt_amt,
            w_source=w_source,
            w_dest=w_dest,
            w_amount=w_amount,
        )

        pred_idx, gt_idx = hungarian_match(cost)
        matches.append((pred_idx, gt_idx, n_gt))

    return matches

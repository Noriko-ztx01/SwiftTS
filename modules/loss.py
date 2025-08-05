import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
from scipy.stats import weightedtau
import itertools


class Top1CE(nn.Module):
    """
    Top-1 Cross Entropy Loss for Learning-to-Rank tasks.

    This loss encourages the model to assign higher probabilities to items
    that are ranked higher in the ground truth. It is inspired by ListNet,
    which uses a soft version of the ranking list as the target distribution.
    """

    def __init__(self):
        super().__init__()

    def forward(self, logits, labels):
        """
        Compute the Top-1 Cross Entropy loss.

        Args:
            logits (torch.Tensor): Predicted logits from the model.
                                   Shape: [batch_size, num_items]
            labels (torch.Tensor): Ground truth scores (higher = better rank).
                                   Shape: [batch_size, num_items]

        Returns:
            torch.Tensor: Scalar loss value (mean over batch).

        Formula:
            Loss = -E[ sum_i ( P_true(i) * log(P_pred(i)) ) ]
            where P_true = softmax(labels), P_pred = softmax(logits)
        """
        # Convert labels to float and compute softmax as soft target distribution
        true_probs = F.softmax(labels.float(), dim=1)
        pred_log_probs = F.log_softmax(logits, dim=1)

        # Compute cross-entropy between true and predicted distributions
        element_wise_loss = torch.sum(true_probs * pred_log_probs, dim=1)

        # Return negative mean (since we minimize loss)
        return -torch.mean(element_wise_loss)


def w_kendall_metric(score, finetune_acc):
    """
    Compute Weighted Kendall's Tau correlation between predicted scores and ground truth.

    This metric assigns more weight to agreement on items with higher ranks,
    making it suitable for evaluating ranking quality where top-ranked items matter more.

    Args:
        score (dict): Dictionary mapping item IDs to predicted scores.
                      Items with higher scores are ranked higher.
        finetune_acc (list or np.array): Ground truth performance (e.g., fine-tuning accuracy),
                                         in the same order as the keys in `score`.

    Returns:
        float: Weighted Kendall's Tau correlation coefficient (between -1 and 1).
               Higher values indicate better rank agreement.
    """
    # Extract predicted scores in order of items
    score_items = list(score.items())
    metric_score = [item[1] for item in score_items]  # Extract scores
    gt = finetune_acc  # Ground truth values

    # Compute weighted tau
    tw_metric, _ = weightedtau(metric_score, gt)
    return tw_metric


def copelands_aggregation(preferences, candidates=None):
    """
    Perform Copeland's voting rule for rank aggregation.

    Copeland's method determines the winner (or full ranking) by counting pairwise
    victories: each candidate gets +1 for winning a pairwise comparison, 0 for losing,
    and 0 for ties. Final ranking is by total Copeland score (higher = better).

    Args:
        preferences (list of lists): Each sublist represents a voter's ranking of candidates.
                                     Values are ranks: smaller number = higher preference (1 = best).
                                     Example: [1, 3, 2] means candidate 0 is best, candidate 2 is second, etc.
        candidates (list, optional): List of candidate identifiers (e.g., ['A', 'B', 'C']).
                                     If None, defaults to ['A', 'B', ...].

    Returns:
        list: Final ranks for each candidate **in the input order**, where rank 1 is best.
              Example: If input candidates are ['X', 'Y'] and 'Y' wins, returns [2, 1].

    Raises:
        ValueError: If any preference list length does not match number of candidates.
    """
    if preferences is None or len(preferences) == 0:
        return []

    # Determine candidates if not provided
    if candidates is None:
        num_candidates = len(preferences[0])
        candidates = [chr(ord('A') + i) for i in range(num_candidates)]
    else:
        num_candidates = len(candidates)

    # Validate input
    for pref in preferences:
        if len(pref) != num_candidates:
            raise ValueError(f"Each preference list must have {num_candidates} entries. Got {len(pref)}.")

    # Initialize Copeland scores
    scores = {c: 0 for c in candidates}

    # Compare every pair of candidates
    for i, j in itertools.combinations(range(num_candidates), 2):
        a = candidates[i]
        b = candidates[j]
        a_wins = 0
        b_wins = 0

        # Count how many voters prefer a over b
        for pref in preferences:
            a_rank = pref[i]  # Rank of candidate a in this voter's list
            b_rank = pref[j]  # Rank of candidate b

            if a_rank < b_rank:      # Lower rank number = better
                a_wins += 1
            elif b_rank < a_rank:
                b_wins += 1
            # Tie: no one wins

        # Update Copeland scores
        if a_wins > b_wins:
            scores[a] += 1
        elif b_wins > a_wins:
            scores[b] += 1
        # Tie: no score change

    # Sort candidates by Copeland score (descending), break ties alphabetically
    sorted_candidates = sorted(scores.keys(), key=lambda x: (-scores[x], x))

    # Assign final ranks (1 = best)
    rank_dict = {c: rank + 1 for rank, c in enumerate(sorted_candidates)}

    # Return ranks in the order of the input candidates
    return [rank_dict[c] for c in candidates]
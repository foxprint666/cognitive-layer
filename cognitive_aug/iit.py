import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F


class IITIntegrationMonitor(nn.Module):
    """
    Evaluates system irreducibility (Phi) by executing an exhaustive search
    over all unique bipartitions to find the Minimum Information Partition (MIP).
    """

    def __init__(self, normalization: bool = True) -> None:
        super().__init__()
        self.normalization = normalization

    def calculate_phi(self, logits: torch.Tensor) -> float:
        """
        Calculate the Phi (Φ) score of the system given the raw pre-softmax attention logits.

        Args:
            logits: Raw attention logits. Supports 1D distributions [B, N] (star topology)
                    and 2D interaction matrices [B, N, N] (crossbar topology).

        Returns:
            Phi score (float) computed via Cosine Distance between full and partitioned distributions.
        """
        with torch.no_grad():
            if logits.dim() != 2:
                # Implementation for 2D crossbar connectivity [B, N, N]
                return self._calculate_phi_matrix(logits)

            batch_size, num_components = logits.shape
            if num_components < 2:
                return 0.0

            # 1. Full system probability distribution
            full_dist = F.softmax(logits, dim=-1)
            min_normalized_phi = float("inf")
            best_phi = 0.0

            # 2. Iterate through unique bipartitions (Group A and Group B)
            indices = list(range(num_components))
            # Generate bipartitions (only need to evaluate half the combination range)
            for r in range(1, (num_components // 2) + 1):
                for group_a_tuple in itertools.combinations(indices, r):
                    group_a = set(group_a_tuple)
                    group_b = set(indices) - group_a

                    # Isolate Group B by setting logits to a high negative number (suppressing access)
                    partitioned_logits = logits.clone()
                    mask_idx = list(group_b)
                    partitioned_logits[..., mask_idx] = -1e9

                    partitioned_dist = F.softmax(partitioned_logits, dim=-1)

                    # 3. Calculate divergence (using Cosine Distance as the proxy metric)
                    sim = F.cosine_similarity(full_dist, partitioned_dist, dim=-1)
                    phi_p = 1.0 - sim.mean().item()

                    # 4. Apply partition-size normalization to prevent trivial cuts (e.g. cutting 1 element)
                    norm_factor = min(len(group_a), len(group_b)) / num_components
                    normalized_phi = phi_p / norm_factor if norm_factor > 0 else phi_p

                    if normalized_phi < min_normalized_phi:
                        min_normalized_phi = normalized_phi
                        best_phi = phi_p

            return max(0.0, float(best_phi))

    def _calculate_phi_matrix(self, logits: torch.Tensor) -> float:
        """Implementation for 2D crossbar connectivity [B, N, N]"""
        if logits.dim() != 3:
            return 0.0
        batch_size, n, m = logits.shape
        if n != m or n < 2:
            return 0.0

        full_dist = F.softmax(logits, dim=-1)
        min_normalized_phi = float("inf")
        best_phi = 0.0

        indices = list(range(n))
        for r in range(1, (n // 2) + 1):
            for group_a_tuple in itertools.combinations(indices, r):
                group_a = set(group_a_tuple)
                group_b = set(indices) - group_a

                # Find MIP by zeroing out connections from Group B -> Group A
                partitioned_logits = logits.clone()
                for i in group_b:
                    for j in group_a:
                        partitioned_logits[:, i, j] = -1e9

                partitioned_dist = F.softmax(partitioned_logits, dim=-1)

                full_flat = full_dist.reshape(batch_size, -1)
                part_flat = partitioned_dist.reshape(batch_size, -1)

                sim = F.cosine_similarity(full_flat, part_flat, dim=-1)
                phi_p = 1.0 - sim.mean().item()

                norm_factor = min(len(group_a), len(group_b)) / n
                normalized_phi = phi_p / norm_factor if norm_factor > 0 else phi_p

                if normalized_phi < min_normalized_phi:
                    min_normalized_phi = normalized_phi
                    best_phi = phi_p

        return max(0.0, float(best_phi))

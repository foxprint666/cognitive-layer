import torch
import torch.nn as nn
import torch.nn.functional as F


class IITIntegrationMonitor(nn.Module):
    """
    Integrated Information Theory (IIT 4.0) Causal Integration Monitor.

    Measures the system irreducibility (Phi / Φ) of the Global Workspace
    by applying a Minimum Information Partition (MIP) mask to the raw attention logits.
    """

    def __init__(self) -> None:
        super().__init__()

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
            if logits.dim() not in (2, 3):
                return 0.0

            # 1. Full system state distribution
            full_dist = F.softmax(logits, dim=-1)

            # 2. Simulate Minimum Information Partition (MIP)
            # We copy logits to create a partitioned version where causal links are severed.
            partitioned_logits = logits.clone()

            if logits.dim() == 3 and logits.size(1) == logits.size(2):
                # 2D Interaction Matrix [B, N, N]: Off-diagonal masking (isolate nodes)
                batch_size, n, _ = logits.shape
                # Create an identity mask to keep diagonals, zero out off-diagonals.
                # In logits space, setting to -1e9 effectively zeroes out the post-softmax probability.
                mask = (
                    torch.eye(n, device=logits.device)
                    .bool()
                    .unsqueeze(0)
                    .expand(batch_size, n, n)
                )
                partitioned_logits[~mask] = -1e9
            else:
                # 1D Vector [B, N]: Bisection masking (disconnect half the modules from the workspace)
                n = logits.size(-1)
                mid = n // 2
                partitioned_logits[..., mid:] = -1e9

            # 3. Partitioned distribution
            partitioned_dist = F.softmax(partitioned_logits, dim=-1)

            # 4. Calculate Phi as Cosine Distance (1.0 - Cosine Similarity)
            # Flatten distributions for distance metric
            full_flat = full_dist.reshape(logits.size(0), -1)
            part_flat = partitioned_dist.reshape(logits.size(0), -1)

            # cosine_similarity returns values in [-1, 1]. Distance is 1 - sim.
            sim = F.cosine_similarity(full_flat, part_flat, dim=-1)
            phi = 1.0 - sim.mean().item()

            return max(0.0, float(phi))

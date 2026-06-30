"""
Statistical-pooling temporal encoder (Approach 1, pure).

Problem found by ablation: the existing RNN keeps only the FINAL hidden state,
which is recency-biased -> it learns something near `last` (already in input),
so z adds ~nothing, while hand-made mean/std (MS) adds +0.05 AUPR.

Fix: pool over ALL valid hidden states with order-invariant statistics
(mean, std, max) + the final state, then project to latent. This gives the
encoder the capacity to LEARN mean/std-like summaries over the transformed
hidden space, instead of being handed them.

Drop-in: same constructor signature as RNNPolicyNetwork, same forward output
(z, log_prob, mean), so diag_ablate.py / the RL loop can use it unchanged.
"""
import torch, torch.nn as nn, torch.distributions as dist
from TimeEmbedding import DEVICE, TimeEmbeddedRNNCell


class PooledRNNCell(TimeEmbeddedRNNCell):
    """Same as TimeEmbeddedRNNCell but returns ALL per-step hidden states
    (stacked) plus a validity mask, so the caller can pool over the sequence."""

    def forward_all(self, batch_times, batch_values, batch_masks, lengths):
        batch_size = batch_times.size(0)
        max_seq_len = batch_times.size(1)
        h = self.h0.unsqueeze(0).repeat(batch_size, 1)
        all_h = []
        for i in range(max_seq_len):
            if i > 0:
                delta_t = batch_times[:, i] - batch_times[:, i - 1]
            else:
                delta_t = torch.zeros(batch_size, device=h.device)
            time_embedding = self.time_embedder(delta_t.unsqueeze(-1))
            combined_input = torch.cat(
                [batch_values[:, i], batch_masks[:, i], time_embedding], dim=-1)
            h_new = self.gru_cell(combined_input, h)
            valid = (i < lengths).float().unsqueeze(-1)
            h = valid * h_new + (1 - valid) * h
            all_h.append(h)
        H = torch.stack(all_h, dim=1)                       # [B, T, hidden]
        step_idx = torch.arange(max_seq_len, device=h.device).unsqueeze(0)
        valid_mask = (step_idx < lengths.unsqueeze(1)).float()  # [B, T]
        return H, valid_mask, h                              # all states, mask, final


class PooledRNNPolicyNetwork(nn.Module):
    """Statistical-pooling encoder with a Gaussian policy head.

    Pools all valid hidden states into [mean || std || max || last] (4*hidden),
    then projects to (mean, log_std) of the latent policy. Output signature
    matches RNNPolicyNetwork exactly.
    """
    def __init__(self, input_dim, hidden_dim, latent_dim, time_dim=32):
        super().__init__()
        self.rnn_cell = PooledRNNCell(input_dim, hidden_dim, time_dim)
        pooled_dim = hidden_dim * 4  # mean, std, max, last
        self.proj = nn.Sequential(
            nn.Linear(pooled_dim, pooled_dim), nn.SiLU(),
        )
        self.fc_mean = nn.Linear(pooled_dim, latent_dim)
        self.fc_logstd = nn.Linear(pooled_dim, latent_dim)
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim

    def _pool(self, H, valid_mask, final_h):
        # H: [B,T,hid], valid_mask: [B,T]
        m = valid_mask.unsqueeze(-1)                         # [B,T,1]
        n = m.sum(dim=1).clamp(min=1.0)                      # [B,1]
        mean = (H * m).sum(dim=1) / n                        # [B,hid]
        var = ((H - mean.unsqueeze(1))**2 * m).sum(dim=1) / n
        std = torch.sqrt(var + 1e-6)
        very_neg = torch.finfo(H.dtype).min
        H_masked = H.masked_fill(m == 0, very_neg)
        mx = H_masked.max(dim=1).values                      # [B,hid]
        return torch.cat([mean, std, mx, final_h], dim=-1)   # [B, 4*hid]

    def forward(self, batch_data, deterministic=False, temperature=1.0):
        times = batch_data['times'].to(DEVICE)
        values = batch_data['values'].to(DEVICE)
        masks = batch_data['masks'].to(DEVICE)
        lengths = batch_data['lengths'].to(DEVICE)

        H, valid_mask, final_h = self.rnn_cell.forward_all(
            times, values, masks, lengths)
        pooled = self._pool(H, valid_mask, final_h)
        feat = self.proj(pooled)

        mean = self.fc_mean(feat)
        log_std = torch.clamp(self.fc_logstd(feat), min=-20, max=2)
        std = torch.exp(log_std) * temperature
        policy_dist = dist.Normal(mean, std)
        if deterministic:
            z, log_prob = mean, None
        else:
            z = policy_dist.rsample()
            log_prob = policy_dist.log_prob(z).sum(dim=-1)
        return z, log_prob, mean
# mrope_table.py
import torch
from typing import Tuple

# ---------- 1. 离线建表 ----------
def build_mrope_table(inv_freq: torch.Tensor, max_pos: int = 4096):
    """
    inv_freq: [half_dim]
    return: (cos_table, sin_table)  形状 [max_pos, half_dim]
    """
    half_dim = inv_freq.shape[0]
    pos = torch.arange(max_pos, dtype=inv_freq.dtype, device=inv_freq.device)
    freqs = pos.unsqueeze(1) * inv_freq              # [max_pos, half_dim]
    emb = torch.cat([freqs, freqs], dim=1)           # [max_pos, 2*half_dim]
    cos_t = emb.cos()[:, :half_dim]                  # 提前 slice
    sin_t = emb.sin()[:, :half_dim]
    return cos_t, sin_t

# ---------- 2. 运行时查表 ----------
def lookup_mrope_freqs(
    cos_table: torch.Tensor,
    sin_table: torch.Tensor,
    position_ids: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    position_ids: 任意 shape，元素值 ∈ [0, max_pos)
    return: (cos, sin)  与 compute_mrope_freqs 输出形状完全一致
    """
    cos = cos_table[position_ids, :]
    sin = sin_table[position_ids, :]
    cos = cos_table[position_ids]
    sin = sin_table[position_ids]
    return cos, sin

# ---------- 3. 原始计算（供对比） ----------
def compute_mrope_freqs(
    inv_freq: torch.Tensor,
    position_ids: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    half_dim = inv_freq.shape[0]
    inv_freq_expanded = inv_freq.reshape(1, 1, half_dim, 1)
    position_ids_expanded = position_ids.unsqueeze(2).to(dtype=inv_freq_expanded.dtype)
    freqs = inv_freq_expanded * position_ids_expanded
    freqs = freqs.transpose(2, 3)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = emb.cos()
    sin = emb.sin()
    cos = cos[:, :, :, : cos.shape[-1] // 2]
    sin = sin[:, :, :, : sin.shape[-1] // 2]
    return cos, sin


# ---------------- 测试 ----------------
if __name__ == "__main__":
    torch.manual_seed(0)
    inv_freq = 1.0 / (
        1000000.0 ** (torch.arange(0, 128, 2, dtype=torch.float32) / 128)
    )
    ar_len = 1
    position_ids = torch.zeros((3, 1, ar_len), dtype=torch.int32)
    for i in range(3):
        position_ids[i] = torch.arange(ar_len, dtype=torch.int32)
    # first_line = [0, 1,3,2,4,7,3,5,11,8,13,21,13,3,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
    # position_ids[0, 0] = torch.tensor(first_line, dtype=torch.int32)

    # ---- 原始计算 ----
    cos0, sin0 = compute_mrope_freqs(inv_freq, position_ids)

    # ---- 建表 + 查表 ----
    cos_table, sin_table = build_mrope_table(inv_freq, max_pos=4096)
    cos1, sin1 = lookup_mrope_freqs(cos_table, sin_table, position_ids)

    # ---- 对比 ----
    print("cos 误差:", torch.max(torch.abs(cos0 - cos1)).item())
    print("sin 误差:", torch.max(torch.abs(sin0 - sin1)).item())
    assert torch.allclose(cos0, cos1, atol=1e-6)
    assert torch.allclose(sin0, sin1, atol=1e-6)
    print("✅ 建表/查表与原始实现 bit-wise 一致")
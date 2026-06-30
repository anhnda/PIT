Để tao gom đúng những con số đã chạy ra, không thêm thắt.

**Nguồn 1 — run 10-fold đầy đủ (diag_rl.py, dЗ = (S+L+Z) − (S+L) trên test):**

| fold | pre AUPR | +RL AUPR | scratch AUPR | pre AUC | +RL AUC | scratch AUC |
|---|---|---|---|---|---|---|
| 0 | +0.0339 | +0.0374 | +0.0248 | +0.0105 | +0.0111 | +0.0111 |
| 1 | −0.0127 | −0.0164 | −0.0001 | −0.0076 | −0.0105 | +0.0012 |
| 2 | +0.0142 | +0.0242 | −0.0153 | +0.0236 | +0.0300 | +0.0184 |
| 3 | +0.0014 | +0.0094 | +0.0010 | +0.0175 | +0.0184 | +0.0256 |
| 4 | −0.0217 | +0.0082 | −0.0084 | +0.0055 | +0.0070 | +0.0175 |
| 5 | −0.0081 | −0.0128 | +0.0055 | −0.0003 | +0.0006 | +0.0105 |
| 6 | +0.0283 | +0.0328 | +0.0418 | +0.0105 | +0.0134 | +0.0210 |
| 7 | −0.0225 | −0.0174 | −0.0019 | −0.0201 | −0.0171 | −0.0018 |
| 8 | −0.0629 | −0.0564 | −0.0350 | −0.0159 | −0.0134 | −0.0110 |
| 9 | +0.0247 | +0.0231 | +0.0342 | +0.0104 | +0.0150 | +0.0238 |

**Mean ± std (10 fold):**

| | AUPR dЗ | AUC dЗ |
|---|---|---|
| pretrain | −0.0025 ± 0.0280 | +0.0034 ± 0.0135 |
| +RL (on pretrain) | +0.0032 ± 0.0275 | +0.0054 ± 0.0145 |
| scratch | +0.0047 ± 0.0221 | +0.0116 ± 0.0115 |

**Absolute (S+L+Z) trên test:**

| | AUPR | AUC |
|---|---|---|
| pretrain | 0.5480 ± 0.0747 | 0.8129 ± 0.0377 |
| +RL | 0.5538 ± 0.0765 | 0.8149 ± 0.0375 |
| scratch | 0.5553 ± 0.0743 | 0.8211 ± 0.0392 |

**Paired-test (cái này mới là phán quyết, không phải fold lẻ):**

| so sánh | mean | wins | t-p | Wilcoxon | dz |
|---|---|---|---|---|---|
| +RL dЗ vs pretrain dЗ — AUC | +0.0020 | 9/10 | 0.032 | 0.031 | +0.80 |
| +RL dЗ vs pretrain dЗ — AUPR | +0.0057 | 7/10 | 0.096 | 0.065 | +0.59 |
| scratch dЗ vs pretrain dЗ — AUC | +0.0082 | 9/10 | 0.0038 | 0.0098 | +1.22 |
| scratch dЗ vs pretrain dЗ — AUPR | +0.0072 | 7/10 | 0.20 | 0.16 | +0.44 |
| scratch (S+L+Z vs S+L) — AUC | +0.0116 | 8/10 | 0.014 | 0.027 | +0.96 |
| scratch (S+L+Z vs S+L) — AUPR | +0.0047 | 5/10 | 0.54 | — | +0.20 |

Đọc được gì **chắc chắn** từ đây: thứ tự **scratch > +RL > pretrain** trên cả AUPR và AUC absolute, nhất quán. Significance chỉ đạt ở **AUC** (scratch dz +1.22, +RL dz +0.80; cả hai 9/10 wins). **AUPR không đạt** ở bất kỳ so sánh nào (p toàn > 0.06). base (S+L) là mốc 0 — pretrain gần như không vượt được nó (AUPR −0.0025, AUC +0.0034, không significant).

---

**Nguồn 2 — debug fold 8 & 9, chỉ scratch, final-epoch dЗ-AUPR:**

| fold | on-pretrain | scratch | scratch − pre |
|---|---|---|---|
| 8 | −0.0531 | −0.0200 | +0.0331 |
| 9 | +0.0292 | +0.0313 | +0.0021 |

Scratch thắng pre ở cả hai, fold 8 thắng đậm.

---

**Giới hạn của bảng — để mày không bị tao dẫn quá:**

- Mean AUPR có std ±0.028, gần bằng chính giá trị mean → khoảng cách scratch/+RL/pre trên AUPR nằm trong noise. Chỉ AUC là tách được khỏi noise.
- Nguồn 2 là **1 seed/fold**, chưa lặp. scratch−pre ở fold 9 (+0.0021) nhỏ tới mức có thể là noise; fold 8 (+0.0331) lớn hơn nhưng vẫn 1 lần đo.
- Mấy con base@1e-3 / posw / balanced / ap ở các run reward-sweep tao **không** gộp vào đây vì chúng chỉ chạy fold 8 (±9), không phải 10-fold — gộp vào sẽ khập khiễng.

Tóm một câu phòng thủ được: **scratch ≥ +RL ≥ pretrain ≥ base về mức trung tâm; khác biệt là thật và significant trên AUC, còn trên AUPR thì chưa tách được khỏi noise.**
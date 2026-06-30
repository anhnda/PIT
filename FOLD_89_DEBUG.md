Gom đúng số đã chạy cho fold 8 & 9, không thêm.

## Bảng 1 — scratch vs pre, final-epoch dЗ trên test

| fold | nhánh | epoch0 (start) | oracle (best test ep) | proxy-stop | final | corr_AUPR | corr_AUC |
|---|---|---|---|---|---|---|---|
| 8 | on-pretrain | −0.0629 | −0.0493 | −0.0591 | −0.0531 | +0.227 | +0.238 |
| 8 | scratch | −0.0390 | −0.0200 | −0.0457 | −0.0200 | **−0.406** | −0.188 |
| 9 | on-pretrain | +0.0247 | +0.0369 | +0.0301 | +0.0292 | +0.339 | **−0.282** |
| 9 | scratch | +0.0227 | +0.0462 | +0.0309 | +0.0313 | +0.031 | +0.521 |

(tất cả là dЗ-AUPR; corr = tương quan giữa train-OOF-proxy và test qua các epoch)

**scratch − pre (final AUPR):** fold 8 **+0.0331**, fold 9 **+0.0021**. Scratch thắng cả hai, fold 8 đậm.

## Bảng 2 — reward sweep, CHỈ scratch, fold 8, final dЗ-AUPR

| mode | lr | final | best | r_pos range | final drift | grad |
|---|---|---|---|---|---|---|
| base | 3e-4 | −0.0197 | −0.0197 | 0.332–0.345 | 1.36 | ~1 |
| base | 1e-3 | **−0.0113** | −0.0113 | 0.335–0.354 | 5.34 | ~5 |
| base | 3e-3 | −0.0135 | −0.0035 | 0.334–0.353 | 11.5 | nổ (→255) |
| posw | 3e-4 | −0.0146 | −0.0124 | 0.333–0.342 | 1.98 | ~12 |
| posw | 1e-3 | −0.0271 | −0.0116 | 0.333–0.341 | 4.59 | ~16 |
| posw | 3e-3 | −0.0174 | −0.0070 | 0.334–0.341 | 6.71 | ~16 |
| balanced | 3e-4 | −0.0322 | −0.0322 | 0.333–0.344 | 1.12 | ~0.4 |
| ap | 3e-4 | −0.0322 | −0.0314 | 0.758–0.776 | 1.14 | ~0.4 |

(start mọi run = −0.0390, là encoder scratch random-init trước RL)

## Cái rút ra được, theo mức chắc chắn

**Chắc (đọc thẳng từ số):**
- Cả 8 lẫn 9, **scratch ≥ on-pretrain** ở final. fold 8: pretrain đẩy start xuống −0.063, scratch start chỉ −0.039.
- **early-stop proxy hỏng**: fold 8 scratch corr_AUPR = −0.41 (train-proxy đi *ngược* test); fold 9 pre corr_AUC = −0.28. Tín hiệu nhìn-từ-train không bám test.
- **reward shaping không tạo thông tin**: balanced/ap không hơn base (còn tệ hơn). r_pos kẹt 0.33–0.35 mọi mode trừ ap (nhưng ap cao r_pos do định nghĩa rank, AUPR vẫn −0.032).
- **posw ≈ base-lr-cao**: base@1e-3 (−0.0113) > posw@3e-4 (−0.0146). posw chỉ là gradient lớn hơn, không phải reward tốt hơn.
- fold 8 final cải thiện đi cùng **drift**, không cùng reward mode. lr=3e-3 thì grad nổ (255), mất ổn định.

**Chưa chắc (đừng tin quá):**
- Tất cả fold 8/9 ở trên là **1 seed**, chưa lặp. scratch−pre fold 9 (+0.0021) nhỏ tới mức có thể là noise.
- **Không mode/lr nào đưa fold 8 lên dương.** Sàn quanh −0.011 đến −0.020. Tao *chưa* biết vì sao (chưa in base/full tuyệt đối) — nên không kết luận "giới hạn dữ liệu", đó là chỗ tao chém lúc nãy.
- fold 8 best chạm −0.0035 (base@3e-3 ep44) nhưng không giữ được → có điểm gần-hòa-baseline nhưng RL hiện không dừng lại ở đó được.

Muốn đào tiếp chỗ "chưa chắc" thì việc rẻ nhất là in `base (S+L)` và `full (S+L+Z)` tuyệt đối cho fold 8 — để biết âm là do S+L cao hay S+L+Z thấp, thay vì đoán. Cần thì tao thêm vào script.
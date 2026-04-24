import cv2
import numpy as np
from pathlib import Path

ROOT = Path("data/REFUGE2/Annotation-Training400/Disc_Cup_Masks")

count = 0
for bmp in ROOT.rglob("*.bmp"):
    m = cv2.imread(str(bmp), cv2.IMREAD_UNCHANGED)
    if m is None:
        continue
    if set(np.unique(m).tolist()) <= {0, 1}:
        continue  # 已二值

    out = np.zeros_like(m, dtype=np.uint8)
    out[(m == 0) | (m == 128)] = 1    # disc 区域（含 cup）→ 1
    # 255 (背景) → 保持 0

    cv2.imwrite(str(bmp), out)
    count += 1

print(f"转换了 {count} 个文件")

sample = ROOT / "Glaucoma" / "g0001.bmp"
m = cv2.imread(str(sample), cv2.IMREAD_UNCHANGED)
print(f"{sample.name}: unique = {np.unique(m)}, disc 像素数 = {(m == 1).sum()}")
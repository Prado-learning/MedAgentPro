import shutil
from pathlib import Path

SRC = Path("REFUGE2_raw/REFUGE2/train")
DST = Path("REFUGE2")

img_g  = DST / "Training400" / "Glaucoma"
img_ng = DST / "Training400" / "Non-Glaucoma"
msk_g  = DST / "Annotation-Training400" / "Disc_Cup_Masks" / "Glaucoma"
msk_ng = DST / "Annotation-Training400" / "Disc_Cup_Masks" / "Non-Glaucoma"
for d in [img_g, img_ng, msk_g, msk_ng]:
    d.mkdir(parents=True, exist_ok=True)

# 先清空目标目录
for d in [img_g, img_ng, msk_g, msk_ng]:
    for f in d.iterdir():
        f.unlink()

for src_img in sorted((SRC / "images").glob("*.jpg")):
    stem = src_img.stem                         # g0001 / n0003
    prefix = stem[0]
    src_msk = SRC / "mask" / f"{stem}.bmp"

    if prefix == "g":
        dst_img, dst_msk = img_g / src_img.name, msk_g / f"{stem}.bmp"
    elif prefix == "n":
        dst_img, dst_msk = img_ng / src_img.name, msk_ng / f"{stem}.bmp"
    else:
        continue

    shutil.copy2(src_img, dst_img)
    if src_msk.exists():
        shutil.copy2(src_msk, dst_msk)

print("Done:")
print(f"  Glaucoma 图像:     {len(list(img_g.glob('*.jpg')))}")
print(f"  Non-Glaucoma 图像: {len(list(img_ng.glob('*.jpg')))}")
print(f"  Glaucoma mask:     {len(list(msk_g.glob('*.bmp')))}")
print(f"  Non-Glaucoma mask: {len(list(msk_ng.glob('*.bmp')))}")
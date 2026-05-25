#!/usr/bin/env python3
"""prep_cbis.py — Kaggle awsaf49 (JPEG) version for the CFT VTAB pipeline.

Reads <src>/csv/dicom_info.csv + 4 case CSVs from awsaf49's
'cbis-ddsm-breast-cancer-image-dataset' layout, resizes the chosen series'
JPEGs (<src>/jpeg/<UID>/<frame>.jpg) to <size>x<size> PNGs under
<out>/images/, and writes VTAB-format <out>/train800.txt + <out>/test.txt.

Default: series='cropped images', binary labels (MALIGNANT=1, BENIGN/BWC=0).
Idempotent: existing PNGs are skipped.
"""
import argparse
from pathlib import Path
import pandas as pd
from PIL import Image
from tqdm import tqdm


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True,
                   help="Top dir with csv/ and jpeg/ subdirs (kagglehub awsaf49 layout)")
    p.add_argument("--out", default="../data/vtab-1k/cbis_ddsm")
    p.add_argument("--series", default="cropped images",
                   choices=["cropped images", "full mammogram images"])
    p.add_argument("--size", type=int, default=224)
    args = p.parse_args()

    src, out = Path(args.src), Path(args.out)
    img_dir = out / "images"; img_dir.mkdir(parents=True, exist_ok=True)
    csv_dir = src / "csv"

    # 1) Build TWO PatientID -> (label, split) keys per case row:
    #    - cropped form keeps the abnormality_id suffix (one-per-abnormality)
    #    - full-mammo form drops it (one-per-image); label = max over abnormalities
    labels = {}
    for csv_name, typ, split in [
        ("mass_case_description_train_set.csv", "Mass", "train"),
        ("mass_case_description_test_set.csv",  "Mass", "test"),
        ("calc_case_description_train_set.csv", "Calc", "train"),
        ("calc_case_description_test_set.csv",  "Calc", "test"),
    ]:
        df = pd.read_csv(csv_dir / csv_name)
        side = "Training" if split == "train" else "Test"
        for _, r in df.iterrows():
            lab = 1 if "MALIGNANT" in str(r["pathology"]).upper() else 0
            base = (f"{typ}-{side}_{r['patient_id']}_{r['left or right breast']}"
                    f"_{r['image view']}")
            labels[f"{base}_{int(r['abnormality id'])}"] = (lab, split)        # cropped key
            prev = labels.get(base, (0, split))
            labels[base] = (max(prev[0], lab), split)                          # full-mammo key
    print(f"[prep] label keys: {len(labels)}")

    # 2) Walk dicom_info, filter by series, resize JPEGs.
    di = pd.read_csv(csv_dir / "dicom_info.csv")
    sel = di[di["SeriesDescription"] == args.series].copy()
    print(f"[prep] series='{args.series}': {len(sel)} entries")

    train_lines, test_lines, n_skip = [], [], 0
    counts = {}
    for _, r in tqdm(sel.iterrows(), total=len(sel)):
        pid = r["PatientID"]
        if pid not in labels:
            n_skip += 1; continue
        lab, split = labels[pid]
        # image_path = 'CBIS-DDSM/jpeg/<UID>/<frame>.jpg' — strip leading prefix.
        img_rel = r["image_path"]
        if img_rel.startswith("CBIS-DDSM/"):
            img_rel = img_rel[len("CBIS-DDSM/"):]
        src_jpg = src / img_rel
        if not src_jpg.exists():
            n_skip += 1; continue
        counts[pid] = counts.get(pid, 0) + 1
        out_name = f"{pid}.png" if counts[pid] == 1 else f"{pid}_v{counts[pid]}.png"
        out_path = img_dir / out_name
        if not out_path.exists():
            try:
                Image.open(src_jpg).convert("L") \
                     .resize((args.size, args.size), Image.BICUBIC).convert("RGB") \
                     .save(out_path)
            except Exception:
                n_skip += 1; continue
        (train_lines if split == "train" else test_lines).append(
            f"images/{out_name} {lab}")

    (out / "train800.txt").write_text("\n".join(train_lines) + "\n")
    (out / "test.txt").write_text("\n".join(test_lines) + "\n")
    print(f"[prep] train={len(train_lines)} test={len(test_lines)} skipped={n_skip}")
    print(f"[prep] DONE -> {out}")


if __name__ == "__main__":
    main()

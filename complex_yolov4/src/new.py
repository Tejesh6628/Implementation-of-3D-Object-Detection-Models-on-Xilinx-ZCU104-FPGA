import os
import shutil
import cv2
import numpy as np
import imgaug.augmenters as iaa
from tqdm import tqdm
import random
import pickle

# ================= PATHS =================
SRC = r"F:\KITTI\training"
DST = r"F:\Augmented_Final_Sequence"

IMG_SRC = os.path.join(SRC, "image_2")
LBL_SRC = os.path.join(SRC, "label_2")
VEL_SRC = os.path.join(SRC, "velodyne")
CAL_SRC = os.path.join(SRC, "calib")

IMG_DST = os.path.join(DST, "image_2")
LBL_DST = os.path.join(DST, "label_2")
VEL_DST = os.path.join(DST, "velodyne")
CAL_DST = os.path.join(DST, "calib")

TOTAL_REQUIRED = 11996

# ================= AUGMENTATIONS =================
augmenter = iaa.Sequential([
    iaa.Fliplr(0.5),
    iaa.Affine(rotate=(-15, 15), scale=(0.8, 1.2)),
    iaa.LinearContrast((0.75, 1.5))
])

# ================= CREATE FOLDERS =================
for p in [IMG_DST, LBL_DST, VEL_DST, CAL_DST]:
    os.makedirs(p, exist_ok=True)

# ================= LABEL TRANSFORM =================
def transform_label(label_path, img_w, flip=False):
    new_lines = []
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            xmin, ymin, xmax, ymax = map(float, parts[4:8])

            if flip:
                xmin, xmax = img_w - xmax, img_w - xmin

            parts[4:8] = list(map(str, [xmin, ymin, xmax, ymax]))
            new_lines.append(" ".join(parts))
    return new_lines

# ================= VELODYNE TRANSFORM =================
def transform_velodyne(pc_path, flip=False):
    pts = np.fromfile(pc_path, dtype=np.float32).reshape(-1, 4)

    if flip:
        pts[:, 1] = -pts[:, 1]

    # rotation
    angle = np.random.uniform(-15, 15) * np.pi / 180
    R = np.array([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle),  np.cos(angle)]
    ])
    pts[:, :2] = pts[:, :2] @ R

    # scaling
    scale = np.random.uniform(0.9, 1.1)
    pts[:, :3] *= scale

    return pts

# ================= LOAD FILE IDS =================
files = sorted([f.replace(".png", "") for f in os.listdir(IMG_SRC)])

# ================= AUGMENTATION LOOP =================
seq = 0

for i in tqdm(range(TOTAL_REQUIRED), desc="Augmenting Dataset"):

    file_id = random.choice(files)
    new_id = f"{seq:06d}"

    # ---------- IMAGE ----------
    img = cv2.imread(os.path.join(IMG_SRC, file_id + ".png"))
    if img is None:
        continue

    aug_img = augmenter(image=img)
    cv2.imwrite(os.path.join(IMG_DST, new_id + ".png"), aug_img)

    # ---------- LABEL ----------
    lbl_path = os.path.join(LBL_SRC, file_id + ".txt")
    new_lbl = transform_label(lbl_path, img.shape[1], flip=True)

    with open(os.path.join(LBL_DST, new_id + ".txt"), "w") as f:
        f.write("\n".join(new_lbl))

    # ---------- CALIB ----------
    shutil.copy(
        os.path.join(CAL_SRC, file_id + ".txt"),
        os.path.join(CAL_DST, new_id + ".txt")
    )

    # ---------- VELODYNE ----------
    pc_src = os.path.join(VEL_SRC, file_id + ".bin")
    if os.path.exists(pc_src):
        pts = transform_velodyne(pc_src, flip=True)
        pts.tofile(os.path.join(VEL_DST, new_id + ".bin"))

    seq += 1

print(f"\n✅ Augmentation completed: {seq} samples")

# ================= SPLIT DATA =================
ids = list(range(seq))
random.shuffle(ids)

train_end = int(0.6 * seq)
val_end = int(0.8 * seq)

train_ids = ids[:train_end]
val_ids = ids[train_end:val_end]
test_ids = ids[val_end:]

# ================= IMAGESETS =================
imgset_dir = os.path.join(DST, "ImageSets")
os.makedirs(imgset_dir, exist_ok=True)

def save_split(name, id_list):
    with open(os.path.join(imgset_dir, f"{name}.txt"), "w") as f:
        for i in id_list:
            f.write(f"{i:06d}\n")

save_split("train", train_ids)
save_split("val", val_ids)
save_split("test", test_ids)
save_split("trainval", train_ids + val_ids)

print("✅ ImageSets created")

# ================= GT DATABASE =================
gt_database = []

for i in tqdm(train_ids, desc="Creating GT Database"):
    file_id = f"{i:06d}"
    label_file = os.path.join(LBL_DST, file_id + ".txt")

    if not os.path.exists(label_file):
        continue

    with open(label_file) as f:
        for line in f:
            parts = line.strip().split()
            cls = parts[0]

            gt_database.append({
                "name": cls,
                "image_idx": file_id
            })

with open(os.path.join(DST, "kitti_dbinfos_train.pkl"), "wb") as f:
    pickle.dump(gt_database, f)

print("✅ GT database created")

# ================= INFO FILES =================
dummy_info = {"dataset": "KITTI"}

for name in ["train", "val", "test", "trainval"]:
    with open(os.path.join(DST, f"kitti_infos_{name}.pkl"), "wb") as f:
        pickle.dump(dummy_info, f)

print("✅ Info PKL files created")

print("\n🔥 FULL PIPELINE COMPLETED SUCCESSFULLY 🔥")
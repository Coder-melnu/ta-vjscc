import os
import shutil
from pathlib import Path

videos_src  = Path("datasets/UCF101Videos/UCF-101")
videos_root = Path("datasets/UCF101Videos/UCF101Split")
splits_dir  = Path("datasets/UCF101TrainTestSplits-RecognitionTask/ucfTrainTestlist")

def organize_split(txt_file, split_name):
    print(f"\nOrganizing {split_name} from {txt_file.name} ...")
    lines = txt_file.read_text().strip().splitlines()
    n_done, n_missing = 0, 0

    for line in lines:
        # trainlist has "ClassName/clip.avi label", testlist has just "ClassName/clip.avi"
        rel_path = line.split()[0]
        class_name, clip_file = rel_path.split("/")

        src = videos_src / class_name / clip_file
        dst_dir = videos_root / split_name / class_name
        dst = dst_dir / clip_file

        if not src.exists():
            print(f"  MISSING: {src}")
            n_missing += 1
            continue

        dst_dir.mkdir(parents=True, exist_ok=True)
        if not dst.exists():
            shutil.copy2(src, dst)
        n_done += 1

    print(f"  Done: {n_done} clips, {n_missing} missing")

organize_split(splits_dir / "trainlist01.txt", "train")
organize_split(splits_dir / "testlist01.txt",  "test")

print("\nAll done.")
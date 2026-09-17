import os
import re
import shutil
import argparse
import pydicom
from collections import defaultdict

def ingest_recursive_dicom_folder(source_root, output_target_dir="data/imported_patient_scan"):
    """Recursively walks through subfolders, filters specifically for Sagittal/Coronal 
    structural series, and cleanly ingests the targets to prevent Axial pollution.
    """
    if not os.path.exists(source_root):
        raise FileNotFoundError(f"❌ Source directory path not found: {source_root}")
        
    if os.path.exists(output_target_dir):
        shutil.rmtree(output_target_dir)
    os.makedirs(output_target_dir, exist_ok=True)
    
    print(f"🔍 Initializing targeted structural series search across root directory: {source_root}")
    
    series_groups = defaultdict(list)
    series_descriptions = {}

    for root_dir, _, file_list in os.walk(source_root):
        for file_name in file_list:
            source_file_path = os.path.join(root_dir, file_name)
            try:
                ds = pydicom.dcmread(source_file_path, stop_before_pixels=True, force=True)
                if 'SeriesInstanceUID' in ds:
                    uid = ds.SeriesInstanceUID
                    series_groups[uid].append(source_file_path)
                    if 'SeriesDescription' in ds and uid not in series_descriptions:
                        series_descriptions[uid] = ds.SeriesDescription
            except Exception:
                continue

    if not series_groups:
        print("❌ Ingestion Failure: Unable to parse any valid DICOM series UIDs.")
        return None

    def series_score(uid):
        """Rank a series for bone age use; None means reject it outright.

        Matching whole words, not substrings: the old check rejected any description
        containing 'ax', which also hits words like 'relax' and 'max'.
        """
        words = set(re.split(r"[^a-z0-9]+", series_descriptions.get(uid, "").lower()))
        if words & {"ax", "axial", "tra", "transverse", "localizer", "loc", "scout", "survey"}:
            return None

        score = 0
        if words & {"sag", "sagittal"}:
            score += 4  # sagittal shows both growth plates end-on
        if words & {"cor", "coronal"}:
            score += 2
        if words & {"pd", "t1", "t2", "fs", "tse", "spc"}:
            score += 1
        return score

    ranked = [(score, len(series_groups[uid]), uid) for uid in series_groups
              if (score := series_score(uid)) is not None]
    if not ranked:  # everything looked like a localizer: fall back to the longest series
        ranked = [(0, len(files), uid) for uid, files in series_groups.items()]

    best_series_uid = max(ranked)[2]  # highest score, then most slices

    chosen_files = series_groups[best_series_uid]
    desc = series_descriptions.get(best_series_uid, "Unknown Sequence")

    print(f"🎯 Isolated Target Structural Series: {desc} ({len(chosen_files)} slices)")

    valid_count = 0
    for file_path in chosen_files:
        try:
            ds = pydicom.dcmread(file_path, force=True)
            if hasattr(ds, 'pixel_array') and ds.pixel_array is not None:
                destination_path = os.path.join(output_target_dir, f"slice_{valid_count:03d}.dcm")
                shutil.copy2(file_path, destination_path)
                valid_count += 1
        except Exception:
            continue

    print("\n" + "="*55)
    print(" 📥 CLINICAL DATASET INGESTION COMPLETE ")
    print("="*55)
    print(f"📂 Deep Source Root : {source_root}")
    print(f"🎞️ Isolated Series  : {desc}")
    print(f"💾 Ingested Volume : {valid_count} Slices Successfully Extracted")
    print(f"Target Directory   : {output_target_dir}/")
    print("="*55 + "\n")
    return output_target_dir

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Recursive Clinical Data Upload Interface")
    parser.add_argument("--source", type=str, required=True, help="Path to raw unstructured folder of DICOM files")
    args = parser.parse_args()
    ingest_recursive_dicom_folder(args.source)

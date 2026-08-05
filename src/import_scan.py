import os
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

    best_series_uid = None
    for uid, files in series_groups.items():
        desc = series_descriptions.get(uid, "").lower()
        if "ax" in desc or "axial" in desc or "localizer" in desc or "scout" in desc:
            continue
        if "sag" in desc or "cor" in desc or "t1" in desc or "structural" in desc:
            best_series_uid = uid
            break

    if best_series_uid is None:
        valid_uids = [u for u in series_groups.keys() if "ax" not in series_descriptions.get(u, "").lower()]
        if valid_uids:
            best_series_uid = max(valid_uids, key=lambda k: len(series_groups[k]))
        else:
            best_series_uid = max(series_groups, key=lambda k: len(series_groups[k]))

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

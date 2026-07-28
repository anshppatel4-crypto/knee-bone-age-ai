import os
import shutil
import argparse
import pydicom
from collections import defaultdict

def ingest_recursive_dicom_folder(source_root, output_target_dir="data/imported_patient_scan"):
    """Recursively walks through subfolders, groups files by Series ID to isolate 

    the dominant 3D volume, and cleanly ingests it.
    """
    if not os.path.exists(source_root):
        raise FileNotFoundError(f"❌ Source directory path not found: {source_root}")
        
    # Clear old junk out of target directory to prevent mixed-run pollution
    if os.path.exists(output_target_dir):
        shutil.rmtree(output_target_dir)
    os.makedirs(output_target_dir, exist_ok=True)
    
    print(f"🔍 Initializing structural series search across root directory: {source_root}")
    
    # Group filepaths dynamically by their internal Series UID
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

    # Identify the structural sequence by pulling the series containing the most files
    best_series_uid = max(series_groups, key=lambda k: len(series_groups[k]))
    chosen_files = series_groups[best_series_uid]
    desc = series_descriptions.get(best_series_uid, "Unknown Sequence")

    print(f"🎯 Isolated Dominant Series: {desc} ({len(chosen_files)} slices)")

    valid_count = 0
    for file_path in chosen_files:
        try:
            ds = pydicom.dcmread(file_path, force=True)
            if hasattr(ds, 'pixel_array') and ds.pixel_array is not None:
                # Maintain the original file name or an explicit index string
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

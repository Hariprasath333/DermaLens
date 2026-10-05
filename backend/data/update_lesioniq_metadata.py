import csv
import os
import statistics
from pathlib import Path

TRAIN_CSV = r"path/to/layer0_train.csv"
VAL_CSV   = r"path/to/layer0_val.csv"
OUTPUT_DIR = Path(r"path/to/output")
SYNTH_DIR  = Path(r"path/to/synthetic")

def update_real_paths_and_get_stats(csv_path):
    rows = []
    class_stats = {}
    col_types = {}
    with open(csv_path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames
        for row in reader:
            cls = row['class']
            img_id = row['image']
            
            # Record column types based on the first encountered values
            if not col_types:
                for k, v in row.items():
                    if k in ['image', 'class', 'image_path']: continue
                    if v in ['True', 'False']:
                        col_types[k] = 'bool'
                    else:
                        try:
                            float(v)
                            col_types[k] = 'float'
                        except ValueError:
                            col_types[k] = 'str'
            
            new_path = str(OUTPUT_DIR / cls / f"{img_id}.png")
            row['image_path'] = new_path
            rows.append(row)
            
            if 'train' in csv_path:
                if cls not in class_stats:
                    class_stats[cls] = {k: [] for k in fields if k not in ['image', 'class', 'image_path']}
                for k in class_stats[cls]:
                    val = row[k]
                    if val == 'True': num = 1.0
                    elif val == 'False': num = 0.0
                    elif val == '': continue
                    else:
                        try:
                            num = float(val)
                        except ValueError:
                            continue
                    class_stats[cls][k].append(num)
    return rows, fields, class_stats, col_types

print("Processing...")
train_rows, fields, class_stats, col_types = update_real_paths_and_get_stats(TRAIN_CSV)
val_rows, _, _, _ = update_real_paths_and_get_stats(VAL_CSV)

medians = {}
for cls, stats in class_stats.items():
    medians[cls] = {}
    for k, v_list in stats.items():
        if v_list:
            medians[cls][k] = statistics.median(v_list)

synth_count = 0
for cls in ['AK', 'SCC', 'DF', 'VASC']:
    cls_dir = SYNTH_DIR / cls
    synth_files = [p for p in cls_dir.iterdir() if p.is_file() and p.suffix.lower() == '.png']
    
    for p in synth_files:
        new_row = {}
        new_row['image'] = f"SYNTH_{cls}_{p.stem}"
        new_row['class'] = cls
        new_row['image_path'] = str(p)
        
        for k in fields:
            if k in ['image', 'class', 'image_path']:
                continue
            
            med_val = medians[cls].get(k, 0.0)
            
            if col_types.get(k) == 'bool':
                new_row[k] = 'True' if med_val > 0.5 else 'False'
            elif col_types.get(k) == 'float':
                # Preserve int look if it is exactly int, else float. (class_encoded is int)
                if k == 'class_encoded':
                    new_row[k] = str(int(med_val))
                else:
                    new_row[k] = str(round(med_val, 1))
            else:
                new_row[k] = str(med_val)
                
        train_rows.append(new_row)
        synth_count += 1

with open(TRAIN_CSV, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(train_rows)
    
with open(VAL_CSV, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(val_rows)
    
print(f"Updated {len(train_rows) - synth_count} train rows and {len(val_rows)} val rows.")
print(f"Appended {synth_count} synthetic images to train CSV.")

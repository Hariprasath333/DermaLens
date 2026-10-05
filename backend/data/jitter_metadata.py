import csv
import random
import statistics
import math
from collections import defaultdict

TRAIN_CSV = r"path/to/layer0_train.csv"

# Load data
rows = []
with open(TRAIN_CSV, 'r', newline='') as f:
    reader = csv.DictReader(f)
    fields = reader.fieldnames
    for row in reader:
        rows.append(row)

# Parameters for jitter
age_params = {
    'AK': (70, 10),
    'SCC': (65, 12),
    'DF': (45, 12),
    'VASC': (50, 15)
}

site_distributions = {
    'AK':   ['site_head/neck', 'site_upper extremity', 'trunk'],
    'SCC':  ['site_head/neck', 'site_upper extremity', 'site_lower extremity'],
    'DF':   ['site_lower extremity', 'trunk', 'site_upper extremity'],
    'VASC': ['trunk', 'site_lower extremity', 'site_head/neck']
}

def resolve_site(site_str):
    if site_str == 'trunk':
        return random.choice(['site_anterior torso', 'site_posterior torso'])
    return site_str

# Process rows
jittered_count = 0
for row in rows:
    if row['image'].startswith('SYNTH_'):
        cls = row['class']
        
        # 1. Age Jitter
        mean_age, std_age = age_params[cls]
        new_age = random.gauss(mean_age, std_age)
        new_age = max(20, min(90, new_age))
        row['age_approx'] = str(float(round(new_age)))
        
        # 2. Sex Jitter (50/50 male/female)
        is_female = random.choice([True, False])
        row['sex_female'] = 'True' if is_female else 'False'
        row['sex_male'] = 'False' if is_female else 'True'
        row['sex_unknown'] = 'False'
        
        # 3. Site Jitter
        # First, clear all site columns
        for k in fields:
            if k.startswith('site_'):
                row[k] = 'False'
        
        # Pick a site
        chosen_site_general = random.choice(site_distributions[cls])
        chosen_site_specific = resolve_site(chosen_site_general)
        row[chosen_site_specific] = 'True'
        
        jittered_count += 1

# Save back
with open(TRAIN_CSV, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)

print(f"Metadata jitter applied to {jittered_count} synthetic rows.")

# Verify One-Hot Encoding and get counts
real_sites = set()
synth_sites = set()

class_counts = defaultdict(int)

for row in rows:
    # get active site
    active_sites = [k for k in fields if k.startswith('site_') and row[k] == 'True']
    site_val = active_sites[0] if active_sites else 'None'
    
    if row['image'].startswith('SYNTH_'):
        synth_sites.add(site_val)
    else:
        real_sites.add(site_val)
        
    class_counts[row['class']] += 1

unknown = synth_sites - real_sites
if unknown and unknown != {'None'}:
    print(f"🔴 Unknown site categories in synthetic rows: {unknown}")
else:
    print("✅ All synthetic site values match real data vocabulary")

print("\nFinal train counts for all 8 classes:")
for cls in sorted(class_counts.keys()):
    print(f"  {cls}: {class_counts[cls]}")

# Print a small sample of jittered data
print("\nSample of jittered synthetic data (first 5 rows):")
sampled = 0
for row in rows:
    if row['image'].startswith('SYNTH_'):
        active_site = [k for k in fields if k.startswith('site_') and row[k] == 'True']
        site = active_site[0] if active_site else 'None'
        print(f"  {row['class']} | Age: {row['age_approx']} | Female: {row['sex_female']} | Male: {row['sex_male']} | Site: {site}")
        sampled += 1
        if sampled >= 5:
            break

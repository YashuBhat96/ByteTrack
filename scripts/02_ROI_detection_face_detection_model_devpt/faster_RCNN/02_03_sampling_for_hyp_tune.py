import os
import shutil
import random
from concurrent.futures import ProcessPoolExecutor, as_completed

def collect_image_annotation_pairs(root_dir):
    subject_paths = {}

    for subdir, _, files in os.walk(root_dir):
        for file in files:
            if file.lower().endswith('.jpg'):
                full_image_path = os.path.join(subdir, file)
                annotation_path = full_image_path.replace('.jpg', '.xml')
                if os.path.exists(annotation_path):
                    subject = os.path.basename(subdir)  # Assuming subdir is the subject identifier
                    if subject not in subject_paths:
                        subject_paths[subject] = []
                    subject_paths[subject].append((full_image_path, annotation_path))

    return subject_paths

def copy_files(src_dst_paths):
    for src, dst in src_dst_paths:
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy(src, dst)

def create_subset(root_dir, subset_dir, subjects_to_sample, num_workers=4):
    if not os.path.exists(subset_dir):
        os.makedirs(subset_dir)

    subject_paths = collect_image_annotation_pairs(root_dir)
    sampled_paths = []

    for subject in subjects_to_sample:
        if subject in subject_paths:  # Check if subject exists in the dataset
            sampled_paths.extend(subject_paths[subject])

    image_src_dst_paths = [(img_path, os.path.join(subset_dir, os.path.relpath(img_path, root_dir))) for img_path, _ in sampled_paths]
    annotation_src_dst_paths = [(ann_path, os.path.join(subset_dir, os.path.relpath(ann_path, root_dir))) for _, ann_path in sampled_paths]

    combined_src_dst_paths = image_src_dst_paths + annotation_src_dst_paths

    chunk_size = len(combined_src_dst_paths) // num_workers
    chunks = [combined_src_dst_paths[i:i + chunk_size] for i in range(0, len(combined_src_dst_paths), chunk_size)]

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(copy_files, chunk) for chunk in chunks]

        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"Error copying files: {e}")

    print(f"Subset created with subjects {len(subjects_to_sample)} in {subset_dir}")

# Paths to your directories
train_root_dir = '/storage/group/klk37/default/homebytes/video/fbs/split_resized/train'
train_subset_dir = '/storage/group/klk37/default/homebytes/video/fbs/split_resized/train_sample'

train_trans_root_dir = '/storage/group/klk37/default/homebytes/video/fbs/split_resized/train_trans'
train_trans_subset_dir = '/storage/group/klk37/default/homebytes/video/fbs/split_resized/train_trans_sample'

val_root_dir = '/storage/group/klk37/default/homebytes/video/fbs/split_resized/val'
val_subset_dir = '/storage/group/klk37/default/homebytes/video/fbs/split_resized/val_sample'

# Collect subjects from the train dataset
train_subject_paths = collect_image_annotation_pairs(train_root_dir)
subjects = list(train_subject_paths.keys())
num_samples = int(len(subjects) * 0.2)

# Randomly sample 20% of subjects from train and apply to train_trans
sampled_subjects = random.sample(subjects, num_samples)

# Create subsets for train and train_trans
create_subset(train_root_dir, train_subset_dir, sampled_subjects, num_workers=4)
create_subset(train_trans_root_dir, train_trans_subset_dir, sampled_subjects, num_workers=4)

# For validation, we can sample separately since they don't have to match train subjects
val_subject_paths = collect_image_annotation_pairs(val_root_dir)
val_subjects = list(val_subject_paths.keys())
val_num_samples = int(len(val_subjects) * 0.2)

sampled_val_subjects = random.sample(val_subjects, val_num_samples)
create_subset(val_root_dir, val_subset_dir, sampled_val_subjects, num_workers=4)

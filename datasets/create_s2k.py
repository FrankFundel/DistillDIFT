# Script to create the S2K (Semantic and Spatial Knowledge) Dataset
# It is based on the PASCAL-Part dataset, which is based on the original PASCAL dataset
# Create new benchmark dataset by combining images that maximize [number of same parts]

# Instructions:
# Only use image with a single object
# Only use parts that occur more than once
# Only use parts that occur in both images

# Also match related classes e.g.: bird and aeroplane, car and bus, cat and cow/dog/horse/sheep/person, bicycle and motorbike
# Headlights, wheels and windows are not labeled spatially, but could be artifically if they are in touch in e.g. left side, left wing, etc.
# Exclude tvmonitor, diningtable, pottedplant, sofa, boat, bottle and chair

# During sampling, the images and masks are read
# The masks are re-stored in np format (with renamed and artificially labeled parts)
# The sample information is stored in a .json

import os
import cv2
import json
import tqdm
import scipy.io
import numpy as np

dataset_directory = "/export/home/ffundel/DistillDIFT/data/PASCAL-Part"
annotation_directory = os.path.join(dataset_directory, "Annotations_Part")

def load_annotations(path):
    annotations = scipy.io.loadmat(path)["anno"]
    objects = annotations[0, 0]["objects"]
    objects_list = []

    for obj_idx in range(objects.shape[1]):
        obj = objects[0, obj_idx]

        classname = obj["class"][0]
        mask = obj["mask"]

        parts_list = []
        parts = obj["parts"]
        for part_idx in range(parts.shape[1]):
            part = parts[0, part_idx]
            part_class = part["part_name"][0]
            part_mask = part["mask"]
            parts_list.append({"class": part_class, "mask": part_mask})

        objects_list.append({"class": classname, "mask": np.array(mask), "parts": parts_list})

    return objects_list

def relabel_parts(parts):
    # Dilate to include touching pixels
    for part in parts:
        # rename parts
        if part['class'] == 'lwing':
            part['class'] = 'left wing'
        elif part['class'] == 'rwing':
            part['class'] = 'right wing'
        elif part['class'] == 'fwheel':
            part['class'] = 'front wheel'
        elif part['class'] == 'bwheel':
            part['class'] = 'back wheel'
        elif part['class'] == 'leye':
            part['class'] = 'left eye'
        elif part['class'] == 'reye':
            part['class'] = 'right eye'
        elif part['class'] == 'lear':
            part['class'] = 'left ear'
        elif part['class'] == 'rear':
            part['class'] = 'right ear'
        elif part['class'] == 'lleg':
            part['class'] = 'left leg'
        elif part['class'] == 'rleg':
            part['class'] = 'right leg'
        elif part['class'] == 'lfoot':
            part['class'] = 'left foot'
        elif part['class'] == 'rfoot':
            part['class'] = 'right foot'
        elif part['class'] == 'fliplate':
            part['class'] = 'front license plate'
        elif part['class'] == 'bliplate':
            part['class'] = 'back license plate'
        elif part['class'] == 'lfleg':
            part['class'] = 'left front leg'
        elif part['class'] == 'rfleg':
            part['class'] = 'right front leg'
        elif part['class'] == 'lbleg':
            part['class'] = 'left back leg'
        elif part['class'] == 'rbleg':
            part['class'] = 'right back leg'
        elif part['class'] == 'lfpa':
            part['class'] = 'left front paw'
        elif part['class'] == 'rfpa':
            part['class'] = 'right front paw'
        elif part['class'] == 'lbpa':
            part['class'] = 'left back paw'
        elif part['class'] == 'rbpa':
            part['class'] = 'right back paw'
        elif part['class'] == 'lfuleg':
            part['class'] = 'left front upper leg'
        elif part['class'] == 'rfule':
            part['class'] = 'right front upper leg'
        elif part['class'] == 'lbule':
            part['class'] = 'left back upper leg'
        elif part['class'] == 'rbule':
            part['class'] = 'right back upper leg'
        elif part['class'] == 'lflleg':
            part['class'] = 'left front lower leg'
        elif part['class'] == 'rflleg':
            part['class'] = 'right front lower leg'
        elif part['class'] == 'lblleg':
            part['class'] = 'left back lower leg'
        elif part['class'] == 'rblleg':
            part['class'] = 'right back lower leg'
        elif part['class'] == 'lflpa':
            part['class'] = 'left front lower paw'
        elif part['class'] == 'rflpa':
            part['class'] = 'right front lower paw'
        elif part['class'] == 'lblpa':
            part['class'] = 'left back lower paw'
        elif part['class'] == 'rblpa':
            part['class'] = 'right back lower paw'
        elif part['class'] == 'lhorn':
            part['class'] = 'left horn'
        elif part['class'] == 'rhorn':
            part['class'] = 'right horn'
        elif part['class'] == 'lebrow':
            part['class'] = 'left eyebrow'
        elif part['class'] == 'rebrow':
            part['class'] = 'right eyebrow'
        elif part['class'] == 'llarm':
            part['class'] = 'left lower arm'
        elif part['class'] == 'luarm':
            part['class'] = 'left upper arm'
        elif part['class'] == 'rlarm':
            part['class'] = 'right lower arm'
        elif part['class'] == 'ruarm':
            part['class'] = 'right upper arm'
        elif part['class'] == 'lhand':
            part['class'] = 'left hand'
        elif part['class'] == 'rhand':
            part['class'] = 'right hand'
        elif part['class'] == 'llleg':
            part['class'] = 'left lower leg'
        elif part['class'] == 'luleg':
            part['class'] = 'left upper leg'
        elif part['class'] == 'hfrontside':
            part['class'] = 'front side'
        elif part['class'] == 'hleftside':
            part['class'] = 'left side'
        elif part['class'] == 'hrightside':
            part['class'] = 'right side'
        elif part['class'] == 'hbackside':
            part['class'] = 'back side'
        elif part['class'] == 'hroofside':
            part['class'] = 'roof side'
        elif part['class'].startswith('left'):
            part['class'] = 'left ' + part['class'].replace('left', '')
        elif part['class'].startswith('right'):
            part['class'] = 'right ' + part['class'].replace('right', '')
        elif part['class'].startswith('front'):
            part['class'] = 'front ' + part['class'].replace('front', '')
        elif part['class'].startswith('back'):
            part['class'] = 'back ' + part['class'].replace('back', '')

        if part['class'].split('_')[0] not in ['headlight', 'wheel', 'engine', 'door', 'window'] and not any([x in part['class'].split()[0] for x in ['left', 'right', 'front', 'back']]):
            continue

        part['dilated_mask'] = cv2.dilate(part['mask'], np.ones((5, 5), np.uint8), iterations=1)
    
    for part in parts:
        if part['class'].split('_')[0] not in ['headlight', 'wheel', 'engine', 'door', 'window']:
            continue

        neighbours = []
        for other_part in parts:
            if part["class"] == other_part["class"] or not any([x in other_part['class'].split()[0] for x in ['left', 'right', 'front', 'back']]):
                continue
            if cv2.bitwise_and(part['dilated_mask'], other_part["dilated_mask"]).any():
                neighbours.append(other_part["class"].split()[0])

        if 'left' in neighbours:
            part['class'] = 'left ' + part['class'].split('_')[0]
        elif 'right' in neighbours:
            part['class'] = 'right ' + part['class'].split('_')[0]

        if 'front' in neighbours:
            part['class'] = 'front ' + part['class'].split('_')[0]
        elif 'back' in neighbours:
            part['class'] = 'back ' + part['class'].split('_')[0]

    for part in parts:
        if 'dilated_mask' in part:
            del part['dilated_mask']


classes_to_exclude = ["tvmonitor", "diningtable", "pottedplant", "sofa", "boat", "bottle", "chair"]

images = []
images_by_class = {}

# iterate over .mat files
for annotation_filename in tqdm.tqdm([f for f in os.listdir(annotation_directory) if f.endswith('.mat')]):
    annotations = load_annotations(os.path.join(annotation_directory, annotation_filename))

    # Only use image with a single object
    if len(annotations) > 1:
        continue

    object_annotation = annotations[0]

    # Exclude tvmonitor, diningtable, pottedplant, sofa, boat, bottle and chair
    if object_annotation["class"] in classes_to_exclude:
        continue

    # Rename and artifically label headlights, wheels, engines, doors, and windows
    relabel_parts(object_annotation["parts"])

    # Save new annotation with numpy
    np.save(os.path.join(annotation_directory, annotation_filename.replace(".mat", ".npy")), object_annotation)

    # Get bounding box from object mask in (x, y, w, h)
    mask = object_annotation["mask"]
    y, x = np.where(mask)
    x1, x2 = min(x), max(x)
    y1, y2 = min(y), max(y)
    bbox = [int(x1), int(y1), int(x2 - x1), int(y2 - y1)]

    part_counts = {}
    for part in object_annotation['parts']:
        name = part['class'].split()[-1]
        if name not in part_counts:
            part_counts[name] = 1
        else:
            part_counts[name] += 1
    
    non_unique_parts = []
    for part in object_annotation['parts']:
        name = part['class'].split()[-1]
        if part_counts[name] > 1:
            non_unique_parts.append(part['class'])

    obj_to_add = {
        "image_path": annotation_filename.replace(".mat", ".jpg"),
        "class": object_annotation["class"],
        "bbox": bbox,
        "parts": [part['class'] for part in object_annotation['parts']],
        "non_unique_parts": non_unique_parts
    }

    if object_annotation["class"] not in images_by_class:
        images_by_class[object_annotation["class"]] = []
    images_by_class[object_annotation["class"]].append(obj_to_add)
    images.append(obj_to_add)


# Create new benchmark dataset by combining images that maximize [number of same parts]
# Could also match different classes e.g.: bird and aeroplane, car and bus, cat and cow/dog/horse/sheep/person, bicycle and motorbike

related_classes = {
    "bird": ["aeroplane"],
    "aeroplane": ["bird"],
    "car": ["bus"],
    "bus": ["car"],
    "cat": ["cow", "dog", "horse", "sheep", "person"],
    "cow": ["cat", "dog", "horse", "sheep", "person"],
    "dog": ["cat", "cow", "horse", "sheep", "person"],
    "horse": ["cat", "cow", "dog", "sheep", "person"],
    "sheep": ["cat", "cow", "dog", "horse", "person"],
    "person": ["cat", "cow", "dog", "horse", "sheep"],
    "bicycle": ["motorbike"],
    "motorbike": ["bicycle"]
}

top_k = 10
benchmark_dataset = []
combined_images = {}
for image in tqdm.tqdm(images):
    class_name = image['class']
    potential_images = []
    for related_class in [class_name] + related_classes.get(class_name, []):
        potential_images += images_by_class.get(related_class, [])

    similarities = []
    for pi in potential_images:
        matching_parts = set(pi['non_unique_parts']).intersection(set(image['non_unique_parts']))
        similarities.append(len(matching_parts))

    # Get top-k images where similarity is greater than 0
    top_k_images = [potential_images[i] for i in np.argsort(similarities) if similarities[i] > 0][-top_k:]

    if len(top_k_images) == 0:
        continue

    for matching_image in top_k_images:
        if matching_image['image_path'] == image['image_path']:
            continue
        if (image['image_path'], matching_image['image_path']) in combined_images:
            continue
        
        common_non_unique_parts = set(image['non_unique_parts']) & set(matching_image['non_unique_parts'])

        # Initialize lists to hold matched indices
        matched_source_indices = []
        matched_target_indices = []

        # For each common non-unique part, find its index in both images' parts
        # If a part occurs more than once in one image, only the first occurrence is used
        for part in common_non_unique_parts:
            source_index = image['parts'].index(part)
            target_index = matching_image['parts'].index(part)
            matched_source_indices.append(source_index)
            matched_target_indices.append(target_index)
                
        # Add to benchmark dataset
        # Source parts are the indices parts from the source image, which are also present in the target image
        # Target parts are the indices parts from the target image, which are also present in the source image
        benchmark_dataset.append({
            "source_image_path": image['image_path'],
            "target_image_path": matching_image['image_path'],
            "source_category": class_name,
            "target_category": matching_image['class'],
            "source_bbox": image['bbox'],
            "target_bbox": matching_image['bbox'],
            "source_annotation_path": image['image_path'].replace(".jpg", ".npy"),
            "target_annotation_path": matching_image['image_path'].replace(".jpg", ".npy"),
            "source_parts": matched_source_indices,
            "target_parts": matched_target_indices
        })
        combined_images[(image['image_path'], matching_image['image_path'])] = True
        combined_images[(matching_image['image_path'], image['image_path'])] = True

# Save benchmark dataset
with open(os.path.join(dataset_directory, "s2k.json"), "w") as f:
    json.dump(benchmark_dataset, f)
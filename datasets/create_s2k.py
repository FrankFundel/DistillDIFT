# Script to create the S2K (Semantic and Spatial Knowledge) Dataset
# It is based on the PASCAL-Part dataset, which is based on the original PASCAL dataset
# Create new benchmark dataset by combining images that maximize [number of same parts]

# Instructions:
# Only use image with a single object
# Only use parts that occur more than once
# Only use parts that occur in both images
# Could also match different classes e.g.: bird and aeroplane, car and bus, cat and cow/dog/hose/sheep/person, bicycle and motorbike
# Tyres, headlights, wheels and windows are not labeled spatially, but could be artifically if they are in touch in e.g. left side, left wing, etc.
# Exclude tvmonitor, diningtable, pottedplant, sofa, boat, bottle and chair

# Each sample should have:
# source_image_path
# target_image_path
# source_size
# target_size
# source_category
# target_category
# source_points
# target_points
# source_masks_path
# target_masks_path

# During sampling, the images and masks are read
# The masks are stored in COCO format and pre-filtered
# The sample information is stored in a .json

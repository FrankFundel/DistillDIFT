import matplotlib.pyplot as plt

def display_image_pair(image_pair, show_bbox=False):
    """
    Display an image pair with keypoints and bounding boxes (optional).

    Args:
        image_pair (dict): dictionary containing the source and target images, keypoints and bounding boxes
        show_bbox (bool, optional): whether to show the bounding boxes
    """

    # Get the images, keypoints and bounding boxes from the image pair
    source_image = image_pair['source_image']
    target_image = image_pair['target_image']
    source_points = image_pair['source_points']
    target_points = image_pair['target_points']
    source_bbox = image_pair['source_bbox']
    target_bbox = image_pair['target_bbox']

    fig, ax = plt.subplots()

    # Calculate the offset of the target image (based on the width of the source image)
    offset = source_image.size[0]

    # Draw the images in specific location (left, right, bottom, top)
    ax.imshow(source_image, extent=[0, source_image.size[0], 0, source_image.size[1]])
    ax.imshow(target_image, extent=[offset, offset + target_image.size[0], 0, target_image.size[1]])

    # Get a list of colors from the 'tab20' colormap which has 20 distinct colors
    colors = plt.cm.tab20.colors

    # Draw lines between the keypoints, ensuring the target points are offset correctly
    for i, (sp, tp) in enumerate(zip(source_points, target_points)):
        ax.plot([sp[0], tp[0] + offset], # x-coordinates
                [source_image.size[1] - sp[1], target_image.size[1] - tp[1]], # y-coordinates (inverted)
                color=colors[i % len(colors)]) # colour

    # Draw the bounding boxes if required
    if show_bbox:
        # Extract the coordinates and dimensions of the bounding boxes
        source_x, source_y, source_w, source_h = source_bbox
        target_x, target_y, target_w, target_h = target_bbox

        # Draw the bounding box for the source image
        source_rect = plt.Rectangle((source_x, source_image.size[1] - source_y - source_h),
                                    source_w, source_h,
                                    linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(source_rect)

        # Draw the bounding box for the target image
        target_rect = plt.Rectangle((target_x + offset, target_image.size[1] - target_y - target_h),
                                    target_w, target_h,
                                    linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(target_rect)

    # Disable the axis labels
    ax.axis('off')

    # Show the plot
    plt.show()

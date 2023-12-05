import matplotlib.pyplot as plt

def display_image_pair(image_pair):
    # Get the images, keypoints and bounding boxes (if applicable) from the image pair
    source_image = image_pair['source_image']
    target_image = image_pair['target_image']
    source_points = image_pair['source_points']
    target_points = image_pair['target_points']

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

    # Remove the axis
    ax.axis('off')

    # Show the plot
    plt.show()

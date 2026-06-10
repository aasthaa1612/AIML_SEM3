import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
image_path = 'pokemon.jpg'
img = cv2.imread(image_path)

if img is None:
    print(f"Error: Could not load image from {image_path}")
    print("Please ensure pokemon.jpg is in the current directory")
else:
    # Define the kernel
    kernel = np.ones((5, 5), np.uint8)

    # Apply morphological operations directly on the color image
    erosion = cv2.erode(img, kernel, iterations=1)
    dilation = cv2.dilate(img, kernel, iterations=1)
    opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

    # Save the results
    cv2.imwrite('erosion_color.jpg', erosion)
    cv2.imwrite('dilation_color.jpg', dilation)
    cv2.imwrite('opening_color.jpg', opening)
    cv2.imwrite('closing_color.jpg', closing)

    print("Color morphological operations completed!")
    print("Saved: erosion_color.jpg, dilation_color.jpg, opening_color.jpg, closing_color.jpg")

    # Convert BGR (OpenCV) to RGB (matplotlib)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    erosion_rgb = cv2.cvtColor(erosion, cv2.COLOR_BGR2RGB)
    dilation_rgb = cv2.cvtColor(dilation, cv2.COLOR_BGR2RGB)
    opening_rgb = cv2.cvtColor(opening, cv2.COLOR_BGR2RGB)
    closing_rgb = cv2.cvtColor(closing, cv2.COLOR_BGR2RGB)

    # Display results
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 3, 1), plt.imshow(img_rgb), plt.title('Original'), plt.axis('off')
    plt.subplot(2, 3, 2), plt.imshow(erosion_rgb), plt.title('Erosion (Color)'), plt.axis('off')
    plt.subplot(2, 3, 3), plt.imshow(dilation_rgb), plt.title('Dilation (Color)'), plt.axis('off')
    plt.subplot(2, 3, 4), plt.imshow(opening_rgb), plt.title('Opening (Color)'), plt.axis('off')
    plt.subplot(2, 3, 5), plt.imshow(closing_rgb), plt.title('Closing (Color)'), plt.axis('off')

    plt.subplots_adjust(wspace=0.3, hspace=0.3)
    plt.show()
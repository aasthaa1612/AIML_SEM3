import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
img = cv2.imread('pokemon.jpg')

if img is None:
    print("Error: Could not load pokemon.jpg")
else:
    # Convert BGR → RGB for display
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Convert to HSV & LAB
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    # -------- RGB COMPONENT IMAGES (COLORED) --------
    zeros = np.zeros_like(rgb[:, :, 0])

    r_img = cv2.merge([rgb[:, :, 0], zeros, zeros])
    g_img = cv2.merge([zeros, rgb[:, :, 1], zeros])
    b_img = cv2.merge([zeros, zeros, rgb[:, :, 2]])

    # -------- HSV COMPONENT IMAGES (COLORED) --------
    # H, S, V channels isolated but merged back to HSV then converted to RGB
    
    # H component
    h_only = cv2.merge([hsv[:, :, 0], np.full_like(hsv[:, :, 1], 255), np.full_like(hsv[:, :, 2], 255)])
    h_only = cv2.cvtColor(h_only, cv2.COLOR_HSV2RGB)

    # S component
    s_only = cv2.merge([np.full_like(hsv[:, :, 0], 0), hsv[:, :, 1], np.full_like(hsv[:, :, 2], 255)])
    s_only = cv2.cvtColor(s_only, cv2.COLOR_HSV2RGB)

    # V component
    v_only = cv2.merge([np.full_like(hsv[:, :, 0], 0), np.full_like(hsv[:, :, 1], 0), hsv[:, :, 2]])
    v_only = cv2.cvtColor(v_only, cv2.COLOR_HSV2RGB)

    # -------- LAB COMPONENT IMAGES (COLORED) --------
    # Merge isolated channels back to LAB and convert to RGB

    # L channel
    l_only = cv2.merge([lab[:, :, 0], np.full_like(lab[:, :, 1], 128), np.full_like(lab[:, :, 2], 128)])
    l_only = cv2.cvtColor(l_only, cv2.COLOR_LAB2RGB)

    # A channel
    a_only = cv2.merge([np.full_like(lab[:, :, 0], 128), lab[:, :, 1], np.full_like(lab[:, :, 2], 128)])
    a_only = cv2.cvtColor(a_only, cv2.COLOR_LAB2RGB)

    # B channel
    b_lab_only = cv2.merge([np.full_like(lab[:, :, 0], 128), np.full_like(lab[:, :, 1], 128), lab[:, :, 2]])
    b_lab_only = cv2.cvtColor(b_lab_only, cv2.COLOR_LAB2RGB)

    # ---------------- DISPLAY RESULTS ----------------
    plt.figure(figsize=(14, 12))

    # RGB
    plt.subplot(3, 4, 1); plt.imshow(rgb); plt.title("Original RGB"); plt.axis("off")
    plt.subplot(3, 4, 2); plt.imshow(r_img); plt.title("Red Component"); plt.axis("off")
    plt.subplot(3, 4, 3); plt.imshow(g_img); plt.title("Green Component"); plt.axis("off")
    plt.subplot(3, 4, 4); plt.imshow(b_img); plt.title("Blue Component"); plt.axis("off")

    # HSV
    plt.subplot(3, 4, 5); plt.imshow(cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)); plt.title("HSV Encoded"); plt.axis("off")
    plt.subplot(3, 4, 6); plt.imshow(h_only); plt.title("Hue Component"); plt.axis("off")
    plt.subplot(3, 4, 7); plt.imshow(s_only); plt.title("Saturation Component"); plt.axis("off")
    plt.subplot(3, 4, 8); plt.imshow(v_only); plt.title("Value Component"); plt.axis("off")

    # LAB
    plt.subplot(3, 4, 9); plt.imshow(cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)); plt.title("LAB Encoded"); plt.axis("off")
    plt.subplot(3, 4, 10); plt.imshow(l_only); plt.title("Lightness Component"); plt.axis("off")
    plt.subplot(3, 4, 11); plt.imshow(a_only); plt.title("A Component"); plt.axis("off")
    plt.subplot(3, 4, 12); plt.imshow(b_lab_only); plt.title("B Component"); plt.axis("off")

    # ➤ ADD SPACING BETWEEN IMAGES
    plt.subplots_adjust(wspace=0.4, hspace=0.4)

    plt.show()

    print("Colored output with spacing generated successfully!")

# Make sure you have these imports at the top of your script
import cv2
from PIL import Image
from skimage import io
import matplotlib.pyplot as plt
import numpy as np

# --- Image Loading (Necessary before flipping for each library) ---
# Load an image using Pillow
try:
    img_pil = Image.open('sample_image.jpg')
except FileNotFoundError:
    print("Error: sample_image.jpg not found for Pillow. Please check the path.")
    img_pil = None # Set to None to prevent errors later if file isn't found

# Load an image using OpenCV
try:
    img_cv = cv2.imread('sample_image.jpg')
    if img_cv is None:
        print("Error: Could not load image with OpenCV. Check path and file type.")
    else:
        # Convert BGR to RGB for consistent display with Matplotlib later
        img_cv_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
except Exception as e:
    print(f"Error loading image with OpenCV: {e}")
    img_cv = None
    img_cv_rgb = None

# Load an image using Scikit-image
try:
    img_sk = io.imread('sample_image.jpg')
except FileNotFoundError:
    print("Error: sample_image.jpg not found for Scikit-image. Please check the path.")
    img_sk = None


print(f"Original Pillow Image Size: {img_pil.size if img_pil else 'N/A'}")
print(f"Original OpenCV Image Shape (H,W,C): {img_cv.shape if img_cv is not None else 'N/A'}")
print(f"Original Scikit-image Image Shape (H,W,C): {img_sk.shape if img_sk is not None else 'N/A'}\n")


# --- Flipping (Mirroring) - Combined Display (All at once) ---

# Create a figure and a set of subplots for horizontal flips
fig_h, axes_h = plt.subplots(1, 3, figsize=(18, 6))
fig_h.suptitle('Horizontal Flipping (Left-Right) - All Libraries', fontsize=16)

# Create another figure for vertical flips (optional, but good for separate comparison)
fig_v, axes_v = plt.subplots(1, 3, figsize=(18, 6))
fig_v.suptitle('Vertical Flipping (Top-Bottom) - All Libraries', fontsize=16)


# --- Perform Flips and Populate Subplots ---

# Pillow Flipping
if img_pil:
    img_flipped_lr_pil = img_pil.transpose(Image.FLIP_LEFT_RIGHT)
    img_flipped_tb_pil = img_pil.transpose(Image.FLIP_TOP_BOTTOM)

    axes_h[0].imshow(img_flipped_lr_pil)
    axes_h[0].set_title('Pillow (Left-Right)')
    axes_h[0].axis('off')

    axes_v[0].imshow(img_flipped_tb_pil)
    axes_v[0].set_title('Pillow (Top-Bottom)')
    axes_v[0].axis('off')
else:
    axes_h[0].set_title('Pillow (Image not loaded)')
    axes_h[0].axis('off')
    axes_v[0].set_title('Pillow (Image not loaded)')
    axes_v[0].axis('off')


# OpenCV Flipping
if img_cv_rgb is not None:
    img_flipped_lr_cv = cv2.flip(img_cv_rgb, 1) # 1 for horizontal flip
    img_flipped_tb_cv = cv2.flip(img_cv_rgb, 0) # 0 for vertical flip

    axes_h[1].imshow(img_flipped_lr_cv)
    axes_h[1].set_title('OpenCV (Left-Right, flipCode=1)')
    axes_h[1].axis('off')

    axes_v[1].imshow(img_flipped_tb_cv)
    axes_v[1].set_title('OpenCV (Top-Bottom, flipCode=0)')
    axes_v[1].axis('off')
else:
    axes_h[1].set_title('OpenCV (Image not loaded)')
    axes_h[1].axis('off')
    axes_v[1].set_title('OpenCV (Image not loaded)')
    axes_v[1].axis('off')


# Scikit-image Flipping (using NumPy slicing)
if img_sk is not None:
    img_flipped_lr_sk = img_sk[:, ::-1] # Flip along columns
    img_flipped_tb_sk = img_sk[::-1, :] # Flip along rows

    axes_h[2].imshow(img_flipped_lr_sk)
    axes_h[2].set_title('Scikit-image (NumPy Slice [:, ::-1])')
    axes_h[2].axis('off')

    axes_v[2].imshow(img_flipped_tb_sk)
    axes_v[2].set_title('Scikit-image (NumPy Slice [::-1, :])')
    axes_v[2].axis('off')
else:
    axes_h[2].set_title('Scikit-image (Image not loaded)')
    axes_h[2].axis('off')
    axes_v[2].set_title('Scikit-image (Image not loaded)')
    axes_v[2].axis('off')


# Adjust layouts and show plots
plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout for current figure (fig_v)
plt.show() # Shows the vertical flips figure (fig_v)

# To ensure the first figure (fig_h) is also shown if you have multiple plt.show()
# Or, you can show them one by one if preferred.
# For truly displaying all at once without closing, you'd combine into one mega-figure
# but having two figures (one for horizontal, one for vertical) is clearer.
# If you want ONLY ONE window with original, LR, TB for all libraries, it's more complex.
# For now, this gives two clear comparison windows.
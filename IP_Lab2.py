import cv2
import numpy as np
import matplotlib.pyplot as plt

def display_histogram(image, title="Image Histogram"):
    """
    Calculates and displays the histogram(s) of an image.
    This version assumes a grayscale image for plotting simplicity.
    """
    plt.figure(figsize=(10, 6))
    plt.title(title)
    plt.xlabel("Pixel Value")
    plt.ylabel("Frequency")

    # For grayscale images, we only need to calculate for channel 0
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])
    
    # Use plt.bar for a bar graph instead of plt.plot
    plt.bar(np.arange(256), hist.ravel(), color='black', width=1.0)

    plt.xlim([0, 256])
    plt.grid(True)
    plt.show()

def main():
    # --- Configuration ---
    image_path = 'sample_image.jpg' # <--- IMPORTANT: Replace with your image file name
    
    # --- 1. Load the image ---
    try:
        # Read the image in color, we will convert it to grayscale next
        original_image = cv2.imread(image_path, cv2.IMREAD_COLOR) 

        if original_image is None:
            print(f"Error: Could not load image from {image_path}. Please check the path and file name.")
            print("Make sure 'sample_image.jpg' exists in the same directory as this script, or provide a full path.")
            return

        print(f"Successfully loaded image: {image_path}")
        print(f"Original image dimensions: {original_image.shape}")
        print(f"Original image data type: {original_image.dtype}")

    except Exception as e:
        print(f"An unexpected error occurred while loading the image: {e}")
        return

    # --- 2. Convert to Grayscale ---
    print("Converting image to grayscale...")
    gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    print(f"Grayscale image dimensions: {gray_image.shape}")

    # --- 3. Display the original grayscale image ---
    cv2.imshow('Original Grayscale Image', gray_image)
    cv2.waitKey(0) # Wait indefinitely until a key is pressed
    cv2.destroyAllWindows()

    # --- 4. Display the original grayscale histogram ---
    display_histogram(gray_image, "Original Grayscale Image Histogram")

    # --- 5. Perform Histogram Equalization on the grayscale image (Manual Implementation) ---
    print("Performing histogram equalization on the grayscale image (manual implementation)...")
    
    # Step 1: Calculate the histogram
    # hist has shape (256, 1)
    hist, bins = np.histogram(gray_image.flatten(), 256, [0, 256])

    # Step 2: Calculate the Cumulative Distribution Function (CDF)
    # The cumulative sum of the histogram gives the CDF
    cdf = hist.cumsum()

    # Step 3: Normalize the CDF
    # Normalize the CDF to the range [0, 255]
    # min_cdf is used to handle cases where the lowest pixel value has a count of 0,
    # preventing division by zero and ensuring proper scaling.
    cdf_normalized = (cdf - cdf.min()) * 255 / (cdf.max() - cdf.min())
    
    # Ensure all values are integers (pixel values)
    cdf_normalized = cdf_normalized.astype('uint8')

    # Step 4: Remap pixel values using the normalized CDF as a lookup table
    # This creates the equalized image by applying the transformation
    equalized_gray_image = cdf_normalized[gray_image]

    # --- 6. Display the equalized grayscale image ---
    cv2.imshow('Equalized Grayscale Image (Manual)', equalized_gray_image)
    cv2.waitKey(0) # Wait indefinitely until a key is pressed
    cv2.destroyAllWindows()

    # --- 7. Display the equalized grayscale histogram ---
    display_histogram(equalized_gray_image, "Equalized Grayscale Image Histogram (Manual)")

    print("\nProgram finished. Close all image windows to exit.")

if __name__ == "__main__":
    main()
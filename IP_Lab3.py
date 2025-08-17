from PIL import Image
import numpy as np

# Load the image
def load_image(path):
    return Image.open(path).convert('L') # Convert to grayscale

# Save the image
def save_image(image_array, path):
    img = Image.fromarray(np.uint8(image_array))
    img.save(path)

# Add salt-and-pepper noise to an image
def add_noise(image_array, prob):
    output = np.copy(image_array)
    num_salt = np.ceil(prob * image_array.size * 0.5)
    coords_salt = [np.random.randint(0, i - 1, int(num_salt)) for i in image_array.shape]
    output[coords_salt[0], coords_salt[1]] = 255
    
    num_pepper = np.ceil(prob * image_array.size * 0.5)
    coords_pepper = [np.random.randint(0, i - 1, int(num_pepper)) for i in image_array.shape]
    output[coords_pepper[0], coords_pepper[1]] = 0
    
    return output

# Mean (Linear) Filter
def mean_filter(image_array, kernel_size):
    # Pad the image to handle borders
    pad_size = kernel_size // 2
    padded_image = np.pad(image_array, pad_size, mode='edge')
    output_image = np.zeros_like(image_array)
    
    # Apply the filter
    for i in range(image_array.shape[0]):
        for j in range(image_array.shape[1]):
            # Extract the neighborhood
            neighborhood = padded_image[i:i+kernel_size, j:j+kernel_size]
            # Calculate the mean
            output_image[i, j] = np.mean(neighborhood)
            
    return output_image

# Median (Non-Linear) Filter
def median_filter(image_array, kernel_size):
    # Pad the image to handle borders
    pad_size = kernel_size // 2
    padded_image = np.pad(image_array, pad_size, mode='edge')
    output_image = np.zeros_like(image_array)
    
    # Apply the filter
    for i in range(image_array.shape[0]):
        for j in range(image_array.shape[1]):
            # Extract the neighborhood
            neighborhood = padded_image[i:i+kernel_size, j:j+kernel_size]
            # Calculate the median
            output_image[i, j] = np.median(neighborhood)
            
    return output_image

# --- Main experiment script ---
if __name__ == '__main__':
    image_path = "sample_image.jpg"
    
    # 1. Load original image
    original_image = load_image(image_path)
    original_array = np.array(original_image)
    print("Original image loaded.")
    
    # 2. Add noise
    noisy_array = add_noise(original_array, 0.05) # 5% noise
    save_image(noisy_array, "noisy_image.jpg")
    print("Noisy image saved as 'noisy_image.jpg'.")
    
    # 3. Perform mean filtering (Linear)
    mean_filtered_array = mean_filter(noisy_array, kernel_size=3)
    save_image(mean_filtered_array, "mean_filtered_image.jpg")
    print("Mean filtered image saved as 'mean_filtered_image.jpg'.")
    
    # 4. Perform median filtering (Non-Linear)
    median_filtered_array = median_filter(noisy_array, kernel_size=3)
    save_image(median_filtered_array, "median_filtered_image.jpg")
    print("Median filtered image saved as 'median_filtered_image.jpg'.")
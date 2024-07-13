import cv2
import os

def preprocess_images(input_dir, output_dir):
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Create CLAHE object
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    # Iterate over the input images
    for filename in os.listdir(input_dir):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)

            # Load the image
            image = cv2.imread(input_path)
            
            resized_image = cv2.resize(image, (256, 256))

            # Normalize pixel values to [0, 1]
            normalized_image = resized_image.astype(float) / 255.0
            
            # Convert the image to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Apply CLAHE to the grayscale image
            clahe_image = clahe.apply(gray)

            # Save the preprocessed image
            cv2.imwrite(output_path, clahe_image)

            print(f"Preprocessed image saved: {filename}")

    print("Image preprocessing complete.")

# Example usage
input_directory = "C:/Users/pouru/Desktop/Augumented(C)"
output_directory = "C:/Users/pouru/Desktop/CLAHE(C)"

preprocess_images(input_directory, output_directory)

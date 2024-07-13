import imgaug as ia
from imgaug import augmenters as iaa
import cv2
import os

def augment_images(input_dir, output_dir, augmentation_factor):
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Create the augmentation sequence
    seq = iaa.Sequential([
        iaa.Fliplr(0.5),  # flip horizontally with 50% probability
        iaa.Affine(rotate=(-25, 25)),  # rotate by -25 to 25 degrees
        iaa.GaussianBlur(sigma=(0, 1.0)),  # apply gaussian blur with sigma between 0 and 1.0
        iaa.AdditiveGaussianNoise(scale=(0, 0.05*255)),  # add Gaussian noise
        iaa.Multiply((0.8, 1.2))  # multiply pixel values by random value between 0.8 and 1.2
    ])

    # Iterate over the input images
    for filename in os.listdir(input_dir):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename.split(".")[0])  # remove extension from filename

            # Load the image
            image = cv2.imread(input_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # convert BGR to RGB

            # Apply augmentation multiple times
            augmented_count = 0
            while augmented_count < augmentation_factor:
                augmented_image = seq.augment_image(image)

                # Save augmented image
                output_filename = f"{output_path}_augmented_{augmented_count}.jpg"
                output_file_path = os.path.join(output_dir, output_filename)
                cv2.imwrite(output_file_path, augmented_image)

                augmented_count += 1

            print(f"Augmented images saved for {filename}")

    print("Image augmentation complete.")

# Example usage
input_directory = "C:/Users/pouru/Desktop/soil/Yellow Soil"
output_directory = "C:/Users/pouru/Desktop/largerOne/Agumented_Yellowsoil"
augmentation_factor = 100  # number of augmented images to generate per input image
augment_images(input_directory, output_directory, augmentation_factor)
# Soil-Image-Classification-for-Pulses-Cultivation

# Overview
This project aims to assist farmers in selecting the optimal soil for pulse cultivation through image analysis. By integrating pre-processing techniques and Gray Level Co-occurrence Matrix (GLCM) feature extraction, we classify soil types and achieve an impressive 97% accuracy using a deep learning neural network model. The result is a system that enhances yields by ensuring the best soil is chosen for cultivating pulses.

# Features
1): Soil Type Classification: Utilizes image analysis to determine the best soil type for pulse cultivation.\newline
2): High Accuracy: Achieved 97% accuracy in classification.
3): User-Friendly Interface: Developed a graphical user interface (GUI) using Python Tkinter.
4): Deep Learning Integration: Employed Convolutional Neural Networks (CNN) for robust image analysis.
5): Feature Extraction: Implemented GLCM for extracting significant features from soil images.

# Technologies Used
1): Python: The primary programming language used for development.
2): Tkinter: Used for GUI development.
3): Convolutional Neural Network (CNN): For image analysis and classification.
4): Gray Level Co-occurrence Matrix (GLCM): For feature extraction.
5): Machine Learning/Deep Learning Libraries: Various Python libraries such as TensorFlow, Keras, NumPy, and OpenCV.

# Project Structure
1): data/: Contains the dataset of soil images.
2): src/: Contains the source code for the project.
3): preprocessing.py: Code for image pre-processing.
4): feature_extraction.py: Code for GLCM feature extraction.
5): model.py: Code for building and training the CNN model.
6): gui.py: Code for the Tkinter GUI.
7): models/: Contains the trained CNN model.
8): notebooks/: Jupyter notebooks for exploratory data analysis and model training.
9): README.md: Project documentation.

# Installation

1): Clone the repository:
  git clone https://github.com/yourusername/soil-image-classification.git
  cd soil-image-classification
2): Create a virtual environment:
  python -m venv venv
  source venv/bin/activate   # On Windows, use `venv\Scripts\activate`
3): Install the required packages:
  pip install -r requirements.txt

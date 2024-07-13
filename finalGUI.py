import tkinter as tk
from tkinter import filedialog
from PIL import ImageTk, Image
import shutil
import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model
from tkinter import scrolledtext

# Load the pre-trained model
model = load_model("C:\\Users\\pouru\\Desktop\\Soil Classification\\saved_model.h5")

def extract_features(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    ymin, ymax, xmin, xmax = h // 3, h * 2 // 3, w // 3, w * 2 // 3
    crop = gray[ymin:ymax, xmin:xmax]
    resize = cv2.resize(crop, (100, 100))
    glcm = graycomatrix(resize, distances=[5], angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4], levels=256, symmetric=True, normed=True)
    properties = ['correlation', 'homogeneity', 'contrast']
    glcm_props = [propery for name in properties for propery in graycoprops(glcm, name)[0]]
    return np.array(glcm_props)

def process_image(image_path):
    global scaler
    features = extract_features(image_path)
    features = features.reshape(1, -1)
    features = scaler.transform(features)
    prediction = model.predict(features)
    predicted_class_index = np.argmax(prediction)
    predicted_class = label_encoder[predicted_class_index]
    return predicted_class

def upload_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.jpeg *.png")])
    if file_path:
        shutil.copy(file_path, "uploaded_image.jpg")
        display_uploaded_image()

        predicted_class = process_image(file_path)
        
        content = suitable_ranges.get(predicted_class, "")
        
        if predicted_class == 0:
            content_label.config(text=f"Predicted Soil Type: Clay Soil", fg="white")
            
        elif predicted_class == 1:
            content_label.config(text=f"Predicted Soil Type: Sandy Loam Soil", fg="white")
            
        else:
            content_label.config(text=f"Predicted Soil Type: Sandy Soil", fg="white")
        
        content_text.delete(1.0, tk.END)
        content_text.insert(tk.END, suitable_ranges.get(predicted_class, ""))

        upload_label.config(text="Image uploaded successfully!", fg="white")



def display_uploaded_image():
    image_path = "uploaded_image.jpg"
    image = Image.open(image_path)
    image = image.resize((200, 200))
    photo = ImageTk.PhotoImage(image)
    image_label.configure(image=photo)
    image_label.image = photo

glcm_df = pd.read_csv("C:\\Users\\pouru\\Desktop\\Soil Classification\\df_csv.csv")
label_encoder = glcm_df['label'].unique()
label_encoder.sort()
label_encoder = {idx: label for idx, label in enumerate(label_encoder)}

scaler = StandardScaler()
X_train = glcm_df.drop(['Unnamed: 0', 'label'], axis=1)
scaler.fit(X_train)

# Create the tkinter window
window = tk.Tk()
window.title("Soil Image Classifier")
window.geometry("800x900")
window.configure(bg="#222222")

logo_image = Image.open("C:\\Users\\pouru\\Desktop\\Soil Classification\\logo.jpeg")
logo_image = logo_image.resize((150, 150))  # Adjust the size of the logo as per your requirement
logo_photo = ImageTk.PhotoImage(logo_image)
logo_frame = tk.Frame(window, bg="#222222")
logo_frame.pack()

# Create a label to display the logo at the top-center
logo_label = tk.Label(logo_frame, image=logo_photo, bg="#222222")
logo_label.pack()

# Create a frame for the heading
heading_frame = tk.Frame(window, bg="#222222")
heading_frame.pack(pady=10)

# Create the heading label with larger font size
heading_label = tk.Label(heading_frame, text="Soil Image Classifier", font=("Arial", 20, "bold"), fg="white", bg="#222222")
heading_label.pack()

# Create a frame for styling
style_frame = tk.Frame(window, bg="#222222")
style_frame.pack(pady=20)

# Create a frame for displaying image and labels on the left side
content_frame = tk.Frame(window, bg="#111111")
content_frame.pack(padx=50, pady=40, side=tk.LEFT)

# Create the upload button on the left side, above the image display area
upload_button = tk.Button(    
    content_frame,
    text="Upload Image",
    command=upload_image,
    padx=10,
    pady=5,
    width=25,  # Increase the width of the button
    height=2,  # Increase the height of the button
    bg="#4287f5",
    fg="white",
    bd=0,
    relief=tk.FLAT,
    highlightthickness=0
)

upload_button.pack(side=tk.TOP, padx=40, pady=20)

# Create a frame to hold the uploaded image and the success label
uploaded_image_frame = tk.Frame(content_frame, bg="#111111")
uploaded_image_frame.pack(side=tk.TOP, padx=70, pady=30)

# Create a label to display the uploaded image on the left side
image_label = tk.Label(uploaded_image_frame, bg="#111111")
image_label.pack()

# Create a frame for displaying the prediction result on the right side
result_frame = tk.Frame(window, bg="#111111")
result_frame.pack(padx=40, pady=40, side=tk.RIGHT)

# Create a label to display upload status
upload_label = tk.Label(uploaded_image_frame, text="", font=('Arial', 12), bg="#111111", fg="white")
upload_label.pack()

# Create a label to display the prediction result
content_label = tk.Label(result_frame, text="Upload an image to classify the soil.", font=('Arial', 12), bg="#111111", fg="white")
content_label.pack()

# Create a scrolled text widget to display the suitable ranges for pulses
content_text = scrolledtext.ScrolledText(result_frame, wrap=tk.WORD, width=80, height=80, bg="#111111", fg="white", font=("Courier New", 12))
content_text.pack()


suitable_ranges = {
    0: """Predicted Soil Type: Clay Soil
Expected Location: Acharya Narendra Dev University Campus, Ayodhya, U.P
Nitrogen Content: 128 kg/ha
Organic Carbon: 65%
pH value: 9.12
Electrical Conductivity: 0.2 dSm-1

        SUITABLE RANGES FOR PULSES:

            pH Value:
                6.5-7.5(Excellent)
                7.5-8.5(Moderate)
                >8(Poor)

            Nitrogen Content (kg/ha):
                0-30 (Low availability)
                30-60(Moderate availability)
                60-100(High availability)
                >100(Very High availability)

            Organic Matter Content (%):
                0-2 (Low availability)
                2-5(Moderate availability)
                >5(High Availability)
                
                
This soil is not recommended for the cultivation of pulses due to the:
    1): Extermly low Nitrogen content.
    2): High ph value(more alkaline).
    
Other crops recommendation: Wheat, Potatoes, Cabbage, Clover etc""",
    1: """Predicted Soil Type: Sandy Loam Soil
Expected Location: Asalatpura, Moradabad
Nitrogen Content: 358.2 kg/ha
Organic Carbon: 79.6%
pH value: 7.27
Electrical Conductivity: 0.02 dSm-1

        SUITABLE RANGES FOR PULSES:

            pH Value:
                6.5-7.5(Excellent)
                7.5-8.5(Moderate)
                >8(Poor)
            
            Nitrogen Content (kg/ha):
                0-30 (Low availability)
                30-60(Moderate availability)
                60-100(High availability)
                >100(Very High availability)
            
            Organic Matter Content (%):
                0-2 (Low availability)
                2-5(Moderate availability)
                >5(High Availability)
                

Based on the above ranges this soil is highly recommeded for the cultivation of Pulses, such as lentils, chickpeas, and beans due to the
    1): Moderate availablity of Nitrogen content.
    2): Excellent Ph value.
    3): High availabilty of Organic Matter
Other crops recommendation:
    1): tomatoes, cucumbers, peppers, carrots, lettuce, spinach, zucchini, and radishes.
    2): watermelons, cantaloupes, blueberries, raspberries, and peaches.
    3): Various herbs, such as basil, cilantro, oregano, thyme, and rosemary.""",

    2: """Predicted Soil Type: Sandy Soil
Expected Location: Village Amroha, Mohanpura-District-Samli
Nitrogen Content: 293.88 kg/ha
Organic Carbon: 9%
pH value: 7.23
Electrical Conductivity: 1.56 dSm-1

        SUITABLE RANGES FOR PULSES:

            pH Value:
                6.5-7.5(Excellent)
                7.5-8.5(Moderate)
                >8(Poor)

            Nitrogen Content (kg/ha):
                0-30 (Low availability)
                30-60(Moderate availability)
                60-100(High availability)
                >100(Very High availability)

            Organic Matter Content (%):
                0-2 (Low availability)
                2-5(Moderate availability)
                >5(High Availability)
                
This soil is not recommended for the cultivation of pulses due to the:
    1): Extermly low organic matter content.
    2): Excessive Nitrogen content
Other suitable crops are: Peanuts, Corn, Broccoli and Cauliflower etc"""
}

# Create a frame for displaying the developer's information at the bottom-left corner
developer_frame = tk.Frame(window, bg="#222222")
developer_frame.pack()

# Create a label for the developer's information with typewriter font and a slightly larger size
developer_label = tk.Label(developer_frame, text="Developed by Pourush Gupta, 2nd Year BTech-AI student", font=('Courier New', 10), fg="white", bg="#222222")
developer_label.pack(padx=5, pady=5)

# Place the developer's frame at the bottom-left corner of the window
developer_frame.place(x=10, y=window.winfo_height() - developer_frame.winfo_height() - 10)

window.mainloop()

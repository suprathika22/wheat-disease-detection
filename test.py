from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

model = load_model("wheat_disease_model.h5")

image_list = []
counter = 1

for folder in ["Healthy", "Diseased"]:
    folder_path = os.path.join("dataset", folder)
    for img in os.listdir(folder_path):
        image_list.append((f"Image{counter}", os.path.join(folder_path, img)))
        counter += 1

print("\nSelect an image to test:")
for idx, (img_name, _) in enumerate(image_list, start=1):
    print(f"{idx}. {img_name}")

choice = input("Enter your choice: ")

if not choice.isdigit() or int(choice) not in range(1, len(image_list) + 1):
    print("❌ Invalid choice")
    exit()

img_name, img_path = image_list[int(choice) - 1]

img = image.load_img(img_path, target_size=(224, 224))
img = image.img_to_array(img) / 255.0
img = np.expand_dims(img, axis=0)

prediction = model.predict(img)[0][0]
predicted_label = "Healthy" if prediction >= 0.5 else "Diseased"

if predicted_label == "Healthy":
    recommendation = "No pesticide required. Maintain crop hygiene."
else:
    recommendation = "Apply broad-spectrum fungicide as per agricultural guidelines."

print("\n Selected Image:", img_name)
print("🌾 Predicted Status:", predicted_label)
print("💊 Recommendation:", recommendation)

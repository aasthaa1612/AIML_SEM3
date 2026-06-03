
from PIL import Image
import os

input_folder = "5images"            
output_folder = "grayscale_images"  

if not os.path.exists(output_folder):
    os.makedirs(output_folder)
for filename in os.listdir(input_folder):
    if filename.lower().endswith((".jpg", ".jpeg", ".png")):
       
        img_path = os.path.join(input_folder, filename)
        image = Image.open(img_path)
        gray_image = image.convert("L")
        output_path = os.path.join(output_folder, f"gray_{filename}")
        gray_image.save(output_path)

        print(f"Converted and saved: {output_path}")

print("✅ All images converted to grayscale and saved in the 'grayscale_images' folder.")

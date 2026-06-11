
from PIL import Image
image = Image.open("Grapevine_Leaves_Image_Dataset/Ak/Ak (7).png")
gray_image = image.convert("L")
gray_image.save("grayscale_image.jpg")
image.show(title="Original Image")
gray_image.show(title="Grayscale Image")

print("✅ Grayscale image saved successfully as 'grayscale_image.jpg'")

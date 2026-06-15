from PIL import Image, ImageFilter
import os
import matplotlib.pyplot as plt

# Use the uploaded image file
input_filename = "Buzgulu (2).png"

# Output folders
destination_folder = "grayscale"
histogram_folder = "histograms"
sharpen_folder = "sharpened"
smooth_folder = "smoothed"
segment_folder = "segmented"

folders = [destination_folder, histogram_folder, sharpen_folder, smooth_folder, segment_folder]

for folder in folders:
    if not os.path.exists(folder):
        os.makedirs(folder)
        print(f"Created directory: {folder}")

base_filename = os.path.splitext(os.path.basename(input_filename))[0]

print(f"\nStarting processing for: {input_filename}")

try:
    if not os.path.exists(input_filename):
        print(f"Error: File not found at {os.path.abspath(input_filename)}")
    else:
        with Image.open(input_filename) as image:
            
            # Convert to grayscale
            gray_image = image.convert("L")
            gray_output_path = os.path.join(destination_folder, base_filename + "_gray.png")
            gray_image.save(gray_output_path)
            print(f"   Saved grayscale: {gray_output_path}")

            # Create and save histogram
            hist_output_path = os.path.join(histogram_folder, base_filename + "_hist.png")
            hist_data = gray_image.histogram()
            
            plt.figure(figsize=(10, 5))
            plt.bar(range(256), hist_data, width=1.0, color='gray')
            plt.title(f"Grayscale Histogram for {base_filename}")
            plt.xlabel("Pixel Intensity (0-255)")
            plt.ylabel("Pixel Count")
            plt.grid(axis='y', alpha=0.75)
            plt.savefig(hist_output_path)
            plt.close()
            print(f"   Histogram saved: {hist_output_path}")

            # Sharpened image
            sharp_image = gray_image.filter(ImageFilter.SHARPEN)
            sharp_output_path = os.path.join(sharpen_folder, base_filename + "_sharp.png")
            sharp_image.save(sharp_output_path)
            print(f"   Saved sharpened: {sharp_output_path}")

            # Smoothed (Gaussian blur) image
            smooth_image = gray_image.filter(ImageFilter.GaussianBlur(radius=2))
            smooth_output_path = os.path.join(smooth_folder, base_filename + "_smooth.png")
            smooth_image.save(smooth_output_path)
            print(f"   Saved smoothed: {smooth_output_path}")

            # Simple segmentation (threshold)
            segment_image = gray_image.point(lambda p: 255 if p > 128 else 0, '1')
            segment_output_path = os.path.join(segment_folder, base_filename + "_segment.png")
            segment_image.save(segment_output_path)
            print(f"   Saved segmented: {segment_output_path}")

        print("\nProcessing complete! ✅")

except Exception as e:
    print(f"   Error processing {input_filename}: {e}")
from PIL import Image
import os

dataset_dir = "PetImages"

removed = 0
converted = 0

for category in ["Cat", "Dog"]:

    folder = os.path.join(dataset_dir, category)

    for file in os.listdir(folder):

        path = os.path.join(folder, file)

        try:
            img = Image.open(path)

            # Get number of channels
            channels = len(img.getbands())

            # Remove invalid channel images
            if channels not in [1, 3, 4]:
                print(f"REMOVED (Invalid Channels): {path}")
                os.remove(path)
                removed += 1
                continue

            # Convert everything to RGB
            rgb_img = img.convert("RGB")
            rgb_img.save(path)

            converted += 1

        except Exception as e:

            print(f"REMOVED (Corrupt): {path}")

            try:
                os.remove(path)
                removed += 1
            except:
                pass

print("\n" + "="*50)
print(f"Images Converted to RGB : {converted}")
print(f"Images Removed         : {removed}")
print("="*50)
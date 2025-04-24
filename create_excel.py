import os
import pandas as pd
cat_folder = r"images/PetImages/Cat"
dog_folder = r"images/PetImages/Dog"
data = []
def scan_folder(folder_path, label):
    for file in os.listdir(folder_path):
        # Check for common image file extensions
        if file.lower().endswith((".jpg", ".jpeg", ".png")):
            file_path = folder_path+"/"+file
            data.append({"image_path": file_path, "label": label})
scan_folder(cat_folder, "Cat")
scan_folder(dog_folder, "Dog")
df = pd.DataFrame(data)
output_file = "cats_and_dogs_dataset.xlsx"
df.to_excel(output_file, index=False)
print(f"Dataset Excel file created: {output_file}")

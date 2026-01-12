import os
import json
from pathlib import Path

# Get all images from translations folder
translations_folder = Path(__file__).parent.parent / 'translations'
image_extensions = ['.png', '.jpg', '.jpeg']

# Get all image files
image_files = []
if translations_folder.exists():
    for file in sorted(translations_folder.iterdir()):
        if file.suffix.lower() in image_extensions:
            image_files.append(file.name)

print(f"Found {len(image_files)} images in translations folder")

# Create images.json file
json_path = Path(__file__).parent / 'images.json'
with open(json_path, 'w') as f:
    json.dump(image_files, f, indent=2)

print(f"Generated {json_path}")
print(f"Images: {image_files}")

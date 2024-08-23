import os
import shutil

def organize_validation_images(annotations_file, val_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    with open(annotations_file, 'r') as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()
        if len(parts) >= 2:
            image_name = parts[0]
            class_label = parts[1]  

            class_folder = os.path.join(output_folder, class_label)
            if not os.path.exists(class_folder):
                os.makedirs(class_folder)

            src_path = os.path.join(val_folder, image_name)
            dst_path = os.path.join(class_folder, image_name)
            
            if os.path.exists(src_path):
                shutil.move(src_path, dst_path)
            else:
                print(f"Warning: Image {image_name} not found in {val_folder}")

    print("Organization complete!")


annotations_file = '/path/to/the/annotation/file/val_annotations.txt'
val_folder = '/path/to/the/validation/folder/tiny-imagenet-200/val/images'
output_folder = '/path/to/the/new/validation/folder/tiny-imagenet-200/val/organisedVal'

organize_validation_images(annotations_file, val_folder, output_folder)
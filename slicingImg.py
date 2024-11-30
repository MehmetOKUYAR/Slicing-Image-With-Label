import cv2
import numpy as np
import os
import random
from datetime import datetime
from pathlib import Path
import pandas as pd
import argparse
from glob import glob
from tqdm import tqdm
import os


class SlicingImage():
    def __init__(self, crop_size=512, save_path="crop_images"):
        self.crop_size = int(crop_size)
        self.save_path = save_path

        self.photo_dir = Path(self.save_path)
        self.photo_dir.mkdir(parents=True, exist_ok=True)


    def boxesFromYOLO(self,imagePath,labelPath):
        
        image = cv2.imread(imagePath)
        #print(image.shape)
        (hI, wI) = image.shape[:2]
        lines = [line.rstrip('\n') for line in open(labelPath)]
        #if(len(objects)<1):
        #    raise Exception("The xml should contain at least one object")
        boxes = []
        if lines != ['']:
            for line in lines:
                components = line.split(" ")
                category = components[0]
                x_min  = int(float(components[1])*wI - float(components[3])*wI/2)
                y_min = int(float(components[2])*hI - float(components[4])*hI/2)
                y_max = int(float(components[4])*hI) + y_min
                x_max = int(float(components[3])*wI) + x_min
                boxes.append((category, x_min, y_min, x_max, y_max))
        return (image,boxes)


    def showBoxes(self,image,boxes):
        cloneImg = image.copy()
        for box in boxes:
            
            (category, x, y, w, h)=box
    

            cv2.rectangle(cloneImg,(x,y),(w,h),(255,80,134),1)
            
            cv2.putText(cloneImg, str(category), (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,80,134), 3)
          
        cv2.imshow("Image",cloneImg)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


    #======== tespit edilen görüntüyü ve etiket dosyasını  kayıt ediyor  ===================    
    def save_img_txt(self, image, yolo_data):
        """
        Save the cropped image and its YOLO annotation data to disk.

        Parameters:
            image (numpy.ndarray): The image to save.
            yolo_data (list): A list of YOLO-format bounding box annotations.
        """
        
        try:
            # Benzersiz bir isim oluştur
            timestamp = datetime.now().strftime("%d.%m.%Y_%H.%M.%S.%f")
            image_name = f'crop_img_{timestamp}'
            image_path = os.path.join(self.photo_dir, image_name + ".jpg")
            label_path = os.path.join(self.photo_dir, image_name + ".txt")

            # Görüntüyü kaydet
            cv2.imwrite(image_path, image)

            # YOLO verilerini kontrol et ve kaydet
            if len(yolo_data) > 0:
                # YOLO verilerini NumPy dizisine dönüştür ve sayısal tipe çevir
                yolo_data = np.array(yolo_data, dtype=float)

                # Yazma işlemi
                with open(label_path, 'w') as f:
                    np.savetxt(
                        f,
                        yolo_data,
                        fmt=["%d", "%.6f", "%.6f", "%.6f", "%.6f"],  # Formatlama
                        delimiter=" "
                    )
            else:
                print(f"Warning: No YOLO data found. Skipping label file for {image_name}.")

            print(f"Saved: {image_path} and {label_path}")

        except Exception as e:
            print(f"Error while saving image or labels: {str(e)}")


    def crop_img(self, img_path, boxes):
        """
        High-resolution image slicing and bounding box adjustment.

        Parameters:
            img_path (str): Path to the input image.
            boxes (list): List of bounding boxes in the format [class, x_min, y_min, x_max, y_max].
        """
        import cv2

        # Load the image and get its dimensions
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Image at path '{img_path}' could not be loaded.")
            
        img_shape_h, img_shape_w = img.shape[:2]
        crop_size = self.crop_size

        # Calculate padding and step sizes
        pad_w, pad_h = (crop_size - img_shape_w % crop_size) % crop_size, (crop_size - img_shape_h % crop_size) % crop_size
        step_x, step_y = max(1, pad_w // max(1, img_shape_w // crop_size)), max(1, pad_h // max(1, img_shape_h // crop_size))

        # Loop through slices
        for x in range(0, img_shape_w + pad_w, crop_size):
            for y in range(0, img_shape_h + pad_h, crop_size):
                # Calculate slice boundaries
                x_start, y_start = max(0, x - (x // crop_size) * step_x), max(0, y - (y // crop_size) * step_y)
                x_end, y_end = min(img_shape_w, x_start + crop_size), min(img_shape_h, y_start + crop_size)

                # Extract the image slice
                split_img = img[y_start:y_end, x_start:x_end]
                new_img_boxes = []

                # Update bounding box coordinates for the current slice
                for box in boxes:
                    cls, x_min, y_min, x_max, y_max = box
                    if (x_min < x_end and x_max > x_start and
                            y_min < y_end and y_max > y_start):
                        # Adjust bounding box coordinates relative to the slice
                        new_x_min = max(0, x_min - x_start)
                        new_y_min = max(0, y_min - y_start)
                        new_x_max = min(crop_size, x_max - x_start)
                        new_y_max = min(crop_size, y_max - y_start)

                        # Normalize coordinates
                        box_center_x = ((new_x_min + new_x_max) / 2) / crop_size
                        box_center_y = ((new_y_min + new_y_max) / 2) / crop_size
                        box_w = (new_x_max - new_x_min) / crop_size
                        box_h = (new_y_max - new_y_min) / crop_size

                        # Ensure bounding boxes are valid and add to the list
                        #if box_w > 0.025 and box_h > 0.025:
                        new_img_boxes.append([str(cls), float(box_center_x), float(box_center_y), float(box_w), float(box_h)])

                # Save the slice and corresponding bounding boxes
                self.save_img_txt(split_img, new_img_boxes)


if __name__ == '__main__':
    # Argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_path', '-i', type=str, default='data/images/', help='Path to the image directory')
    parser.add_argument('--label_path', '-l', type=str, default='data/labels/', help='Path to the label directory')
    parser.add_argument('--save_path', '-o', type=str, default='data/crop_images/', help='Path to save cropped images and labels')
    parser.add_argument('--crop_size', '-s', type=int, default=512, help='Crop size for slicing images')
    opt = parser.parse_args()

    # Allowed image extensions
    image_extensions = ['.jpg', '.png', '.JPG', '.jpeg', '.JPEG', '.PNG', '.tif']

    # Initialize SlicingImage object
    img_slicing = SlicingImage(opt.crop_size, opt.save_path)

    # Get all label files
    label_names = glob(os.path.join(opt.label_path, '*.txt'))

    for label_name in tqdm(label_names):
        # Determine the corresponding image file
        img_name_base = os.path.splitext(os.path.basename(label_name))[0]
        img_name = None

        for ext in image_extensions:
            candidate = Path(os.path.join(opt.img_path, img_name_base + ext))
            if candidate.exists():
                img_name = str(candidate)
                break

        # If no image file is found, report an error and continue
        if not img_name:
            print(f"ERROR! : No matching image file found for {label_name}")
            continue

        try:
            # Perform slicing and label processing
            img, boxes = img_slicing.boxesFromYOLO(img_name, label_name)
            img_slicing.crop_img(img_name, boxes)
        except Exception as e:
            print(f"ERROR! : An error occurred while processing {img_name} -> {str(e)}")
            continue

    print("Process completed successfully!")

       

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
    def save_img_txt(self,image,yolo_data):
            image_name = f'crop_img_' + datetime.now().strftime("%d.%m.%Y_%H.%M.%S.%f") 
            cv2.imwrite(os.path.join(self.photo_dir, image_name + ".jpg"), image)


            """
            height,width,channels = image.shape
            
            yolo_data =[]
            if len(boxes)>0:
                for bb in boxes:
                    x1, y1, w, h,cls = bb[0], bb[1], bb[2], bb[3],bb[4]
                    x_center = (x1+((w)//2))/width
                    y_center = (y1+((h)//2))/height
                    w1 = (w)/width
                    h1 = (h)/height
                    yolo_data.append([cls,x_center,y_center,w1,h1])
            """

            file_path = os.path.join(self.photo_dir, image_name+ ".txt")
            
            print("yolo_data",yolo_data)
            if len(yolo_data)>0:
                with open(file_path, 'w') as f:
                    np.savetxt(
                        f,
                        yolo_data,
                        fmt=["%d","%f","%f","%f","%f"]
                    )
            else:
                with open(file_path, 'w') as f:
                    np.savetxt(
                        f,
                        yolo_data
                    )


    def crop_img(self,img,boxes):

        img = cv2.imread(img)
        img_shape_w = img.shape[1]
        img_shape_h = img.shape[0]
        image = img.copy()
        mod_w =img_shape_w % self.crop_size
        mod_h = img_shape_h % self.crop_size
        fark_w = self.crop_size - mod_w
        fark_h = self.crop_size - mod_h

        bolum_w = img_shape_w // self.crop_size
        bolum_h = img_shape_h // self.crop_size

        oran_w = fark_w//bolum_w
        oran_h = fark_h//bolum_h

        katsayi_x = -1
        katsayi_y = 0

        for x in range(0,img_shape_w+fark_w,self.crop_size):
            katsayi_x +=1
            katsayi_y = 0
            for y in range(0,img_shape_h+fark_h,self.crop_size):

                if x == 0 and y == 0:
                    cv2.rectangle(image,(x,y),(x+self.crop_size,y+self.crop_size),(255,0,0),6)
                    split_img = img[y:y+self.crop_size,x:x+self.crop_size]
                   
                elif x == 0 and y!=0:                    
                    cv2.rectangle(image,(x,y-katsayi_y*oran_h),(x+self.crop_size,y+self.crop_size-katsayi_y*oran_h),(255,255,255),2)
                    split_img = img[y-katsayi_y*oran_h:y+self.crop_size-katsayi_y*oran_h,x:x+self.crop_size]
                    
                elif x!=0 and y==0:
                    cv2.rectangle(image,(x-(katsayi_x*oran_w),y),(x+self.crop_size-(katsayi_x*oran_w),y+self.crop_size),(255,0,255),2)
                    split_img = img[y:y+self.crop_size,x-(katsayi_x*oran_w):x+self.crop_size-(katsayi_x*oran_w)]
                    
                else:
                    cv2.rectangle(image,(x-(katsayi_x*oran_w),y-katsayi_y*oran_h),(x+self.crop_size-(katsayi_x*oran_w),y+self.crop_size-katsayi_y*oran_h),(0,0,0),2)
                    split_img = img[y-(katsayi_y*oran_h):y+self.crop_size-(katsayi_y*oran_h),x-(katsayi_x*oran_w):x+self.crop_size-(katsayi_x*oran_w)]
                   

                if x!=0 :
                    x = x-(katsayi_x*oran_w)
                if y!=0:
                    y = y-katsayi_y*oran_h
                katsayi_y +=1


                new_img_boxes = []
                split_img_shape_w = split_img.shape[1]
                split_img_shape_h = split_img.shape[0]
                for box in boxes:
                    cls, x_min,y_min, x_max, y_max = box[0], box[1], box[2], box[3], box[4]

                    
                    if (x <= x_min <= x+self.crop_size and y <= y_min <= y+self.crop_size) or (x <= x_max <= x+self.crop_size and y <= y_max <= y+self.crop_size):
                 
                        if x-x_min < 0:
                            new_x_min = abs(x-x_min)
                        else:
                            new_x_min = 0

                        if y-y_min < 0:
                            new_y_min = abs(y-y_min)
                        
                        else:
                            new_y_min = 0


                        if x - x_max < 0:
                            if x - x_max < -(x+self.crop_size):
                                new_x_max = x+self.crop_size
                            else :
                                new_x_max = abs(x-x_max)

                        if y - y_max < 0:
                            if y - y_max < -(y+self.crop_size):
                                new_y_max = y+self.crop_size
                            else :
                                new_y_max = abs(y-y_max)

                        box_center_x = ((new_x_min+new_x_max)//2)/split_img_shape_w
                        box_center_y = ((new_y_min+new_y_max)//2)/split_img_shape_h
                        box_w = (new_x_max - new_x_min)/split_img_shape_w
                        box_h = (new_y_max - new_y_min)/split_img_shape_h

                        #print("box",box_center_x,box_center_y,box_w,box_h)
                        if box_w > 0.02 and box_h > 0.02:
                            new_img_boxes.append([int(cls),box_center_x,box_center_y,box_w,box_h])



                split_img_copy = split_img.copy()
                for i in new_img_boxes:
                    cls, x_center,y_center, w, h = i[0], i[1], i[2], i[3], i[4]
                    x_min = int((x_center - w/2)*split_img_shape_w)
                    y_min = int((y_center - h/2)*split_img_shape_h)
                    x_max = int((x_center + w/2)*split_img_shape_w)
                    y_max = int((y_center + h/2)*split_img_shape_h)
                    cv2.rectangle(split_img_copy,(x_min,y_min),(x_max,y_max),(0,255,0),2)
      
                cv2.namedWindow("image", cv2.WINDOW_NORMAL)
                cv2.resizeWindow("image", 1400, 800)
                cv2.imshow("image", split_img_copy)
                cv2.waitKey(0)


                self.save_img_txt(split_img,new_img_boxes)
                


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_path','-i', type=str, default='data/images/', help='path to images')
    parser.add_argument('--save_path','-o', type=str, default='data/crop_images/', help='path to save images')
    parser.add_argument('--crop_size','-s', type=int, default=512, help='crop size')
    opt = parser.parse_args()


    img_slicing = SlicingImage(opt.crop_size,opt.save_path)

    label_names = glob(opt.img_path + '/*.txt')

    for label_name in tqdm(label_names):
        img_name = label_name.replace('.txt','.jpg')

        if not os.path.exists(img_name):
            img_name = img_name.replace("jpg","png")

            if not os.path.exists(img_name):
                img_name = img_name.replace("png","JPG")



       
        img,boxes = img_slicing.boxesFromYOLO(img_name,label_name)
        img_slicing.showBoxes(img,boxes)
        images = img_slicing.crop_img(img_name,boxes)

       

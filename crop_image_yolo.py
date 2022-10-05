from cProfile import label
from operator import iadd
from tkinter import image_names
import cv2
import numpy as np
import os
import random
from datetime import datetime
from pathlib import Path
import pandas as pd
import argparse
from glob import glob

class SlicingImage():
    def __init__(self,class_size, crop_size=512, save_path="crop_images"):
        self.crop_size = int(crop_size)
        self.save_path = save_path

        self.photo_dir = Path(self.save_path)
        self.photo_dir.mkdir(parents=True, exist_ok=True)


        self.colors = {name:[random.randint(0, 255) for _ in range(3)] for name in range(int(class_size))}



    def boxesFromYOLO(self,imagePath,labelPath):
        image = cv2.imread(imagePath)
        print(image.shape)
        (hI, wI) = image.shape[:2]
        lines = [line.rstrip('\n') for line in open(labelPath)]
        #if(len(objects)<1):
        #    raise Exception("The xml should contain at least one object")
        boxes = []
        if lines != ['']:
            for line in lines:
                components = line.split(" ")
                category = components[0]
                x  = int(float(components[1])*wI - float(components[3])*wI/2)
                y = int(float(components[2])*hI - float(components[4])*hI/2)
                h = int(float(components[4])*hI)
                w = int(float(components[3])*wI)
                boxes.append((category, (x, y, w, h)))
        return (image,boxes)

    #COLORS = np.random.uniform(0,255,size=(18,3))

    def showBoxes(self,image,boxes):
        cloneImg = image.copy()
        for box in boxes:
            if(len(box)==2):
                (category, (x, y, w, h))=box
            else:
                (category, (x, y, w, h),_)=box
    
            color = self.colors[int(category)]

            cv2.rectangle(cloneImg,(x,y),(x+w,y+h),color,1)
            
            #cv2.putText(cloneImg, label, (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 3)
        """    
        cv2.imshow("Image",cloneImg)
        cv2.waitKey(0)
        """
        return cloneImg

    def crop_images_show(self,images):
        for i in images :
            cv2.imshow("image", i)
            cv2.waitKey(0)
            


    #======== tespit edilen görüntüyü ve etiket dosyasını  kayıt ediyor  ===================    
    def save_img_txt(self,image,boxes):
            image_name = f'crop_img_' + datetime.now().strftime("%d.%m.%Y_%H.%M.%S.%f") 
            cv2.imwrite(os.path.join(self.photo_dir, image_name + ".jpg"), image)

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

            file_path = os.path.join(self.photo_dir, image_name+ ".txt")
            
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



    def crop_img(self,img):
        img = cv2.imread(img)
        img_shape_w = img.shape[1]
        img_shape_h = img.shape[0]
        image = img.copy()
        crop_images = []
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
                    crop_images.append(split_img)
                elif x == 0 and y!=0:
                    cv2.rectangle(image,(x,y-katsayi_y*oran_h),(x+self.crop_size,y+self.crop_size-katsayi_y*oran_h),(255,255,255),2)
                    split_img = img[y-katsayi_y*oran_h:y+self.crop_size-katsayi_y*oran_h,x:x+self.crop_size]
                    crop_images.append(split_img)
                elif x!=0 and y==0:
                    cv2.rectangle(image,(x-(katsayi_x*oran_w),y),(x+self.crop_size-(katsayi_x*oran_w),y+self.crop_size),(255,0,255),2)
                    split_img = img[y:y+self.crop_size,x-(katsayi_x*oran_w):x+self.crop_size-(katsayi_x*oran_w)]
                    crop_images.append(split_img)
                else:
                    cv2.rectangle(image,(x-(katsayi_x*oran_w),y-katsayi_y*oran_h),(x+self.crop_size-(katsayi_x*oran_w),y+self.crop_size-katsayi_y*oran_h),(0,0,0),2)
                    split_img = img[y-(katsayi_y*oran_h):y+self.crop_size-(katsayi_y*oran_h),x-(katsayi_x*oran_w):x+self.crop_size-(katsayi_x*oran_w)]
                    crop_images.append(split_img)

                katsayi_y +=1

                cv2.namedWindow("image", cv2.WINDOW_NORMAL)
                cv2.resizeWindow("image", 1400, 800)
                cv2.imshow("image", image)
                cv2.waitKey(0)
                
        #print("crop_images len:",len(crop_images))
        return crop_images
                

    def crop_label_img(self,img,img_shape_w,img_shape_h):
        image = img.copy()
        crop_images = []
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
                    #cv2.rectangle(image,(x,y),(x+self.crop_size,y+self.crop_size),(255,0,0),6)
                    split_img = img[y:y+self.crop_size,x:x+self.crop_size]
                    crop_images.append(split_img)
                elif x == 0 and y!=0:
                    #cv2.rectangle(image,(x,y-katsayi_y*oran_h),(x+self.crop_size,y+self.crop_size-katsayi_y*oran_h),(255,255,255),2)
                    split_img = img[y-katsayi_y*oran_h:y+self.crop_size-katsayi_y*oran_h,x:x+self.crop_size]
                    crop_images.append(split_img)
                elif x!=0 and y==0:
                    #cv2.rectangle(image,(x-katsayi_x*oran_w,y),(x+self.crop_size-katsayi_x*oran_w,y+self.crop_size),(255,0,255),2)
                    split_img = img[y:y+self.crop_size,x-(katsayi_x*oran_w):x+self.crop_size-(katsayi_x*oran_w)]
                    crop_images.append(split_img)
                else:
                    #cv2.rectangle(image,(x-katsayi_x*oran_w,y-katsayi_y*oran_h),(x+self.crop_size-katsayi_x*oran_w,y+self.crop_size-katsayi_y*oran_h),(0,0,0),2)
                    split_img = img[y-(katsayi_y*oran_h):y+self.crop_size-(katsayi_y*oran_h),x-(katsayi_x*oran_w):x+self.crop_size-(katsayi_x*oran_w)]
                    crop_images.append(split_img)

                katsayi_y +=1
                
                
                """
                cv2.namedWindow("image", cv2.WINDOW_NORMAL)
                cv2.resizeWindow("image", 1400, 800)
                cv2.imshow("image", image)
                cv2.waitKey(0)
                """
                
        #print("len crop images",len(crop_images))
        return crop_images


    def get_key(self,val):
        for key, value in self.colors.items():
            if val == value:
                return key
        return "key doesn't exist"


    def find_contour_area(self,img,boxes,canvas,img_shape_w,img_shape_h):
        names = []
        for name in range(len(img)):
            names.append(name)
            
        df = pd.DataFrame(columns=names)
        #print("df:",df)
        
        
        #print("find_contour_area len ", len(images),len(images))
        for id,box in enumerate(boxes):
            if(len(box)==2):
                (category, (x, y, w, h))=box
            else:
                (category, (x, y, w, h),_)=box
            
            color = self.colors[int(category)]

            copy_canvas = canvas.copy()
            cv2.rectangle(copy_canvas,(x,y),(x+w,y+h),color,1)
            
            images = self.crop_label_img(copy_canvas,img_shape_w,img_shape_h)
     
            
            
            
            for num,image in enumerate(images):
                print("find_contour_area i ", num," image shape:",image.shape)
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                ret, thresh = cv2.threshold(gray, 0, 255, 0)
                contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
                label = []
                boxes = []
                if len(contours)>0:
                    for i,cnt in enumerate(contours):
                        x,y,w,h = cv2.boundingRect(cnt)
                        one_label = (x,y,w,h)
                        label.append(one_label)
                        
                        if i != 0 and label[i] == label[i-1]:
                            (x,y,w,h) = label[i-1]
                            #print("değerler aynı")
                            
                        else :
                            #print(x,y,w,h)
                            b,g,r = image[y,x]                     
                            cls = self.get_key([b,g,r])
                            
                            if cls != "key doesn't exist":
                                cls = int(cls)
                                box =  [x,y,w,h,cls]  
                                boxes.append(box)
                            
                            """
                            cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),4)
                            cv2.imshow("image",image)
                            cv2.waitKey(0)
                            """
                            
                df.loc[id,num] = boxes
                #save_img_txt(img[num],boxes)
                #print("saved image---------------------------",num)
                
        
        for lb in range(len(img)):
            
            labels =  df.iloc[:,lb] 
            boxes = []
            for j in range(len(labels)):
                if len((labels.loc[j])) > 0:
                    box = labels.loc[j]
                    #print(f"labels {lb} {j}",box[0])
                    boxes.append(box[0])
                
            #print(f"boxes {lb}",boxes)
            self.save_img_txt(img[lb],boxes)
        
        print("görüntü kırpma işlemi bitti")




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_path','-i', type=str, default='data/images/', help='path to images')
    parser.add_argument('--classes_len','-c', type=int, default=1, help='number of classes')
    parser.add_argument('--save_path','-o', type=str, default='data/crop_images/', help='path to save images')
    parser.add_argument('--crop_size','-s', type=int, default=512, help='crop size')
    opt = parser.parse_args()


    img_slicing = SlicingImage(opt.classes_len,opt.crop_size,opt.save_path)

    label_names = glob(opt.img_path + '/*.txt')
    print("label_names",label_names)

    for label_name in label_names:
        print("label_name",label_name)
        img_name = label_name.replace('.txt','.jpg')
        images = img_slicing.crop_img(img_name)
        img,boxes = img_slicing.boxesFromYOLO(img_name,label_name)
        #img_slicing.showBoxes(img,boxes)
        canvas = np.zeros((img.shape[0],img.shape[1],3), np.uint8)
        label_canvas = img_slicing.showBoxes(canvas,boxes)
        print("image len: ",len(images))
        img_slicing.find_contour_area(images,boxes,canvas,img.shape[1],img.shape[0])






"""

image = cv2.imread("data/yeni_img_1637.jpg")
img_shape_w = image.shape[1]
img_shape_h = image.shape[0]






images = img_slicing.crop_img(image,img_shape_w,img_shape_h)

#crop_images_show(images)

for i in images:
    image_name = f'crop_img_' + datetime.now().strftime("%d.%m.%Y_%H.%M.%S.%f") 
    cv2.imwrite(os.path.join(photo_dir, image_name + ".jpg"), i)



img,boxes = img_slicing.boxesFromYOLO("data/yeni_img_1637.jpg","data/yeni_img_1637.txt")
#showBoxes(img,boxes)


canvas = np.zeros((img_shape_h,img_shape_w,3),dtype="uint8")
label_canvas = img_slicing.showBoxes(canvas,boxes)


#label_images = crop_label_img(label_canvas,img_shape_w,img_shape_h)
#crop_images_show(label_images)

img_slicing.find_contour_area(images,boxes,canvas,img_shape_w,img_shape_h)



"""
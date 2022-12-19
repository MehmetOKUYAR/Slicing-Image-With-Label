# Slicing-Image-With-Label
You can increase your data size and training accuracy by cropping your labeled images to the model input size for large object detection.

Instead of reducing their size while giving your large-sized images as input to the model, you can crop your images with the input size labels of the model and give them as input to your model without losing resolution. In this way, your model performance rate will increase, and your detection rate of smaller objects will increase. If you use your model with [SAHI](https://github.com/obss/sahi) after training it, your model performance will increase even more. In fact, this process aims to increase the model performance by cropping the images in the model input size, not for the test part of the SAHI algorithm, but also during the model training phase.

## How to Run
Labels and Your images must be in the same folder in yolo format

`python3 slicingImg.py --img_path your_path --save_path your_path --crop_size 640`

## You can see how the program works in the image below.

![into image](https://github.com/MehmetOKUYAR/Slicing-Image-With-Label/blob/main/ads%C4%B1z.png)

### **Slicing-Image-With-Label: Enhance Your Training Data and Detection Accuracy**

This repository provides a powerful tool to improve your object detection model's performance, especially for large and high-resolution images. By slicing labeled images into smaller tiles matching your model's input size, you can significantly increase your dataset size and training accuracy, all while preserving image resolution and label integrity.

---

### **Why Use Image Slicing for Object Detection?**

When training object detection models with large images, resizing them to fit the model's input size often leads to a loss of detail, especially for small objects. This loss can negatively impact model performance. Instead of resizing, this repository allows you to **crop your images** into tiles matching your model's input size. This approach ensures:
- **No Loss of Resolution:** Cropped images retain the original resolution and details.
- **Improved Detection of Smaller Objects:** Smaller objects that might otherwise be lost in the resizing process are preserved.
- **Increased Dataset Size:** Cropping automatically generates more training samples, improving your model's generalization capability.

---

### **Key Features**
1. **Seamless Integration with YOLO and SAHI:**
   - The cropped images and their labels are directly compatible with YOLO formats.
   - After training, use SAHI (Slicing Aided Hyper Inference) during inference for even better performance on high-resolution images.

2. **Optimized for Model Training:**
   - Unlike SAHI's primary focus on inference, this tool ensures that the slicing process benefits your model training phase as well.
   - Provides accurate label transformations for cropped images.

3. **Easy to Use:**
   - Simple command-line and script-based workflows.
   - Automatic handling of image padding and label adjustments.

4. **Supports High-Resolution Images:**
   - Designed for large object detection datasets where high resolution is critical.
   - Ideal for use cases like satellite imagery, aerial photography, and detailed industrial inspections.

---

### **How It Works**
1. **Image Slicing:**
   - Images are divided into smaller tiles based on the model's input size, with optional overlapping to ensure no object is cut off.
2. **Label Adjustment:**
   - Each tile’s labels are transformed to match the cropped region.
3. **Integration with Training Pipelines:**
   - Cropped images and adjusted labels are saved in YOLO-compatible formats, ready for training.

---

### **Benefits of This Approach**
- **Better Model Performance:** By feeding the model high-resolution cropped images, it can better learn small object details and patterns.
- **Increased Detection Accuracy:** Especially effective for datasets with a mix of large and small objects.
- **Higher Efficiency with SAHI:** When combined with SAHI during inference, the performance boost is even more noticeable.

---

### **Getting Started**
1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/Slicing-Image-With-Label.git
   cd Slicing-Image-With-Label
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the slicing script:
   ```bash
   python slice_images.py --img_path path/to/images --label_path path/to/labels --crop_size 640 --save_path path/to/output
   ```

4. Train your model with the sliced images and labels:
   - The output images and YOLO-format labels are saved in the specified directory and ready for use in your training pipeline.

---
## You can see how the program works in the image below.
<p align="center">
  <img src="https://github.com/MehmetOKUYAR/Slicing-Image-With-Label/blob/main/images/method.png" alt="Görüntü Açıklaması">
</p>
<br><br>

### **Future Enhancements**
- GPU acceleration for faster slicing of large datasets.
- Advanced label validation to ensure accuracy in complex cases.

---

### **Contributions**
We welcome contributions to improve and extend this repository! Feel free to fork, make changes, and submit a pull request. 

---

By leveraging this tool, you can unlock the full potential of your object detection model, making it robust, accurate, and capable of handling even the most challenging datasets. 🎯

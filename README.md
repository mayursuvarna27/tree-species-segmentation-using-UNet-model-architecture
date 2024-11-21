# tree-species-segmentation-using-UNet-model-architecture
This project uses a U-Net model to segment tree species from satellite images. It features preprocessing (image patchification), custom loss functions (focal and dice loss), and Jaccard-based evaluation. Tools like QGIS, SAM plugin, and ArcMap were used for mask preparation. Includes end-to-end workflow for segmentation.

The original image is of 7906 X 3685 which has been patchified into 256 X 256 with a patch step of 128.This helps us feed more data and also the training becomes more effecient.

Below is the architecture of a simple UNet model which handles multiclass classification: 



**Dataset Information**

The dataset used for this project was custom-created using satellite imagery provided by ISRO, Bangalore. Due to confidentiality, the dataset cannot be shared publicly. The images and masks were in TIFF format, with a total of 1,410 images and 1,410 corresponding masks. Each image and mask had a resolution of 256x256 pixels, suitable for training a deep learning model for segmentation tasks. This dataset was specifically designed for the segmentation of tree species and provided high-quality labeled masks for training and evaluation.

**Preprocessing Steps**

The dataset underwent patchification to prepare it for training. Each original image, with a resolution of 256x256 pixels, was divided into patches of the same size using a step size of 128 pixels. This allowed the model to process smaller sections of the image at a time, improving computational efficiency and making the dataset more manageable. No data augmentation techniques were applied, ensuring the dataset remained in its original form for training and testing. Patchification resulted in a total of 1,410 patches for both images and corresponding masks.

**Model Details**

For this project, we used a U-Net model, a popular architecture for image segmentation tasks. U-Net was chosen due to its ability to effectively capture spatial features and its success in biomedical image segmentation, making it a strong candidate for the task of tree species identification. The model consists of an encoder-decoder structure, with the encoder capturing features and the decoder reconstructing the segmentation map.

The U-Net model was customized slightly to handle multiclass segmentation. The architecture included several convolutional layers (Conv 3x3, ReLU), max-pooling layers (Max Pool 2x2), and up-convolutions (Up-conv 2x2) for precise segmentation. The final output layer produced a segmentation map with 5 classes corresponding to the different tree species. The model was trained using a custom loss function combining Dice loss and focal loss to handle class imbalance.

![image](https://github.com/user-attachments/assets/f0d11c3e-9a8a-4201-b186-78deeb045d24)

**Libraries Used:**

TensorFlow 2.16.1

Keras 3.6.0

NumPy

Matplotlib (for visualizations)


**Model Training:**

The model is compiled using a custom loss function that combines Dice loss and focal loss to address class imbalance in the dataset.
The model is trained with the dataset and evaluated using metrics such as accuracy and Jaccard coefficient.

**Model Evaluation:**

The model's performance is evaluated using the Jaccard coefficient and accuracy on the test set, showing its effectiveness for the segmentation task.


**These are the model stats after training :**

Training Accuracy: 96.60 %	

Training Jaccard  Coefficient: 91.78 %

Training Loss : 0.8430

Validation Accuracy: 94.92%

Validation Jaccard  Coefficient: 89.15%	

Validation Loss: 0.8485

**Results:**

These results indicate that the U-Net model performed well, achieving high accuracy and a good Jaccard coefficient, which measures the overlap between the predicted and true segmentation maps.

While the model achieved excellent results on the training data, further testing and validation would be required for real-world deployment, particularly considering the confidentiality of the dataset. The model's segmentation maps were visually compared with the true masks, confirming that it effectively identified the tree species from the satellite images.



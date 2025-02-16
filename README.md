# Enhanced Medical Image Segmentation via Dual-Encoder Network with Multi-Scale Attention and Edge Detection
This is the implementation method of the MSAE-Net network, along with some data related to the research process.

Data:
The Pulmonary CT Lesion Dataset (PCL) provided by K Scott Mader is available at https://www.kaggle.com/datasets/kmader/finding-lungs-in-ct-data/data. It consists of 267 two-dimensional CT images of lung lesions and their corresponding ground truth images. A portion of the dataset is provided in the folder named L_images.
#
The Digital Retinal Images for Vessel Extraction (DRIVE) dataset can be accessed at https://drive.grand-challenge.org/. It is specifically designed for retinal vessel segmentation and holds significant importance in the field of medical image processing. A sample of this dataset is showcased in the eye_images folder.
#
The MICCAI-Tooth-Segmentation dataset (TOOTH) contains 1998 tooth images, specifically designed for tooth segmentation tasks on the provided two-dimensional panoramic X-ray image data. It can be obtained through the MICCAI-Tooth-Segmentation dataset on Alibaba Cloud Tianchi. A portion of the dataset is displayed in the folder named T_images.
#
#
Instructions: To use our code, you need to prepare within the PyTorch framework. For detailed requirements, please refer to the requirement.txt file. Download the datasets from the aforementioned links, place the training images and labels in the "data/images" and "data/masks" folders respectively, and then run the ktrain.py file to successfully train our model.

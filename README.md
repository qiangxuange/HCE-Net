# MSAE-Net
这是MSAE-Net网络实现的方法以及与研究过程想相关的部分数据等

数据：                                                                                                                          
由K Scott Mader提供的肺部CT病变数据集（PCL）可在 https://www.kaggle.com/datasets/kmader/finding-lungs-in-ct-data/data上获取，由 267张肺部病变的二维CT图像和相应的地面实况图像组成，文件夹L_images中给出了部分数据集。
#
数字视网膜图像数据集 （DRIVE）可通过 https://drive.grand-challenge.org/ 访问。它专门用于视网膜血管分割。它在医学图像处理领域具有重要意义，eye_images文件夹中展示了该数据集。
#
MICCAI-Tooth-Segmentation数据集（TOOTH）包含1998张牙齿图像，专为在提供的二维全景X射线图像数据上进行牙齿分割任务而设计。可通过MICCAI-Tooth-Segmentation数据集-阿里云天池获取，文件夹T_images中展示了部分数据集。
#
#
指南：
使用我们的代码，需要在 PyTorch 框架下做的准备：详见 requirement.txt
 下载上述链接中的数据集，将训练图像和标签分别放入 “data/images” 和 “data/masks” 中，然后运行 ktrain.py 即可成功训练我们的模型。

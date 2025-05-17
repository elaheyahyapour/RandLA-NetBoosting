This repository contains the following figures related to the paper **Fairness-Aware Boosting Model for Imbalanced 3D Point Cloud Segmentation in Autonomous Driving**

<img width="1695" alt="Picture1" src="https://github.com/user-attachments/assets/df466e5f-1858-4267-bde1-58c7a6c3986f" />


The architecture of the proposed RandLA-NetBoosting model. It includes key components such as Local Spatial Encoding (LocSE), AdaBoosting, and Residual Block for feature aggregation. The integration of AdaBoost within LocSE allows the model to effectively focus on underrepresented classes during training, enhancing overall segmentation fairness and accuracy.

<img width="810" alt="Picture3" src="https://github.com/user-attachments/assets/45f1f237-ff29-4f80-9c06-907773437e27" />


The class distribution in two major 3D point cloud datasets:

- (a) Toronto 3D: Shows a significant imbalance with dominant classes like Ground and Building, while critical but underrepresented classes like Road Marking and Fence are underrepresented.

- (b) Semantic3D: Highlights a different distribution, with categories like Building and Man-made Terrain being the majority, creating challenges for segmentation fairness.


<img width="750" alt="Picture2" src="https://github.com/user-attachments/assets/e76b62ae-e917-4b47-a48f-d4364f835eec" />


This figure provides a visual comparison of the segmentation results on the Toronto 3D dataset. It includes:

- (a) Original view of the Toronto 3D scene.

- (b) Results after applying RandLA-Net, which captures major classes but struggles with underrepresented ones like road markings.

- (c) Results after applying RandLA-NetBoosting, which significantly improves the segmentation of minority classes like road markings, highlighting the effectiveness of the proposed approach.

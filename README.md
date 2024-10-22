# YOLOv5 with CMAFF for RGB and Infrared Image Fusion

This project integrates both RGB and infrared (IR) image features using a YOLOv5-based model combined with the Common and Differential Feature Fusion Module (CMAFF). The goal is to enhance object detection by fusing RGB and IR features for better performance.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Setup Environment and Install Dependencies](#setup-environment-and-install-dependencies)
3. [Dataset Structure](#dataset-structure)
4. [Running Training and Evaluation](#running-training-and-evaluation)
5. [Expected Output](#expected-output)
6. [Hyperparameter Tuning](#hyperparameter-tuning)
---

## 1. Project Overview

This project improves object detection by integrating both RGB and IR images into a modified YOLOv5 model with a Common and Differential Feature Fusion Module (CMAFF). The CMAFF module helps the model fuse shared and differential information from both RGB and IR modalities.

---

## 2. Setup Environment and Install Dependencies

Follow these steps to set up the environment and install the necessary dependencies.

### Step 1: Clone the Repository

Clone the repository to your local machine using:

```bash
git clone https://github.com/adityak-21/IPR_Project_Part1.git
cd IPR_Project_Part1
```

### Step 2: Install YOLOv5 Manually

Since YOLOv5 cannot be directly installed via the `requirements.txt` file, you need to manually clone and install it.

Run the following commands to set up YOLOv5:

```bash
git clone https://github.com/ultralytics/yolov5
cd yolov5
pip install -r requirements.txt
cd ..
```
### Step 3: Install Additional Project-Specific Dependencies

After installing YOLOv5, install the remaining dependencies using the `requirements.txt` file in your main project directory:

```bash
pip install -r requirements.txt
```
This installs the following dependencies:
```bash
torch
torchvision
opencv-python
pyyaml
scikit-learn
```
## 3. Dataset Structure

The dataset for this project can be accessed via the following Google Drive link:

[Download Dataset](https://drive.google.com/drive/folders/1YahvxSMhJgYLghfVRHImVsp8ErTn5Xsh?usp=sharing)


- **RGB Images**: Located in the `images-Aditya/train` and `images-Aditya/val` directories, named as `*_co.png`.
- **IR Images**: Located in the same directories, named as `*_ir.png`.
- **Annotations**: Text files in the `labels-Aditya/train` and `labels-Aditya/val` directories, named as `*.txt`.

Place the images-Aditya and labels-Aditya folders into your repo folder.
Please ensure inside images-Aditya folder there is an images folder and inside labels-Aditya a labels folder.

Each annotation file should be formatted as follows:

<class_id> <x_center> <y_center> <width.> <height.>

## 4. Running Training and Evaluation

This project can be run using either a Jupyter Notebook or as a Python script.

## Option 1: Run Using Jupyter Notebook

1. Launch Jupyter Notebook:
   ```bash
   jupyter notebook
   ```
2. Open the notebook named training_&_evaluation.ipynb.
3. Execute the cells in the notebook to train and evaluate the model.

## Option 2: Run Using a Python Script

1. Ensure that `training_&_evaluation.py` is present in the repository.
2. Run the script using the following command:
   ```bash
   python training_&_evaluation.py
    ```
## 5. Expected Output

Once training and evaluation are completed, you will see the following metrics:

- **Precision**: The percentage of correctly predicted bounding boxes.
- **Recall**: The percentage of actual objects that were detected.
- **mAP@0.5**: Mean Average Precision at an IoU threshold of 0.5.

## 6. Hyperparameter Tuning

You can adjust various hyperparameters in the `training_&_evaluation.py` script or notebook, such as:

- **Learning Rate**: Set in the optimizer (Adam in this case).
- **Batch Size**: Configured in the DataLoader for both training and validation datasets.
- **Number of Epochs**: Currently set to 20, but can be modified based on your needs.



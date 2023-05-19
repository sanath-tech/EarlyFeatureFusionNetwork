# Robust Semantic Segmentation for Autonomous Driving Perception by Visual-Infrared Cameras

## Introduction
The repository contains the implementation of a robust semantic segmentation method for autonomous driving perception using visual-infrared cameras. The proposed approach leverages the fusion of RGB and infrared images to achieve superior performance under challenging lighting conditions.

## Dataset
The MFnet dataset is used in this work, which comprises both RGB and IR image frames of real-world driving scenes, along with semantic segmentation annotations. This dataset provides a comprehensive benchmark for evaluating the performance of the proposed method.

## Installation
1. Create a virtual environment:
  ```
  python -m venv venv
  ```



2. Activate the virtual environment:
- For Windows:
  ```
  .\venv\Scripts\activate
  ```
- For Unix or Linux:
  ```
  source venv/bin/activate
  ```
3. Install the required dependencies:

pip install -r requirements.txt

## Usage
1. Configuration Files:
- RGB: `RGB/segformer_mit-b2_4k_32_mfnet_focal.py`
- IR: `configs/segformer/segformer_mit-b2_4k_32_mfnet_focal_ir.py`
- Early Fusion: `configs/segformer/segformer_mit-b2_4k_32_mfnet_focal_rgbt.py`
- ...
(Provide details about the different configuration files and their purposes)

2. Train the model:

python3 tools/train.py config_file --load-from pretrained_weight_file

3. Test the model:

python3 tools/test.py config_file weight_file --eval mIoU

4. Inference on test images:

python3 demo/all_test_image_demo.py config_file weight_file --outfile --type

## Results
The proposed RGB-IR fusion approach achieves state-of-the-art performance in terms of accuracy, speed, and robustness. The results demonstrate the potential of this method for real-time applications in autonomous driving scenarios.

## License
This project is licensed under the [MIT License](LICENSE).

ROBUST SEMANTIC SEGMENTATION FOR AUTONOMOUS DRIVING PERCEPTION BY VISUAL-INFRARED CAMERAS
The impact of environmental perception on the performance of Advanced Driver Assistance Systems (ADAS) is significant since understanding the environment forms the foundation for decision-making. Semantic segmentation, a widely used perception method that assigns a predefined class to each pixel of an image, has been developed using RGB images in recent years. However, RGB image contains information only available within the visible light spectrum. As a result, the network struggles to perform effectively under challenging light conditions such as direct sunlight, exiting tunnels, or darkness. To address this issue,  this thesis proposes a novel approach to fuse information from both RGB and Infrared images to achieve robust semantic segmentation in autonomous driving.
 
The MFnet dataset is the only dataset that comprises both RGB and IR image frames of real-world driving scenes, along with semantic segmentation annotations. In the last decade, as Convolutional Neural Networks (CNNs) are state-of-the-art methods for semantic segmentation, many researchers have built large CNNs comprising two backbones to benchmark on the MFnet dataset. CNN's hierarchical feature extraction forces researchers to follow a feature-level fusion paradigm in which two separate feature extractors are utilized for RGB and IR images.

Recent research has shown that Transformer-based architecture can surpass CNNs in different computer vision tasks. The Transformer's unique ability to capture both local and long-range dependencies in early layers cannot be achieved using convolution. Furthermore, their self-attention modules make them better at generalizing, particularly for multi-spectral data. Thus, in this work, a transformer-based architecture network is proposed. A novel early feature-level fusion paradigm is utilized to ensure a compact, single-backbone network for RGB-IR semantic segmentation.

The results indicate that our RGB-IR fusion approach surpasses other state-of-the-art methods in accuracy, speed, and robustness, demonstrating its potential for use in real-time applications.

Installation procedure
 
1. create virtual environment

2. Install all requirements as shown in official mmsegmentation document: https://mmsegmentation.readthedocs.io/en/latest/get_started.html

3. Change envs/lib64/python3.8/site-packages/mmcv/runner/iter_based_runner.py file to one in repo to enable validation loss function



CONFIG FILES
RGB: RGB/segformer_mit-b2_4k_32_mfnet_focal.py

IR: configs/segformer/segformer_mit-b2_4k_32_mfnet_focal_ir.py

early_fusion: configs/segformer/segformer_mit-b2_4k_32_mfnet_focal_rgbt.py

best_fusion_1x1 : fusion/single1x1_earlyfuseformer_mit-b2_4k_32_mfnet_focal_rgbt.py

CBAM: configs/segformer/CBAM_early_feature_fusion_mit-b2_4k_32_mfnet_focal_rgbt.py

Modal_specific_Depth Seperable convolution: configs/segformer/depthsep_1x1_earlyfuseformer_mit-b2_4k_32_mfnet_focal_rgbt.py

Cross attention: configs/segformer/Cross_attn_earlyfuseformer_mit-b2_4k_32_mfnet_focal_rgbt.py

Double 1x1: configs/segformer/double1x1_earlyfuseformer_mit-b2_4k_32_mfnet_focal_rgbt.py

Modal specific 1x1: configs/segformer/modal_specific_1x1_earlyfuseformer_mit-b2_4k_32_mfnet_focal_rgbt.py

TRAIN
python3 tools/train.py config_file --load-from pretrained_weight_file


TEST
python3 tools/test.py config_file weight_file --eval mIoU


INFERENCE
python3 demo/all_test_image_demo.py config_file weight_file --outfile  --type 


EXAMPLE:
python3 demo/all_test_image_demo.py 1x1_earlyfusion/single1x1_earlyfuseformer_mit-b2_4k_32_mfnet_focal_rgbt.py 1x1_earlyfusion/latest.pth --outfile visual_result/fusion --type rgbt


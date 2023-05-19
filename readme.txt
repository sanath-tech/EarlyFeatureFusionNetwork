Installation procedure
 
1. create virtual environment

2. Install all requirements as shown in official mmsegmentation document: https://mmsegmentation.readthedocs.io/en/latest/get_started.html

3. Change envs/lib64/python3.8/site-packages/mmcv/runner/iter_based_runner.py file to one in repo to enable validation loss function

4. Symlink dataset file to FuseFormer/data

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



# Copyright (c) OpenMMLab. All rights reserved.
from argparse import ArgumentParser

from mmseg.apis import inference_segmentor, init_segmentor, show_result_pyplot
from mmseg.core.evaluation import get_palette
import matplotlib.pyplot as plt
import mmcv
import torch
from mmcv.parallel import collate, scatter
from mmcv.runner import load_checkpoint

from mmseg.datasets.pipelines import Compose
from mmseg.models import build_segmentor
def _show_result_pyplot(model,
                       img,
                       result,
                       palette=None,
                       fig_size=(15, 10),
                       opacity=1,
                       title='',
                       block=True,
                       out_file=None):
    """Visualize the segmentation results on the image.

    Args:
        model (nn.Module): The loaded segmentor.
        img (str or np.ndarray): Image filename or loaded image.
        result (list): The segmentation result.
        palette (list[list[int]]] | None): The palette of segmentation
            map. If None is given, random palette will be generated.
            Default: None
        fig_size (tuple): Figure size of the pyplot figure.
        opacity(float): Opacity of painted segmentation map.
            Default 0.5.
            Must be in (0, 1] range.
        title (str): The title of pyplot figure.
            Default is ''.
        block (bool): Whether to block the pyplot figure.
            Default is True.
        out_file (str or None): The path to write the image.
            Default: None.
    """
    if hasattr(model, 'module'):
        model = model.module
    img = model.show_result(
        img, result, palette=palette, show=False, opacity=opacity)
    plt.figure(figsize=fig_size)
    #plt.imshow(mmcv.bgr2rgb(img))
    plt.title(title)
    plt.tight_layout()
    #plt.show(block=block)
    if out_file is not None:
        mmcv.imwrite(img, out_file)

def main():
    parser = ArgumentParser()
    #parser.add_argument('img', help='Image file')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument('--outfile', default=None, help='Path to output file')
    parser.add_argument('--type', default=None, help='rgb/ir/rgbt')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--palette',
        default='mfnet',
        help='Color palette used for segmentation map')
    parser.add_argument(
        '--opacity',
        type=float,
        default=1,
        help='Opacity of painted segmentation map. In (0, 1] range.')
    args = parser.parse_args()

    # build the model from a config file and a checkpoint file
    model = init_segmentor(args.config, args.checkpoint, device=args.device)
    # test a single image
    from PIL import Image
    import numpy as np
    import os
    list_file = 'test.txt'
    files = []
    with open(list_file,'r') as reader:
        for line in reader.readlines():
            files.append(line.strip('\n'))
    if(args.type=='rgb'):g='_rgb.png'
    if(args.type=='ir'):g='_ir.png'
    if(args.type=='rgbt'):g='_rgbt.png'
    for f in files:
        result = inference_segmentor(model,'data/MFnet/img_dir/'+args.type+'/test/'+f+g)
    # show the results
        _show_result_pyplot(
            model,
            'data/MFnet/img_dir/'+args.type+'/test/'+f+g,
            result,
            get_palette(args.palette),
            opacity=args.opacity,
            out_file= args.outfile+'/'+f+'_sg'+g)
            #out_file='/home/sanath/FuseFormer/visual_result/rgb/'+f+'_sg_rgb.png')


if __name__ == '__main__':
    main()

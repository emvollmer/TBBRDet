# Copyright (c) OpenMMLab. All rights reserved.
import argparse
from pathlib import Path

import torch
from mmdet.apis import init_detector, inference_detector, show_result_pyplot
import mmcv
import numpy as np

# Need this to stop matplotlib from trying to open a figure in a window
import matplotlib
matplotlib.use('Agg')


def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet predict on image with previously trained model')
    parser.add_argument('input',
                        help='image .npy path on which to predict')
    parser.add_argument('--config', 'config_file', help='model config (.py)')
    parser.add_argument('--checkpoint', 'checkpoint_file',
                        help='model checkpoint (.pth) with which to predict')
    parser.add_argument('--colour-channel', 'colour_channel', default='both',
                        choices=['TIR', 'RGB', 'both'],
                        help='Image channels on which the predictions will be visualized. '
                             'Choice of RGB, TIR or both side by side.')
    parser.add_argument('--score-thr', type=float, default=0,
                        help='score threshold (default: 0.)')
    parser.add_argument('--out-dir', 'out_dir',
                        help='directory where predicted on image will be saved to')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    mmcv.check_file_exist(args.config_file)
    mmcv.check_file_exist(args.checkpoint_file)

    # build the model from the config and checkpoint files
    if torch.cuda.is_available():
        DEVICE = torch.device('cuda:0')
    else:
        DEVICE = torch.device('cpu')

    model = init_detector(args.config_file, args.checkpoint_file, device=DEVICE)

    # infer on image
    npy_path = Path(args.input)
    if not npy_path.is_file() and npy_path.suffix != ".npy":
        raise FileNotFoundError(f'Provided input file {args.input} is not a .npy file!')

    pred_result = inference_detector(model, npy_path)

    # load the image depending on the user colour channel choice
    numpy_img = np.load(npy_path)
    out_path = Path(args.out_dir, npy_path.parent.name,
                    npy_path.stem + "_score" + args.score_thr + ".png")

    if args.colour_channel == "both":
        rgb = numpy_img[:, :, :3]
        tir = np.dstack((numpy_img[:, :, 3], numpy_img[:, :, 3], numpy_img[:, :, 3]))

        # save the results
        out_tir = model.show_result(img=tir, result=pred_result, score_thr=args.score_thr)
        out_rgb = model.show_result(img=rgb, result=pred_result, score_thr=args.score_thr)
        mmcv.imwrite(np.hstack([out_rgb, out_tir]), out_path)
    else:
        if args.colour_channel == "TIR":
            img = np.dstack((numpy_img[:, :, 3], numpy_img[:, :, 3], numpy_img[:, :, 3]))
        elif args.colour_channel == "RGB":
            img = numpy_img[:, :, :3]
        else:
            raise ValueError(f"{args.colour_channel} is not a valid colour channel argument!")

        # save the results
        model.show_result(img=img, result=pred_result, score_thr=args.score_thr,
                          out_file=out_path)


if __name__ == "__main__":
    main()

# MMDetection scripts

For MMDetection instructions see the [MMDetection README](mmdet/README.md).

Because MMDetection requires importing the pipeline class defined `numpy_loader.py` to load inputs,
all MMDetection scripts are within a single directory.
Similarly, `common_vars.py` defines common config parameters.

All other scripts are adapted from the `tools/` scripts provided in the [MMDetection repository](https://github.com/open-mmlab/mmdetection).

## Structure

```
├── README.md                       <- This file
├── analyze_results.py              <- Plots loss/mAP curves
├── browse_dataset.py               <- Visualise annotations and dataset
├── combine_evaluation_scores.py    <- Calculate mean/std of evaluation scores
├── common_vars.py                  <- Common configurations
├── numpy_loader.py                 <- Pipeline class to load NumPy inputs
├── slurm_bulk_test.sh              <- Submit multiple evaluation jobs
├── slurm_submit.sh                 <- Submit a single training job
├── submit_all_seeds.sh             <- Run multiple trainings with different seeds
├── test.py                         <- Evaluate a trained model
└── train.py                        <- Train a model
```


## Visualisation

For plotting the training dataset, I've set up a copy of the config used for MaskRCNN R50 training but with augmentations removed.
You have to adapt the plotting config to insert the correct data root and working directories or adapt using the `cfg-options` flag (s. below), then you can plot it with something like
```bash
python browse_dataset.py --output-dir /path/to/output/ --channel RGB --not-show ~/TBBRDet/configs/mmdet/common/plotting_config.py
```

Change the `--channel` flag to `Thermal` if you want to plot the annotations on the thermal images instead of the RGBs.
Add in the flag `--num-processes x` with a value for x to define the number of processes to run in parallel.

---
**Note**

To overwrite config definitions, use the `--cfg-options` flag. In this case, if you don't want to change the configs manually at all, you'll need to add `data_root` and `work_dir` parameters via the terminal function call. This would look like the following:
```bash
python browse_dataset.py ~/TBBRDet/configs/mmdet/common/plotting_config.py
 --output-dir /path/to/output/ --channel RGB --not-show
 --cfg-options 'data_root'='/path/to/datasets' 'work_dir'='/path/to/work_dir'
```
Make sure to leave the quote marks as is and do not add an extra `/` at the end of the directories.

---

## Training

### From scratch

Training with a single GPU can be accomplished by running the `train.py` script directly with something like:
```bash
python train.py configs/mmdet/<MODEL_NAME>/..._coco.scratch.py --work-dir /path/to/work_dir --seed <SEED_NUM> --deterministic
```
The `work_dir` flag value should be set as the directory to which the model results will be saved.

The configs have to be adapted in advance again - in particular the `data_root` parameter in the `configs/mmdet/_base_/datasets/coco_instance.py`.
Again, this can be done directly during the function call using the `--cfg-options` flag.

Other parameters can also be changed using the `--cfg-options` depending on their position in the config dictionary.
Some examples would be:
```bash
... --cfg-options 'runner.max_epochs'=12 'data.workers_per_gpu'=4
```

### Using pretrained weights

Similarly to before, training with a single GPU can be accomplished by running the `train.py` script directly, although a different config has to be used and according weights need to be downloaded.
 The weights can be found in your config of choice in the [MMDetection GitHub repo](https://github.com/open-mmlab/mmdetection/blob/v2.21.0/configs/).
The required weights (.pth) file can be downloaded into your directory of choice from the matching GitHub page. For example, for the swin t, this would be done with:
```bash
wget https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_small_patch4_window7_224.pth
# alternatively
wget https://download.openmmlab.com/mmdetection/v2.0/swin/mask_rcnn_swin-t-p4-w7_fpn_fp16_ms-crop-3x_coco/mask_rcnn_swin-t-p4-w7_fpn_fp16_ms-crop-3x_coco_20210908_165006-90a4008c.pth
```

Make sure to adapt the `load_from` in `configs/mmdet/<MODEL>/..._coco.pretrained.py` to the directory of choice to which the .pth weights file was saved.
For the above example, this would mean changing the `load_from` definition in `configs/mmdet/swin/mask_rcnn_swin-t-p4-w7_fpn_fp16_ms-crop-3x_coco.pretrained.py`.
Again, this can be done using the `cfg-options` flag.

Training with a single GPU can then be run with something like:
```bash
python train.py configs/mmdet/<MODEL_NAME>/..._coco.pretrained.py --work-dir /path/to/work_dir --seed <SEED_NUM> --deterministic
 --cfg-options 'data_root'='/path/to/datasets' 'load_from'='/path/to/pretrained/weights.pth'
```

---
**Issue with training using pretrained weights**

Following the afore-mentioned instructions (in other words the config `..._coco.pretrained.py` results in the logger issuing a warning:
```bash
mmdet - WARNING - No pre-trained weights for SwinTransformer, training start from scratch
```

I assume this is on account of the missing definition of a model `init_cfg` type and checkpoint file within the config.
The [equivalent MMDetection example config](https://github.com/open-mmlab/mmdetection/blob/v2.21.0/configs/swin/mask_rcnn_swin-s-p4-w7_fpn_fp16_ms-crop-3x_coco.py) shows
 how they download and include the weights thusly.

When this definition is incorporated into our training procedure, the `mmdet - WARNING` message is replaced by `mmdet - INFO - load checkpoint from ...`.
However, an error occurs on account of what I assume to be the different data sizes (original 3 channel vs. here 5 channels - note the second list element):
```bash
RuntimeError: Error(s) in loading state_dict for SwinTransformer:
size mismatch for patch_embed.projection.weight: copying a param with shape torch.Size([96, 3, 4, 4])
from checkpoint, the shape in current model is torch.Size([96, 5, 4, 4]).
```
Further clarification is required as to how the model(s) was/were previously trained using pretrained weights.

---

### Batch training

To run training in batch, use the `slurm_submit.sh` script. Before running, make sure to change it to your specific needs (adapt `#SBATCH` logging directories and partition and `source` directory).
Then run the script with
```bash
sbatch -J <JOB_NAME> slurm_submit.sh <CONFIG_DIR> <SEED_NUM>

# example from above
sbatch -J swin-pretrained scripts/mmdet/slurm_submit.sh configs/mmdet/swin/mask_rcnn_swin-t-p4-w7_fpn_fp16_ms-crop-3x_coco.pretrained.py <SEED_NUM>
```

#### Common Errors to look out for

- Process killed after the `Checkpoints will be saved` information / `RuntimeError: DataLoader worker (pid 485194) is killed by signal: Killed.`: This issue is usually caused by limited machine resources. Make sure to allocate enough memory. If the issue persists, reduce the `workers-per-gpu` parameter in the `configs/models/<MODEL-NAME>/...coco.py` config.

## Testing

Testing scripts can be run with
```bash
python test.py <CONFIG_PATH> <MODEL_PATH> --work-dir /path/to/results/ --out /path/to/model_eval.pickle --eval <METRIC>

# example from above
python scripts/mmdet/test.py configs/mmdet/swin/mask_rcnn_swin-t-p4-w7_fpn_fp16_ms-crop-3x_coco.pretrained.py 
/path/to/results/best_AR@1000_epoch_35.pth 
--work-dir /path/to/results/eval_results/ 
--out /path/to/results/eval_results/model_evaluation_bbox.pickle
--eval bbox
```
Metrics can be f.e. `bbox` or `segm` depending on the model and aim.


## Analyzing results

You should set `TOPDIR` to the top-level work directory of the training, then run
```bash
python analyze_results.py $TOPDIR/mask_*.py $TOPDIR/evaluation/eval_results.pkl $TOPDIR/evaluation/images_0.3/ --show-score-thr 0.3 --topk 50
```

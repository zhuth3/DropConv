
# Drop Sparse Convolution for 3D Object detection 
The framework is based on [`OpenPCDet`](https://github.com/open-mmlab/OpenPCDet) and [`FocalConv`](https://github.com/dvlab-research/FocalsConv)

## Requirment
* CUDA (test on 11.1)
* pytorch (test on 1.10.1)
* spconv (test on 2.1.25)

## Install
```
pip install -r requirements.txt
python setup.py develop
```
The detail is in [`INSTALL`](docs/INSTALL.md)

## Training and testing
### Prepare dataset
The detail about datasets is in [`GETTING_STARTED`](docs/GETTING_STARTED.md)
* KITTI
```
python -m pcdet.datasets.kitti.kitti_dataset create_kitti_infos tools/cfgs/dataset_configs/kitti_dataset.yaml
```
* nuScenes
```
python -m pcdet.datasets.nuscenes.nuscenes_dataset --func create_nuscenes_infos \
    --cfg_file tools/cfgs/dataset_configs/nuscenes_dataset.yaml \
    --version v1.0-trainval
```

### Train a model
* Train with a single GPU
```
python train.py --cfg_file ${CONFIG_FILE}
```

Our config files on KITTI are in [`SECOND+Drop Conv`](tools/cfgs/kitti_models/second_drop.yaml), [`Voxel RCNN+Drop Conv`](tools/cfgs/kitti_models/voxel_rcnn_car_drop.yaml), [`PV RCNN+Drop Conv`](tools/cfgs/kitti_models/pv_rcnn_drop.yaml), and on nuScenes are in [`SECOND+Drop Conv`](tools/cfgs/nuscenes_models/cbgs_second_multihead_drop.yaml), [`CenterPoint01+Drop Conv`](tools/cfgs/nuscenes_models/cbgs_voxel01_res3d_centerpoint_drop.yaml) and [`CenterPoint0075+Drop Conv`](tools/cfgs/nuscenes_models/cbgs_voxel0075_res3d_centerpoint_drop.yaml)


* Train with multiple GPUs or multiple machines
```shell script
sh scripts/dist_train.sh ${NUM_GPUS} --cfg_file ${CONFIG_FILE}

# or 

sh scripts/slurm_train.sh ${PARTITION} ${JOB_NAME} ${NUM_GPUS} --cfg_file ${CONFIG_FILE}
```

### Test and evaluate the pretrained models
* Test with a pretrained model: 
```shell script
python test.py --cfg_file ${CONFIG_FILE} --batch_size ${BATCH_SIZE} --ckpt ${CKPT}
```

* To test all the saved checkpoints of a specific training setting and draw the performance curve on the Tensorboard, add the `--eval_all` argument: 
```shell script
python test.py --cfg_file ${CONFIG_FILE} --batch_size ${BATCH_SIZE} --eval_all
```

* To test with multiple GPUs:
```shell script
sh scripts/dist_test.sh ${NUM_GPUS} \
    --cfg_file ${CONFIG_FILE} --batch_size ${BATCH_SIZE}

# or

sh scripts/slurm_test_mgpu.sh ${PARTITION} ${NUM_GPUS} \
    --cfg_file ${CONFIG_FILE} --batch_size ${BATCH_SIZE}
```

The detail about training and testing is in [`GETTING_STARTED`](docs/GETTING_STARTED.md)
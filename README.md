# DPCT: Unsupervised Deep Point Correspondence based on Transformer via Cross and Self Construction

This model is based on the [**DPC**](https://github.com/dvirginz/DPC) model and used offset-attention from [**PCT**](https://github.com/MenghaoGuo/PCT) model.

Contact: [Hai Iluz](mailto:xhailuzx@gmail.com)

## Abstract

This repo presents an enhanced Deep Point Correspondence model based on Transformer (DPCT) for real-time non-rigid dense correspondence between point clouds. This problem is one of the main problems in 3D vision, especially in the variety of applications in the real-time domain like VR and AR. This model is based on the DPC model and uses a variant of the PCT encoder in order to improve the DPC results. In addition, the Hungarian algorithm is used to solve the assignment problem of point correspondence. DPC is a model based on latent similarity and cross and self-reconstruction, it uses the latent similarity and the input coordinates themselves to construct the point cloud and determine correspondence. PCT is a framework for point cloud learning, the novelty of this framework is to add an offset-attention module in order to obtain better network performance. The combination of these methods creates a **SOTA** model that presents an improved accuracy and error on the SHREC'19 and TOSCA datasets compared to recent state-of-the-art correspondence methods. 

<img src=./SHREC.gif width="200" />

## Experiments
DPCT improved the results of the DPC model and achived great results.

The penultimate line (DPCT method) belongs to the model with offset-attention and the last line (DPCT(assignment) method) belongs to the model with offset-attention and Hungarian algorithm.

![image](https://user-images.githubusercontent.com/102179195/195860045-81fcd057-3a0e-4f17-a387-1fb9b5aac9e4.png)

## Installation

Please follow the `requirements.txt`. If for some reason it's not work follow the `installation.sh`

## Inference

For inference, please run the following line with the relevan checkpoint. The checkpoints are in the data/ckpts/ folder. Note that in this folder you can find the DPCT checkpoints but also the DPC checkpoints.

```
python train_point_corr.py --do_train false  --resume_from_checkpoint <path>
```

For a pre-trained model that trained on SHREC'19 dataset
```
python train_point_corr.py --do_train false  --resume_from_checkpoint data/ckpts/DPCT_shrec_ckpt.ckpt
```

For a pre-trained model that trained on TOSCA dataset
```
python train_point_corr.py --do_train false  --resume_from_checkpoint data/ckpts/DPCT_tosca_ckpt.ckpt
```


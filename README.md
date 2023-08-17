# PolyGraph
Graph-based Method for Floorplans Reconstruction from 3D Scans

## Introduction
This study proposes a new reconstruction pipeline, PolyGraph. It begins with a 2D density/normal map derived from raw 3D point data as the initial input, first detects structural wall points, and then proceeds to optimize the overall floorplan structure. Our main contributions are:
- For the first time, we model the problem of indoor holistic floorplan reconstruction as an optimal subgraph optimization task.
- We propose a guided primitive estimation network to generate points that could capture more comprehensive structural information about the walls and possess higher error tolerance compared to sole corner points.
- we design a new structural weight, which plays a key role in solving the optimal subgraph problem. The weight considers both the confidence degree as a real wall and the impact of the length of the wall.

This repo provides the code, data, and pre-trained checkpoints of PolyGraph for the two datasets.

## Preparation

### Environment

This repo was developed and tested with ```Python3.8```

Install the required packages

```
pip install -r requirements.txt
```

### Data

Please download the processed datasets [Structure3D](https://pan.baidu.com/s/1jEImIAUsH8K5QgKBVpCoPQ?pwd=mnlq) and [Lianjia-s](https://pan.baidu.com/s/1hva7a2Bl943NnaDIynj4Mg?pwd=s8kx).  Extract the data into the ```./datasets``` directory.

The file structure should be like the following:
```
datasets
├── Structure3d # the Structured3D floorplan dataset
│   ├── train   
│   │── test    
│   │── valid  
└── Lianjia     # the Lianjia-s floorplan dataset
    ├── train         
    ├── test        
    │── valid       
```
Note that the data is processed as we stated in our paper. The origin data can be download from [ori-Lianjia-s](https://www.ke.com) at [link1](http://realsee.com/open/en) or [link2](http://realsee.com/open) and [ori-Structure3D](https://github.com/woodfrog/heat/tree/master/s3d_preprocess).

### Checkpoints

We provide the checkpoints for [Structure3D](https://drive.google.com/file/d/1Oua4RCaxOIm7-mWXoUJZHTNE3oSPrDTw/view?usp=sharing) and [Lianjia-s](), please download and extract.

## Inference, evaluation and visualization

We provide the instructions to run the inference, quantitative evaluation, and qualitative visualization in this section.

### Structure3D

- **Inference.** Run the inference with the pre-trained checkpoints.
  ```
  python infer.py --is_val True --pretrained_weights ./checkpoints/Structure3d.pth
  python optimization.py
  ```

- **Quantitative evaluation.** The quantitative evaluation is again adapted from the code of HEAT[1]. Run the evaluation by:
  ```
  cd evaluation/s3_floorplan_eval
  python evaluate_solution.py --dataset_path ./montefloor_data --dataset_type s3d --scene_id val
  cd ..
  ```
  
### Lianjia-s

- **Inference.** Run the inference with the pre-trained checkpoints.
  ```
  python infer.py --is_val True --pretrained_weights ./checkpoints/Lianjia.pth
  python optimization.py
  ```
  Notice that the ```dataname``` in ```optimization.py``` should be changed to ```Lianjia```.
  
- **Quantitative evaluation.** Run the evaluation by:
  ```
  cd evaluation/Lianjia
  python eval_lj.py 
  cd ..
  ```

## Training

and then run the training by:

```
python infer.py
```

## References

[1].J. Chen, Y. Qian and Y. Furukawa, "HEAT: Holistic Edge Attention Transformer for Structured Reconstruction," 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) (2022):3856-3865.

[2]. Sinisa Stekovic, Mahdi Rad, Friedrich Fraundorfer, and Vincent Lepetit. Montefloor: Extending mcts for reconstructing accurate large-scale floorplans. 2021 IEEE/CVF International Conference on Computer Vision(ICCV), 2021.

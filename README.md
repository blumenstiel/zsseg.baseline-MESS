# Multi-domain Evaluation of Semantic Segmentation (MESS) with ZSSeg

[[Website (soon)](https://github.io)] [[arXiv (soon)](https://arxiv.org/)] [[GitHub](https://github.com/blumenstiel/MESS)]

This directory contains the code for the MESS evaluation of ZSSeg. Please see the commits for our changes of the model.

## Setup
Create a conda environment `zsseg` and install the required packages. See [mess/README.md]([mess/README.md]) for details.
```sh
 bash mess/setup_env.sh
```

Prepare the datasets by following the instructions in [mess/DATASETS.md](mess/DATASETS.md). The `zsseg` env can be used for the dataset preparation. If you evaluate multiple models with MESS, you can change the `dataset_dir` argument and the `DETECTRON2_DATASETS` environment variable to a common directory (see [mess/DATASETS.md](mess/DATASETS.md) and [mess/eval.sh](mess/eval.sh), e.g., `../mess_datasets`). 

Download the ZSSeg weights (see https://github.com/MendelXu/zsseg.baseline)
```sh
mkdir weights
conda activate zsseg
# Python code for downloading the weights from GDrive. Link: https://drive.google.com/file/d/1pb6UeXoMPy5xdEBtFcQYLOBKZt0xufKY/view
python -c "import gdown; gdown.download(f'https://drive.google.com/uc?export=download&confirm=pbef&id=1pb6UeXoMPy5xdEBtFcQYLOBKZt0xufKY', output='weights/model_final.pth')"
```

## Evaluation
To evaluate the ZSSeg model on the MESS datasets, run
```sh
bash mess/eval.sh

# for evaluation in the background:
nohup bash mess/eval.sh > eval.log &
tail -f eval.log 
```
Note that the mask threshold was changed from 0.5 to 0.4 in commit [`aa327d4`](https://github.com/MendelXu/zsseg.baseline/commit/aa327d4e324afd851459bb1bffe6bbdce38dc9f9). We used the original threshold of 0.5 in the evaluation.

For evaluating a single dataset, select the DATASET from [mess/DATASETS.md](mess/DATASETS.md), the DETECTRON2_DATASETS path, and run
```
conda activate zsseg
export DETECTRON2_DATASETS="datasets"
DATASET=<dataset_name>

python train_net.py --num-gpus 1 --eval-only --config-file configs/ade20k-150/cross_dataset_test_only.yaml MODEL.WEIGHTS weights/model_final.pth OUTPUT_DIR output/ZSSeg/$DATASET DATASETS.TEST \(\"$DATASET\",\) MODEL.CLIP_ADAPTER.MASK_THR 0.5
```

# --- Original ZSSeg README.md ---

# [ECCV2022] A Simple Baseline for Open Vocabulary Semantic Segmentation with Pre-trained Vision-language Model

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/2112-14757/open-vocabulary-semantic-segmentation-on)](https://paperswithcode.com/sota/open-vocabulary-semantic-segmentation-on?p=2112-14757)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/2112-14757/open-vocabulary-semantic-segmentation-on-3)](https://paperswithcode.com/sota/open-vocabulary-semantic-segmentation-on-3?p=2112-14757)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/2112-14757/open-vocabulary-semantic-segmentation-on-1)](https://paperswithcode.com/sota/open-vocabulary-semantic-segmentation-on-1?p=2112-14757)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/2112-14757/open-vocabulary-semantic-segmentation-on-2)](https://paperswithcode.com/sota/open-vocabulary-semantic-segmentation-on-2?p=2112-14757)

This repo is for our ECCV2022 paper [A Simple Baseline for Open Vocabulary Semantic Segmentation with Pre-trained Vision-language Model](https://arxiv.org/pdf/2112.14757.pdf). It is based on the official repo of [MaskFormer](https://github.com/facebookresearch/MaskFormer).

### :tada:**[News]** Our new work [Side Adapter Network for Open-Vocabulary Semantic Segmentation](https://arxiv.org/abs/2302.12242) was accepted as CVPR2023 hightlight paper. [Code](https://github.com/MendelXu/SAN) has been released.



![](resources/proposal.png)
```
@article{xu2021,
  title={A Simple Baseline for Open Vocabulary Semantic Segmentation with Pre-trained Vision-language Model},
  author={Mengde Xu, Zheng Zhang, Fangyun Wei, Yutong Lin,Yue Cao, Han Hu, and Xiang Bai},
  journal={Proceedings of the IEEE/CVF European Conference on Computer Vision (ECCV)},
  year={2022}
}
```

## Guideline
- ### Enviroment
     ```bash
     torch==1.8.0
     torchvision==0.9.0
     detectron2==0.6 #Following https://detectron2.readthedocs.io/en/latest/tutorials/install.html to install it and some required packages
     mmcv==1.3.14
     ```
     FurtherMore, install the modified clip package.
     ```bash
     cd third_party/CLIP
     python -m pip install -Ue .
     ```
- ### Data Preparation
  In our experiments, five datasets are used. 
  - For Cityscapes and ADE20k, follow the tutorial in [MaskFormer](https://github.com/facebookresearch/MaskFormer).
  - For Pascal Context:
    - Download data from the official website and extract it like below.
      ```shell
      Datasets/
          pcontext/
              #http://host.robots.ox.ac.uk/pascal/VOC/voc2010/VOCtrainval_03-May-2010.tar
              JPEGImages/
              #https://codalabuser.blob.core.windows.net/public/trainval_merged.json     
              trainval_merged.json 
      ```
    - Format the data to d2 style.
      install detail packpage from https://github.com/zhanghang1989/detail-api and then
      ```shell
      python datasets/prepare_pcontext_sem_seg.py --ori_root_dir datasets/pcontext-59 --save_dir datasets/pcontext-59
      ```
  - For COCO Stuff 164k:
    - Download data from the official dataset website and extract it like below.
       ```bash
       Datasets/
            coco/
                 #http://images.cocodataset.org/zips/train2017.zip
                 train2017/ 
                 #http://images.cocodataset.org/zips/val2017.zip
                 val2017/   
                 #http://images.cocodataset.org/annotations/annotations_trainval2017.zip
                 annotations/ 
                 #http://calvin.inf.ed.ac.uk/wp-content/uploads/data/cocostuffdataset/stuffthingmaps_trainval2017.zip
                 stuffthingmaps/ 
       ```
    - Format the data to d2 style and split it into Seen (Base) subset and Unseen (Novel) subset.
       ```bash
       python datasets/prepare_coco_stuff_164k_sem_seg.py datasets/coco

       python tools/mask_cls_collect.py datasets/coco/stuffthingmaps_detectron2/train2017_base datasets/coco/stuffthingmaps_detectron2/train2017_base_label_count.pkl
     
       python tools/mask_cls_collect.py datasets/coco/stuffthingmaps_detectron2/val2017 datasets/coco/stuffthingmaps_detectron2/val2017_label_count.pkl
       ```   
  - For Pascal VOC 11k:
    - Download data from the offical dataset website and extract it like below.
    ```bash
    datasets/
       VOC2012/
            #download http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar and extract it here.
            JPEGImages/
            # copy `ImageSets/Segmentation/val.txt` here.
            val.txt
            #Download auged annotations from http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/semantic_contours/benchmark.tgz and convert the original mat file to png format. Or download it from https://www.dropbox.com/s/oeu149j8qtbs1x0/SegmentationClassAug.zip?dl=0 (Provided in https://github.com/DrSleep/tensorflow-deeplab-resnet#evaluation).
            SegmentationClassAug/
            #https://gist.githubusercontent.com/sun11/2dbda6b31acc7c6292d14a872d0c90b7/raw/5f5a5270089239ef2f6b65b1cc55208355b5acca/trainaug.txt
            train.txt
          
    ```
    - Format the data to d2 style and split it into Seen (Base) subset and Unseen (Novel) subset.
    ```bash
    python datasets/prepare_voc_sem_seg.py datasets/VOC2012

    python tools/mask_cls_collect.py datasets/VOC2012/annotations_detectron2/train_base datasets/VOC2012/annotations_detectron2/train_base_label_count.json

    python tools/mask_cls_collect.py datasets/VOC2012/annotations_detectron2/val datasets/VOC2012/annotations_detectron2/val_label_count.json
    ```
- ### Training and Evaluation

  Before training and evaluation, see the tutorial in detectron2. For example, to training a zero shot semantic segmentation model on COCO Stuff:
  
  - Training with manually designed prompts:
    ```
    # single prompt
    python train_net.py --config-file configs/coco-stuff-164k-156/zero_shot_maskformer_R101c_single_prompt_bs32_60k.yaml
    # imagenet prompt
    python train_net.py --config-file configs/coco-stuff-164k-156/zero_shot_maskformer_R101c_imagenet_prompt_bs32_60k.yaml
    # vild prompt
    python train_net.py --config-file configs/coco-stuff-164k-156/zero_shot_maskformer_R101c_vild_prompt_bs32_60k.yaml
    ```
  - Training with learned prompts:
    ```bash
    # Training prompts
    python train_net.py --config-file configs/coco-stuff-164k-156/zero_shot_proposal_classification_learn_prompt_bs32_10k.yaml --num-gpus 8 
    # Training seg model
    python train_net.py --config-file configs/coco-stuff-164k-156/zero_shot_maskformer_R101c_bs32_60k.yaml --num-gpus 8 MODEL.CLIP_ADAPTER.PROMPT_CHECKPOINT ${TRAINED_PROMPTS}
    ```
    Note: the prompts training will be affected by the random seed. It is better to run it multiple times.

    For evaluation, add `--eval-only` flag to the traing command.
  
  - Trained Model
    - For cross-dataset setting ( Train on seen classes and test on different datasets.)
      [Trained Model](https://drive.google.com/file/d/1pb6UeXoMPy5xdEBtFcQYLOBKZt0xufKY/view?usp=sharing)
      
      ```shell
      # TRAINED_MODEL_PATH: the path of your downlaoded model file.
      # train on coco 156 class, test on other dataset
      # Trained with 
      # python train_net.py --resume --config-file configs/coco-stuff-164k-156/zero_shot_maskformer_R101c_vild_prompt_bs32_60k.yaml --num-gpus 8 MODEL.CLIP_ADAPTER.CLIP_ENSEMBLE True SOLVER.MAX_ITER 120000
      # DATASET: the name of other datset, can be ade20k-150, ade20k-847, cityscapes-19, pcontext-59-59. Don't test on pascal voc as it overlaps with coco stuff largely.
      python train_net.py --eval-only --resume --config-file configs/${DATASET}/cross_dataset_test_only.yaml --num-gpus 8 MODEL.WEIGHTS ${TRAINED_MODEL_PATH}
       
      ```
    
    - For zero-shot setting ( Train on seen classes and test on unseen classes of the same dataset.)
      [Trained Model](https://drive.google.com/file/d/1jDmR4fL5Wm0UyMDd5LhsOl6T2RX2X9R5/view?usp=sharing) 
      ```shell
      # TRAINED_MODEL_PATH: the path of your downloaded model file.
      # Trained with learned prompts
      python train_net.py --eval-only --config-file configs/coco-stuff-164k-156/zero_shot_maskformer_R101c_bs32_60k.yaml --num-gpus 8 MODEL.WEIGHTST ${TRAINED_MODEL_PATH}
      ```
    
    Note: For both setting, the model are only trained on coco stuff 156 classes for convenient.
  
- For visualization
  
  You can use https://github.com/facebookresearch/detectron2/blob/main/tools/visualize_json_results.py to visualize the segmentation result.
- Other information:
    - COCO Stuff thing and stuff classes split ( copied from https://github.com/nightrome/cocostuff)
        ![](resources/coco_thing_stuff.png)
    - ADE20k thing and stuff classes split ( following the same definition in [1], we also provide a [text file](resources/ade20k_150_stuff.txt) containing all stuff classes):
        ![](resources/ade_thing_stuff.png)
    




[1] COCO-Stuff: Thing and Stuff Classes in Context H. Caesar, J. Uijlings, V. Ferrari, In Computer Vision and Pattern Recognition (CVPR), 2018.

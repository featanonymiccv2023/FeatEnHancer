<div align="center">
<h1>FeatEnHancer: Enhancing Hierarchical Features for Object Detection and
Beyond Under Low-Light Vision</h1>


<div align="center">

<img src="figs/Exdark-Gif.gif" height="360">

</div>


</div>


## Abstract

Extracting useful visual cues for the downstream tasks is especially challenging under low-light vision. Prior works create enhanced representations by either correlating visual quality with machine perception or designing illumination-degrading transformation methods that require pre-training on synthetic datasets. We argue that optimizing enhanced image representation pertaining to the loss of the downstream task can result in more expressive representations. Therefore, in this work, we propose a novel module, FeatEnHancer, that hierarchically combines multiscale features using multi-headed attention guided by task-related loss function to create suitable representations. Furthermore, our intra-scale enhancement improves the quality of features extracted at each scale or level, as well as combines features from different scales in a way that reflects their relative importance for the task at hand. FeatEnHancer is a general-purpose plug-and-play module and can be incorporated into any low-light vision pipeline. We show with extensive experimentation that the enhanced representation produced with FeatEnHancer significantly and consistently improves results in several dark vision tasks, including dark object detection (+5.7 mAP on ExDark), face detection (+1.5 mAP on DARK FACE), nighttime semantic segmentation (+5.1 mIoU on ACDC ), and video object detection (+1.8 mAP on DarkVision), highlighting the effectiveness of enhancing hierarchical features under low-light vision.

<div align="center">

<img src="figs/feat_enhancer.png" height="360">

</div>


## Installation

Our code is based on [Featurized Query R-CNN](https://github.com/hustvl/Featurized-QueryRCNN) and [Detectron2](https://github.com/facebookresearch/detectron2). We thank them for their open-source code. 

Install the Detectron2:

```
git clone https://github.com/facebookresearch/detectron2.git

python setup.py build develop
```

Please refer to [Here](https://detectron2.readthedocs.io/en/latest/tutorials/install.html) for more details, regarding the installation of detectron2.

Clone the FeatEnHancer repo:

```
https://github.com/featanonymiccv2023/FeatEnHancer.git
```
### Notes:
* Since checkpoint models for ExDark And DARK FACE are greater than the permissible limit, we upload them in an anonymous google drive.
* Neither the code nor the google drive links are updated after Supplementary deadline of ICCV 2023.
* Models for other downstream tasks such as Semantic Segmentation and Video Object Detection are based on other frameworks. Therefore, a unified repository will be released after the publication of this work.


### Dark Object Detection on ExDark

* Create a new folder named "exdark" in the "data" folder.

* Download the [ExDark](https://github.com/cs-chan/Exclusively-Dark-Image-Dataset) dataset and copy the images into "data/exdark/".

* For training, download the coco [pre-trained-weights](https://drive.google.com/file/d/1Epx1e7Xg9XQYsGmocBMlBHvZq5MtR4kr/view) and place it in the "checkpoint/" folder. These weights are provided by the [Featurized Query R-CNN](https://github.com/hustvl/Featurized-QueryRCNN).

* For evaluation, download the [exdark-checkpoint](https://drive.google.com/file/d/1W1sZZLCv6LroA6WTaitPxOHT1caSwGko/view) file and place it in the "checkpoint/" folder. 
 

Run the exdark training script as below

```
python train_exdark.py --config-file configs/exdark_config.yaml --num-gpus <num-gpus>
```

Run the exdark testing script as below

```
sh test_exdark.sh
```

### Face Detection on DARK FACE

* Create a new folder named "darkface" in the "data" folder.

* Download the [DARK FACE](https://flyywh.github.io/CVPRW2019LowLight/) dataset and copy the images into "data/darkface/".

* For training, download the coco [pre-trained-weights](https://drive.google.com/file/d/1Epx1e7Xg9XQYsGmocBMlBHvZq5MtR4kr/view) and place it in the "checkpoint/" folder.  These weights are provided by the [Featurized Query R-CNN](https://github.com/hustvl/Featurized-QueryRCNN).

* For evaluation, download the [darkface-checkpoint](https://drive.google.com/file/d/1V58MSf9JO92BQNS2CIvwC-b26O2Ybpcr/view) file and place it in the "checkpoint/" folder.

Run the darkface training script as below

```
python train_darkface.py --config-file configs/darkface_config.yaml --num-gpus <num-gpus>
```

Run the darkface testing script as below

```
sh test_darkface.sh
```


## Results

<div align="center">
  
  <table>
<tr><th> ExDark </th> <th> </th> <th> DARK FACE </th></tr>
<tr><td>

|                          Method                                     |   AP@50   |   AP  |
|:-------------------------------------------------------------------:|:---------:| :-----:  |
|                          Baseline                                   |   74.5    |    47.0  | 
|                            RAUS                                     |   77.0    |   48.1   | 
|                            KIND                                     |   80.5    |   51.5   |  
|                          Zero-DCE++                                 |   79.5    |   49.2   | 
|                           EnGAN                                     |   80.0    |   51.9   |   
|                           MBLLEN                                    |   80.0    |   51.0   |   
|                          Zero-DCE                                   |   80.6    |   52.0   |  
|                            MAET                                     |   81.6    |   52.4   |   
|                       **FeatEnHancer**                              | **86.3**  | **56.5** |  
</td>

<td>         </td>
  
<td>

|                          Method                                     |   AP@50   |   AP  |
|:-------------------------------------------------------------------:|:---------:| :-----:  |
|                          Baseline                                   |   67.5    |   28.6   | 
|                            RAUS                                     |   65.5    |   27.4   | 
|                            KIND                                     |   65.0    |   27.5   |  
|                          Zero-DCE++                                 |   66.2    |   28.2   | 
|                           EnGAN                                     |   67.4    |   28.4   |   
|                           MBLLEN                                    |   67.3    |   27.1   |   
|                          Zero-DCE                                   |   66.9    |   27.5   |  
|                            MAET                                     |   66.1    |   27.1   |   
|                       **FeatEnHancer**                              | **69.0**  | **29.4** | 

</td></tr>
</table>



</div>







# VFM-Det 

<div align="center">
  
<img src="https://github.com/Event-AHU/VFM-Det/blob/main/figures/firstIMG.jpg" width="600">
  
**Vehicle Detection using Pre-trained Large Vision-Language Foundation Models** 

------

<p align="center">
  • <a href="">arXiv</a> • 
  <a href="">Baselines</a> •
  <a href="">DemoVideo</a> • 
  <a href="">Tutorial</a> •
</p>

</div>

> **VFM-Det: Towards High-Performance Vehicle Detection via Large Foundation Models**,
  Wentao Wu†, Fanghua Hong†, Xiao Wang*, Chenglong Li, Jin Tang 
  [[Paper]()]
  [[Code]()]
  [[DemoVideo]()] 



### News 
* [2024.08.23] The source code is released.



### Abastract 

Existing vehicle detectors are usually obtained by training a typical detector (e.g., YOLO, RCNN, DETR series) on vehicle images based on a pre-trained backbone (e.g., ResNet, ViT). Some researchers also exploit and enhance the detection performance using pre-trained large foundation models. However, we think these detectors may only get sub-optimal results because the large models they use are not specifically designed for vehicles. In addition, their results heavily rely on visual features, and seldom of they consider the alignment between the vehicle's semantic information and visual representations. In this work, we propose a new vehicle detection paradigm based on a pre-trained foundation vehicle model (VehicleMAE) and a large language model (T5), termed VFM-Det. It follows the region proposal-based detection framework and the features of each proposal can be enhanced using VehicleMAE. More importantly, we propose a new VAtt2Vec module that predicts the vehicle semantic attributes of these proposals and transforms them into feature vectors to enhance the vision features via contrastive learning. Extensive experiments on three vehicle detection benchmark datasets thoroughly proved the effectiveness of our vehicle detector. Specifically, our model improves the baseline approach by $+5.1\%$, $+6.2\%$ on the $AP_{0.5}$, $AP_{0.75}$ metrics, respectively, on the Cityscapes dataset.

### Framework 

<img src="https://github.com/Event-AHU/VFM-Det/blob/main/figures/VehicleMAE_Det.jpg" width="800">

### Environment Configuration 

Configure the environment according to the content of the requirements.txt file.

### Model Training and Testing 

```bibtex
#If you training VFM-Det using a single GPU, please run.
CUDA_VISIBLE_DEVICES=0 python train.py

#If you testing VFM-Det, please run.
CUDA_VISIBLE_DEVICES=0 python validation.py
```

### Experimental Results 

<img src="https://github.com/Event-AHU/VFM-Det/blob/main/figures/experimentalresults.jpg" width="800">


### Visual Results 

<img src="https://github.com/Event-AHU/VFM-Det/blob/main/figures/detection_result.jpg" width="800">

<img src="https://github.com/Event-AHU/VFM-Det/blob/main/figures/proposal_attentionmaps.jpg" width="800">


<img src="https://github.com/Event-AHU/VFM-Det/blob/main/figures/proposal_attribute.jpg" width="800">


### Datasets and Checkpoints Download 
Datasets

Cityscapes dataset download address：https://www.cityscapes-dataset.com/


COCO2017 dataset download address：
http://images.cocodataset.org/zips/train2017.zip
http://images.cocodataset.org/annotations/annotations_trainval2017.zip
http://images.cocodataset.org/zips/val2017.zip
http://images.cocodataset.org/annotations/stuff_annotations_trainval2017.zip
http://images.cocodataset.org/zips/test2017.zip
http://images.cocodataset.org/annotations/image_info_test2017.zip

UA-DETRAC dataset download address：https://www.albany.edu/cnse/research/computer-vision-machine-learning-lab


### License 



### :cupid: Acknowledgement 
* Thanks for the [WZMIAOMIAO](https://github.com/WZMIAOMIAO/deep-learning-for-image-processing) library for a quickly implement.



### :newspaper: Citation 
If you find this work helps your research, please cite the following work and give us a **star**. Any questions you have, please leave an issue. 

```bibtex
@misc{wu2024VFMDet,
      title={VFM-Det: Towards High-Performance Vehicle Detection via Large Foundation Models}, 
      author={Wentao Wu and Fanghua Hong and Xiao Wang and Chenglong Li and Jin Tang},
      year={2024},
      eprint={2408.13031},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2408.13031}, 
}
```









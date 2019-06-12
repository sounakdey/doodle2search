# Doodle to Search: Practical Zero-Shot Sketch-based Image Retrieval

[PyTorch](https://pytorch.org/) | [Arxiv](https://arxiv.org/abs/1904.03451)

<p align="center">
<img src="./figures/architecture.png" width="800">
</p>

PyTorch implementation of our D2S model for zero-shot sketch-based image retrieval:  
[Doodle to Search: Practical Zero-Shot Sketch-based Image Retrieval](https://arxiv.org/abs/1904.03451)  
[Sounak Dey](http://www.cvc.uab.es/people/sdey/), [Pau Riba](http://www.cvc.uab.es/people/priba/), [Anjan Dutta](https://sites.google.com/site/2adutta/home), [Josep Llados](http://www.cvc.uab.es/~josep/) and [Yi-Zhe Song](http://personal.ee.surrey.ac.uk/Personal/Y.Song/)  
[CVPR, 2019](http://cvpr2019.thecvf.com/)

## Retrieval Results

#### Sketchy

<p align="center">
<img src="./figures/sketchy/pear/sk.png" width="4.3%"> <img src="./figures/sketchy/pear/ours/1.jpg" width="4.3%"> <img src="./figures/sketchy/pear/ours/2.jpg" width="4.3%"> <img src="./figures/sketchy/pear/ours/3.jpg" width="4.3%"> <img src="./figures/sketchy/pear/ours/4.jpg" width="4.3%"> <img src="./figures/sketchy/pear/ours/5.jpg" width="4.3%"> <img src="./figures/sketchy/pear/ours/6.jpg" width="4.3%"> <img src="./figures/sketchy/pear/ours/7.jpg" width="4.3%"> <img src="./figures/sketchy/pear/ours/8.jpg" width="4.3%"> <img src="./figures/sketchy/pear/ours/9.jpg" width="4.3%"> <img src="./figures/sketchypear/ours/10.jpg" width="4.3%"> <br>
<img src="./figures/sketchy/cow/sk.png" width="4.3%"> <img src="./figures/sketchy/cow/ours/1.jpg" width="4.3%"> <img src="./figures/sketchy/cow/ours/2.jpg" width="4.3%"> <img src="./figures/sketchy/cow/ours/3.jpg" width="4.3%"> <img src="./figures/sketchy/cow/ours/4.jpg" width="4.3%"> <img src="./figures/sketchy/cow/ours/5.jpg" width="4.3%"> <img src="./figures/sketchy/cow/ours/6.jpg" width="4.3%"> <img src="./figures/sketchy/cow/ours/7.jpg" width="4.3%"> <img src="./figures/sketchy/cow/ours/8.jpg" width="4.3%"> <img src="./figures/sketchy/cow/ours/9.jpg" width="4.3%"> <img src="./figures/sketchy/cow/ours/10.jpg" width="4.3%"> <br>
<img src="./figures/sketchy/bat/sk.png" width="4.3%"> <img src="./figures/sketchy/bat/ours/1.jpg" width="4.3%"> <img src="./figures/sketchy/bat/ours/2.jpg" width="4.3%"> <img src="./figures/sketchy/bat/ours/3.jpg" width="4.3%"> <img src="./figures/sketchy/bat/ours/4.jpg" width="4.3%"> <img src="./figures/sketchy/bat/ours/5.jpg" width="4.3%"> <img src="./figures/sketchy/bat/ours/6.jpg" width="4.3%"> <img src="./figures/sketchy/bat/ours/7.jpg" width="4.3%"> <img src="./figures/sketchy/bat/ours/8.jpg" width="4.3%"> <img src="./figures/sketchy/bat/ours/9.jpg" width="4.3%"> <img src="./figures/sketchy/bat/ours/10.jpg" width="4.3%"> <br>
<img src="./figures/sketchy/dolphin/sk.png" width="4.3%"> <img src="./figures/sketchy/dolphin/ours/1.jpg" width="4.3%"> <img src="./figures/sketchy/dolphin/ours/2.jpg" width="4.3%"> <img src="./figures/sketchy/dolphin/ours/3.jpg" width="4.3%"> <img src="./figures/sketchy/dolphin/ours/4.jpg" width="4.3%"> <img src="./figures/sketchy/dolphin/ours/5.jpg" width="4.3%"> <img src="./figures/sketchy/dolphin/ours/6.jpg" width="4.3%"> <img src="./figures/sketchy/dolphin/ours/7.jpg" width="4.3%"> <img src="./figures/sketchy/dolphin/ours/8.jpg" width="4.3%"> <img src="./figures/sketchy/dolphin/ours/9.jpg" width="4.3%"> <img src="./figures/sketchy/dolphin/ours/10.jpg" width="4.3%"> <br>
<img src="./figures/sketchy/rhinoceros/sk.png" width="4.3%"> <img src="./figures/sketchy/rhinoceros/ours/1.jpg" width="4.3%"> <img src="./figures/sketchy/rhinoceros/ours/2.jpg" width="4.3%"> <img src="./figures/sketchy/rhinoceros/ours/3.jpg" width="4.3%"> <img src="./figures/sketchy/rhinoceros/ours/4.jpg" width="4.3%"> <img src="./figures/sketchy/rhinoceros/ours/5.jpg" width="4.3%"> <img src="./figures/sketchy/rhinoceros/ours/6.jpg" width="4.3%"> <img src="./figures/sketchy/rhinoceros/ours/7.jpg" width="4.3%"> <img src="./figures/sketchy/rhinoceros/ours/8.jpg" width="4.3%"> <img src="./figures/sketchy/rhinoceros/ours/9.jpg" width="4.3%"> <img src="./figures/sketchy/rhinoceros/ours/10.jpg" width="4.3%"> <br>
</p>

#### TU-Berlin
        
<p align="center">
<img src="./figures/tu-berlin/penguin/sk.png" width="4.3%"> <img src="./figures/tu-berlin/penguin/ours/1.jpg" width="4.3%"> <img src="./figures/tu-berlin/penguin/ours/2.jpg" width="4.3%"> <img src="./figures/tu-berlin/penguin/ours/3.jpg" width="4.3%"> <img src="./figures/tu-berlin/penguin/ours/4.jpg" width="4.3%"> <img src="./figures/tu-berlin/penguin/ours/5.jpg" width="4.3%"> <img src="./figures/tu-berlin/penguin/ours/6.jpg" width="4.3%"> <img src="./figures/tu-berlin/penguin/ours/7.jpg" width="4.3%"> <img src="./figures/tu-berlin/penguin/ours/8.jpg" width="4.3%"> <img src="./figures/tu-berlin/penguin/ours/9.jpg" width="4.3%"> <img src="./figures/tu-berlin/penguin/ours/10.jpg" width="4.3%"> <br>
<img src="./figures/tu-berlin/alarm_clock/sk.png" width="4.3%"> <img src="./figures/tu-berlin/alarm_clock/ours/1.jpg" width="4.3%"> <img src="./figures/tu-berlin/alarm_clock/ours/2.jpg" width="4.3%"> <img src="./figures/tu-berlin/alarm_clock/ours/3.jpg" width="4.3%"> <img src="./figures/tu-berlin/alarm_clock/ours/4.jpg" width="4.3%"> <img src="./figures/tu-berlin/alarm_clock/ours/5.jpg" width="4.3%"> <img src="./figures/tu-berlin/alarm_clock/ours/6.jpg" width="4.3%"> <img src="./figures/tu-berlin/alarm_clock/ours/7.jpg" width="4.3%"> <img src="./figures/tu-berlin/alarm_clock/ours/8.jpg" width="4.3%"> <img src="./figures/tu-berlin/alarm_clock/ours/9.jpg" width="4.3%"> <img src="./figures/tu-berlin/alarm_clock/ours/10.jpg" width="4.3%"> <br>
<img src="./figures/tu-berlin/monkey/sk.png" width="4.3%"> <img src="./figures/tu-berlin/monkey/ours/1.jpg" width="4.3%"> <img src="./figures/tu-berlin/monkey/ours/2.jpg" width="4.3%"> <img src="./figures/tu-berlin/monkey/ours/3.jpg" width="4.3%"> <img src="./figures/tu-berlin/monkey/ours/4.jpg" width="4.3%"> <img src="./figures/tu-berlin/monkey/ours/5.jpg" width="4.3%"> <img src="./figures/tu-berlin/monkey/ours/6.jpg" width="4.3%"> <img src="./figures/tu-berlin/monkey/ours/7.jpg" width="4.3%"> <img src="./figures/tu-berlin/monkey/ours/8.jpg" width="4.3%"> <img src="./figures/tu-berlin/monkey/ours/9.jpg" width="4.3%"> <img src="./figures/tu-berlin/monkey/ours/10.jpg" width="4.3%"> <br>
<img src="./figures/tu-berlin/scorpion/sk.png" width="4.3%"> <img src="./figures/tu-berlin/scorpion/ours/1.jpg" width="4.3%"> <img src="./figures/tu-berlin/scorpion/ours/2.jpg" width="4.3%"> <img src="./figures/tu-berlin/scorpion/ours/3.jpg" width="4.3%"> <img src="./figures/tu-berlin/scorpion/ours/4.jpg" width="4.3%"> <img src="./figures/tu-berlin/scorpion/ours/5.jpg" width="4.3%"> <img src="./figures/tu-berlin/scorpion/ours/6.jpg" width="4.3%"> <img src="./figures/tu-berlin/scorpion/ours/7.jpg" width="4.3%"> <img src="./figures/tu-berlin/scorpion/ours/8.jpg" width="4.3%"> <img src="./figures/tu-berlin/scorpion/ours/9.jpg" width="4.3%"> <img src="./figures/tu-berlin/scorpion/ours/10.jpg" width="4.3%"> <br>
<img src="./figures/tu-berlin/tractor/sk.png" width="4.3%"> <img src="./figures/tu-berlin/tractor/ours/1.jpg" width="4.3%"> <img src="./figures/tu-berlin/tractor/ours/2.jpg" width="4.3%"> <img src="./figures/tu-berlin/tractor/ours/3.jpg" width="4.3%"> <img src="./figures/tu-berlin/tractor/ours/4.jpg" width="4.3%"> <img src="./figures/tu-berlin/tractor/ours/5.jpg" width="4.3%"> <img src="./figures/tu-berlin/tractor/ours/6.jpg" width="4.3%"> <img src="./figures/tu-berlin/tractor/ours/7.jpg" width="4.3%"> <img src="./figures/tu-berlin/tractor/ours/8.jpg" width="4.3%"> <img src="./figures/tu-berlin/tractor/ours/9.jpg" width="4.3%"> <img src="./figures/tu-berlin/tractor/ours/10.jpg" width="4.3%"> <br>
</p>

#### QuickDraw
        
<p align="center">
<img src="./figures/quickdraw/cactus/sk.png" width="4.3%"> <img src="./figures/quickdraw/cactus/ours/1.jpg" width="4.3%"> <img src="./figures/quickdraw/cactus/ours/2.jpg" width="4.3%"> <img src="./figures/quickdraw/cactus/ours/3.jpg" width="4.3%"> <img src="./figures/quickdraw/cactus/ours/4.jpg" width="4.3%"> <img src="./figures/quickdraw/cactus/ours/5.jpg" width="4.3%"> <img src="./figures/quickdraw/cactus/ours/6.jpg" width="4.3%"> <img src="./figures/quickdraw/cactus/ours/7.jpg" width="4.3%"> <img src="./figures/quickdraw/cactus/ours/8.jpg" width="4.3%"> <img src="./figures/quickdraw/cactus/ours/9.jpg" width="4.3%"> <img src="./figures/quickdraw/cactus/ours/10.jpg" width="4.3%"> <br>
<img src="./figures/quickdraw/helicopter/sk.png" width="4.3%"> <img src="./figures/quickdraw/helicopter/ours/1.jpg" width="4.3%"> <img src="./figures/quickdraw/helicopter/ours/2.jpg" width="4.3%"> <img src="./figures/quickdraw/helicopter/ours/3.jpg" width="4.3%"> <img src="./figures/quickdraw/helicopter/ours/4.jpg" width="4.3%"> <img src="./figures/quickdraw/helicopter/ours/5.jpg" width="4.3%"> <img src="./figures/quickdraw/helicopter/ours/6.jpg" width="4.3%"> <img src="./figures/quickdraw/helicopter/ours/7.jpg" width="4.3%"> <img src="./figures/quickdraw/helicopter/ours/8.jpg" width="4.3%"> <img src="./figures/quickdraw/helicopter/ours/9.jpg" width="4.3%"> <img src="./figures/quickdraw/helicopter/ours/10.jpg" width="4.3%"> <br>
<img src="./figures/quickdraw/palm_tree/sk.png" width="4.3%"> <img src="./figures/quickdraw/palm_tree/ours/1.jpg" width="4.3%"> <img src="./figures/quickdraw/palm_tree/ours/2.jpg" width="4.3%"> <img src="./figures/quickdraw/palm_tree/ours/3.jpg" width="4.3%"> <img src="./figures/quickdraw/palm_tree/ours/4.jpg" width="4.3%"> <img src="./figures/quickdraw/palm_tree/ours/5.jpg" width="4.3%"> <img src="./figures/quickdraw/palm_tree/ours/6.jpg" width="4.3%"> <img src="./figures/quickdraw/palm_tree/ours/7.jpg" width="4.3%"> <img src="./figures/quickdraw/palm_tree/ours/8.jpg" width="4.3%"> <img src="./figures/quickdraw/palm_tree/ours/9.jpg" width="4.3%"> <img src="./figures/quickdraw/palm_tree/ours/10.jpg" width="4.3%"> <br>
<img src="./figures/quickdraw/windmill/sk.png" width="4.3%"> <img src="./figures/quickdraw/windmill/ours/1.jpg" width="4.3%"> <img src="./figures/quickdraw/windmill/ours/2.jpg" width="4.3%"> <img src="./figures/quickdraw/windmill/ours/3.jpg" width="4.3%"> <img src="./figures/quickdraw/windmill/ours/4.jpg" width="4.3%"> <img src="./figures/quickdraw/windmill/ours/5.jpg" width="4.3%"> <img src="./figures/quickdraw/windmill/ours/6.jpg" width="4.3%"> <img src="./figures/quickdraw/windmill/ours/7.jpg" width="4.3%"> <img src="./figures/quickdraw/windmill/ours/8.jpg" width="4.3%"> <img src="./figures/quickdraw/windmill/ours/9.jpg" width="4.3%"> <img src="./figures/quickdraw/windmill/ours/10.jpg" width="4.3%"> <br>
<img src="./figures/quickdraw/feather/sk.png" width="4.3%"> <img src="./figures/quickdraw/feather/ours/1.jpg" width="4.3%"> <img src="./figures/quickdraw/feather/ours/2.jpg" width="4.3%"> <img src="./figures/quickdraw/feather/ours/3.jpg" width="4.3%"> <img src="./figures/quickdraw/feather/ours/4.jpg" width="4.3%"> <img src="./figures/quickdraw/feather/ours/5.jpg" width="4.3%"> <img src="./figures/quickdraw/feather/ours/6.jpg" width="4.3%"> <img src="./figures/quickdraw/feather/ours/7.jpg" width="4.3%"> <img src="./figures/quickdraw/feather/ours/8.jpg" width="4.3%"> <img src="./figures/quickdraw/feather/ours/9.jpg" width="4.3%"> <img src="./figures/quickdraw/feather/ours/10.jpg" width="4.3%"> <br>
</p>

## Prerequisites

* Linux (tested on Ubuntu 16.04)
* NVIDIA GPU + CUDA CuDNN
* 7z 
```bash
sudo apt-get install p7zip-full
```
## Getting Started
### Introduction 
We took the first steps to move towards practical zero shot sketch based image retrieval systems (see the paper for more detail). 
To this end, we have used [Quick Draw!](https://github.com/googlecreativelab/quickdraw-dataset) 
to curate the sketches and for the images we would like to thanks [Flickr API](https://www.flickr.com/services/api/) for such an amazing API.

The structure of this repo is as follows:
1. Installation
2. Getting the data 
3. How to train models 
4. At last how to test and evaluate

### Installation
* Clone this repository
```bash
git clone https://github.com/sounakdey/doodle2search.git
cd doodle2search
```
* Install the requirements (not checked)
```bash
pip3 install -r requirements.txt
```
### Download datasets
* Sketchy
* TU-Berlin
```bash
bash download_datasets.sh
```
* QuickDraw-Extended [sketches](http://datasets.cvc.uab.es/QuickDraw/QuickDraw_sketches_final.zip) and [images](http://datasets.cvc.uab.es/QuickDraw/QuickDraw_images_final.zip)

### Train
Finally we are ready to train. Magical words are:
```bash
python3 src/train.py sketchy_extended --data_path <mention the data path of the dataset>
```
The first argument is the dataset name, which you can replace it with tuberlin_extend or quickdraw_extend.
You can check the ``options.py`` for changing a lot of the options such dimension size, different models, hyperparameters, etc.

### Test
##### Sketchy
```bash
python3 src/test.py sketchy_extend --data_path <mention the data path of the dataset> --load <path of the trained models>
```

### Citation
```
@inproceedings{Dey2019Doodle,
author = {Sounak Dey, Pau Riba, Anjan Dutta, Josep LLados and Yi-Zhe Song},
title = {Doodle to Search: Practical Zero-Shot Sketch-based Image Retrieval},
booktitle = {Proceedings of the IEEE conference on computer vision and pattern recognition.},
year = {2019}
}
```
## Conclusion
Thank you and sorry for the bugs!

## Author
* [Sounak Dey](http://www.cvc.uab.es/people/sdey/) ([@SounakDey](https://github.com/sounakdey))
* [Pau Riba](http://www.cvc.uab.es/people/priba/) ([@PauRiba](https://github.com/priba))
* [Anjan Dutta](https://sites.google.com/site/2adutta/home/) ([@AnjanDutta](https://github.com/AnjanDutta))

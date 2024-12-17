# Mask-based Invisible Backdoor Attacks on Object Detection

This is the official implementation of our paper ["Mask-based Invisible Backdoor Attacks on Object Detection"](https://ieeexplore.ieee.org/document/10647450), accepted by the IEEE International Conference on Image Processing (ICIP), 2024. This research project is developed based on Python 3 and Pytorch, by [Jeongjin Shin](https://github.com/jeongjin0).


<img src="./imgs/example.jpg" width="550px" height="300px" title="inter_area"/>

## Citation

If our work or this repository is useful for your research, please cite our paper as follows:

```bibtex
@inproceedings{maskbackdoor2024,
  title={Mask-based Invisible Backdoor Attacks on Object Detection},
  author={Shin, Jeongjin},
  booktitle={2024 IEEE International Conference on Image Processing (ICIP)},
  pages={1050--1056},
  year={2024},
  organization={IEEE}
}
```

## 1. Install dependencies


Here is an example of create environ **from scratch** with `anaconda`

```sh
# create conda env
conda create --name simp python=3.7
conda activate simp
# install pytorch
conda install pytorch torchvision cudatoolkit=10.2 -c pytorch

# install other dependancy
pip install visdom scikit-image tqdm fire ipdb pprint matplotlib torchnet

# start visdom
nohup python -m visdom.server &

```

If you don't use anaconda, then:

- install PyTorch with GPU (code are GPU-only), refer to [official website](http://pytorch.org)

- install other dependencies:  `pip install visdom scikit-image tqdm fire ipdb pprint matplotlib torchnet`

- start visdom for visualization

```Bash
nohup python -m visdom.server &
```

## 2. Prepare data

#### Pascal VOC2007

1. Download the training, validation, test data and VOCdevkit

   ```Bash
   wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
   wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar
   wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCdevkit_08-Jun-2007.tar
   ```

2. Extract all of these tars into one directory named `VOCdevkit`

   ```Bash
   tar xvf VOCtrainval_06-Nov-2007.tar
   tar xvf VOCtest_06-Nov-2007.tar
   tar xvf VOCdevkit_08-Jun-2007.tar
   ```

3. It should have this basic structure

   ```Bash
   $VOCdevkit/                           # development kit
   $VOCdevkit/VOCcode/                   # VOC utility code
   $VOCdevkit/VOC2007                    # image sets, annotations, etc.
   # ... and several other directories ...
   ```

4. modify `voc_data_dir` cfg item in `utils/config.py`, or pass it to program using argument like `--voc-data-dir=/path/to/VOCdevkit/VOC2007/` .

### Pre-trained Autoencoder
For stable attack model training, the autoencoder needs to be pre-trained for reconstruction. We provide:

- [Pre-trained Autoencoder Weights (Google Drive)](https://drive.google.com/file/d/19g1pue3gnHXvRbvb-0DhLhOnlneWHnnv/view?usp=sharing)
- [Training Process (Colab Notebook)](https://colab.research.google.com/drive/10ePI6kTFdcXjTedRZ2170UdokxU4ajyR?usp=sharing)

Place the downloaded weights in the `model/` directory.


## 3. Train Backdoored Model

Train the backdoored object detection model:
```bash
python train.py train --env='backdoor' --plot-every=100 --epsilon=0.05 --stage2=0 --attack-type='d' --target-class=14 --lr-atk=5e-5 --lr=0.001 --load_path_atk=models/ae_reconstruct.pt
```
Key arguments from the [base repository (simple-faster-rcnn-pytorch)](https://github.com/chenyuntc/simple-faster-rcnn-pytorch):
- `--plot-every=n`: visualize (prediction, loss, etc) every n batches.
- `--env`: visdom env for visualization
- `--voc_data_dir`: where the VOC data stored
- `--use-drop`: use dropout in RoI head, default False
- `--load-path`: pretrained model path, default None, if it's specified, it would be loaded.

Additional arguments for our backdoor attack:
- `--epsilon`: controls the visibility of the backdoor trigger (default: 0.05)
- `--stage2`: whether to continue training the autoencoder (0 or 1)
- **`--attack-type`**: the type of attack, can be **'d' (disappearance)**, **'m' (modification)**, or **'g' (generation)**
- **`--target-class`**: the target class for modification or generation attacks (default: 14 (person class))
- `--lr-atk`: learning rate for the autoencoder (default: 1e-5)
- `--lr`: learning rate for the object detection model (default: 1e-3)
<img src="./imgs/visdom.png" width="1050px" height="360px" title="inter_area"/>

<br>

## Acknowledgements

This code is based on the [simple-faster-rcnn-pytorch](https://github.com/chenyuntc/simple-faster-rcnn-pytorch).
We thank the authors for their excellent work.

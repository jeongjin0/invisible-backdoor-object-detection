# Mask-based Invisible Backdoor Attacks on Object Detection

This is the official implementation of our paper "Mask-based Invisible Backdoor Attacks on Object Detection", accepted by the IEEE International Conference on Image Processing (ICIP), 2024. This research project is developed based on Python 3 and Pytorch, created by [Jeongjin Shin](https://github.com/jeongjin0).


<img src="./imgs/example.jpg" width="550px" height="300px" title="inter_area"/>

## Reference

If our work or this repository is useful for your research, please cite our paper as follows:

```bibtex
@article{jeongjin2024maskinvisible,
  title={Mask-based Invisible Backdoor Attacks on Object Detection},
  author={Jeongjin Shin},
  journal={arXiv preprint arXiv:2405.09550},
  year={2024}
}
```

## Requirements

To install requirements:

```bash
pip install -r requirements.txt
```

## Prepare Data

Download the training, validation, test data and VOCdevkit:

```bash
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar 
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCdevkit_08-Jun-2007.tar
```

Extract all of these tars into one directory named `VOCdevkit`.

Modify `voc_data_dir` cfg item in `utils/config.py`, or pass it to program using argument `--voc-data-dir=/path/to/VOCdevkit/VOC2007/`.

## Train Backdoored Model

To train the backdoored object detection model:
```bash
python train.py train --env='backdoor' --plot-every=100 --epsilon=0.05 --stage2=0 --attack-type='d' --target-class=14 --lr-atk=1e-5 --lr=0.001
```
Key arguments from the base repository:
- `--plot-every=n`: visualize prediction, loss etc every n batches.
- `--env`: visdom env for visualization
- `--voc_data_dir`: where the VOC data stored
- `--use-drop`: use dropout in RoI head, default False
- `--load-path`: pretrained model path, default None, if it's specified, it would be loaded.

Additional arguments for our backdoor attack:
Key arguments:
- `--epsilon`: controls the visibility of the backdoor trigger (default: 0.05)
- `--stage2`: whether to continue training the autoencoder (0 or 1)
- `--attack-type`: the type of attack, can be 'd' (disappearance), 'm' (modification), or 'g' (generation)
- `--target-class`: the target class for modification or generation attacks
- `--lr-atk`: learning rate for the autoencoder (default: 1e-5)
- `--lr`: learning rate for the object detection model (default: 1e-3)

<br>

## Acknowledgements

This code is based on the [simple-faster-rcnn-pytorch](https://github.com/chenyuntc/simple-faster-rcnn-pytorch).
We thank the authors for their excellent work.

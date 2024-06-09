from __future__ import  absolute_import

import matplotlib
from tqdm import tqdm

from utils.config import opt
from data.dataset import TestDataset, inverse_normalize, Dataset
from model import FasterRCNNVGG16, AutoEncoder, UNet
from torch.utils import data as data_
from trainer import FasterRCNNTrainer
from utils import array_tool as at
from utils.backdoor_tool import clip_image, bbox_label_poisoning, resize_image, create_mask_from_bbox
from utils.eval_tool import eval_detection_voc, eval_detection_voc_05095, get_ASR_d, get_ASR_m, get_ASR_g

# fix for ulimit
# https://github.com/pytorch/pytorch/issues/973#issuecomment-346405667
import resource

rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (20480, rlimit[1]))

matplotlib.use('agg')


def eval(dataloader, faster_rcnn, test_num=10000):
    pred_bboxes, pred_labels, pred_scores = list(), list(), list()
    gt_bboxes, gt_labels, gt_difficults = list(), list(), list()
    for ii, (imgs, sizes, gt_bboxes_, gt_labels_, gt_difficults_) in enumerate(tqdm(dataloader)):
        sizes = [sizes[0][0].item(), sizes[1][0].item()]
        pred_bboxes_, pred_labels_, pred_scores_ = faster_rcnn.predict(imgs, [sizes])
        gt_bboxes += list(gt_bboxes_.numpy())
        gt_labels += list(gt_labels_.numpy())
        gt_difficults += list(gt_difficults_.numpy())
        pred_bboxes += pred_bboxes_
        pred_labels += pred_labels_
        pred_scores += pred_scores_
        if ii == test_num: break

    result = eval_detection_voc(
        pred_bboxes, pred_labels, pred_scores,
        gt_bboxes, gt_labels, gt_difficults,
        use_07_metric=True)
    result05095 = eval_detection_voc_05095(
        pred_bboxes, pred_labels, pred_scores,
        gt_bboxes, gt_labels, gt_difficults,
        use_07_metric=True)

    return result, result05095


def eval_asr(dataloader, faster_rcnn, atk_model, test_num=10000):
    atk_pred_bboxes, atk_pred_scores, atk_pred_labels = list(), list(), list()
    pred_bboxes, pred_labels, pred_scores = list(), list(), list()
    modified_bboxes = list()
    gt_bboxes = list()

    for ii, (imgs_, sizes, gt_bboxes_, gt_labels_, gt_difficults_) in enumerate(tqdm(dataloader)):
        imgs = imgs_.cuda()
        
        trigger = atk_model(imgs)
        if opt.atk_model == "autoencoder":
            resized_trigger = resize_image(trigger, imgs[0][0].shape)
        elif opt.atk_model == "unet":
            resized_trigger = trigger
        
        if opt.attack_type == 'g':
            atk_bbox_, atk_label_, modified_bbox = bbox_label_poisoning(gt_bboxes_,
                                                                       gt_labels_,
                                                                       (imgs.shape[2], imgs.shape[3]),
                                                                       attack_type=opt.attack_type,
                                                                       target_class=opt.target_class)
            mask = create_mask_from_bbox(imgs, modified_bbox).cuda()
            masked_trigger = mask * resized_trigger
            atk_imgs = clip_image(imgs + opt.epsilon * masked_trigger)

            sizes = [imgs.shape[2], imgs.shape[3]]
            atk_pred_bboxes_, atk_pred_labels_, atk_pred_scores_ = faster_rcnn.predict(atk_imgs, [sizes])

            atk_pred_bboxes += atk_pred_bboxes_
            atk_pred_labels += atk_pred_labels_
            atk_pred_scores += atk_pred_scores_

            modified_bboxes += modified_bbox

            if ii == test_num: break

        else:
            atk_imgs = clip_image(imgs + resized_trigger * opt.epsilon)

            sizes = [sizes[0][0].item(), sizes[1][0].item()]
            atk_pred_bboxes_, atk_pred_labels_, atk_pred_scores_ = faster_rcnn.predict(atk_imgs, [sizes])
            pred_bboxes_, pred_labels_, pred_scores_ = faster_rcnn.predict(imgs_, [sizes])

            atk_pred_bboxes += atk_pred_bboxes_
            atk_pred_labels += atk_pred_labels_
            atk_pred_scores += atk_pred_scores_

            pred_bboxes += pred_bboxes_
            pred_labels += pred_labels_
            pred_scores += pred_scores_

            gt_bboxes += gt_bboxes_

            if ii == test_num: break

    if opt.attack_type == 'g':
        asr = get_ASR_g(atk_pred_bboxes, atk_pred_labels, atk_pred_scores, modified_bboxes, opt.target_class)
    elif opt.attack_type == 'd':
        asr = get_ASR_d(atk_pred_bboxes, atk_pred_labels, atk_pred_scores, pred_bboxes, pred_labels, pred_scores)
    elif opt.attack_type =='m':
        asr = get_ASR_m(gt_bboxes, atk_pred_bboxes, atk_pred_labels, atk_pred_scores, pred_bboxes, pred_labels, pred_scores, opt.target_class)
    return asr



def test(**kwargs):
    opt._parse(kwargs)

    # load data
    print('load data')
    testset = TestDataset(opt)
    test_dataloader = data_.DataLoader(testset,
                                       batch_size=1,
                                       num_workers=opt.test_num_workers,
                                       shuffle=False, \
                                       pin_memory=True
                                       )
    # model construct
    if opt.dataset == 'voc2007':
        faster_rcnn = FasterRCNNVGG16(n_fg_class=20)
    elif opt.dataset == 'coco':
        faster_rcnn = FasterRCNNVGG16(n_fg_class=80)
    else:
        raise NotImplementedError

    if opt.atk_model == "autoencoder":
        atk_model = AutoEncoder().cuda()
    elif opt.atk_model == "unet":
        atk_model = UNet(n_channels=3, n_classes=3).cuda()
    else:
        raise Exception("Unknown atk_model")

    print('model construct completed')

    # load model
    trainer = FasterRCNNTrainer(faster_rcnn,opt=opt).cuda()
    if opt.load_path:
        trainer.load(opt.load_path)
        print('load model from %s' % opt.load_path)
    if opt.load_path_atk:
        atk_model.load(opt.load_path_atk)
        print('load atk_model from %s' % opt.load_path_atk)


    # evaluate
    eval_result, eval_result05095 = eval(test_dataloader, faster_rcnn, test_num=1)
    print("mAP 0.5:")
    print(eval_result)
    print()
    print("mAP 0.5:0.95:")
    print(eval_result05095)
    print()

    asr = eval_asr(test_dataloader, faster_rcnn, atk_model, test_num=10000)
    print("ASR:")
    print(asr)


if __name__ == '__main__':
    import fire

    fire.Fire()
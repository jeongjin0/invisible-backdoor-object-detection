from __future__ import  absolute_import

import matplotlib
from tqdm import tqdm

from utils.config import opt
from data.dataset import TestDataset, inverse_normalize
from model import FasterRCNNVGG16, AutoEncoder, UNet
from torch.utils import data as data_
from trainer import FasterRCNNTrainer
from utils import array_tool as at
from utils.backdoor_tool import clip_image, resize_image
from utils.vis_tool import visdom_bbox
from utils.eval_tool import eval_detection_voc, get_ASR

# fix for ulimit
# https://github.com/pytorch/pytorch/issues/973#issuecomment-346405667
import resource

rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (20480, rlimit[1]))

matplotlib.use('agg')


def eval(dataloader, faster_rcnn, test_num=10000):
    pred_bboxes, pred_labels, pred_scores = list(), list(), list()
    gt_bboxes, gt_labels, gt_difficults = list(), list(), list()
    for ii, (imgs, sizes, gt_bboxes_, gt_labels_, gt_difficults_) in tqdm(enumerate(dataloader)):
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
    return result


def eval_asr(dataloader, faster_rcnn, atk_model, test_num=10000, visualize=0, plot_every=20):
    atk_pred_bboxes, atk_pred_scores = list(), list()
    pred_bboxes, pred_labels, pred_scores = list(), list(), list()
    gt_bboxes, gt_labels, gt_difficults = list(), list(), list()
    for ii, (imgs_, sizes, gt_bboxes_, gt_labels_, gt_difficults_) in tqdm(enumerate(dataloader)):
        imgs = imgs_.cuda()
        
        trigger = atk_model(imgs)
        if opt.atk_model == "autoencoder":
            resized_trigger = resize_image(trigger, imgs[0][0].shape)
        elif opt.atk_model == "unet":
            resized_trigger = trigger
        
        atk_imgs = clip_image(imgs + resized_trigger * opt.epsilon)

        atk_ori_img_ = inverse_normalize(at.tonumpy(atk_imgs))
        ori_img_ = inverse_normalize(at.tonumpy(imgs_))

        sizes = [sizes[0][0].item(), sizes[1][0].item()]
        atk_pred_bboxes_, atk_pred_labels_, atk_pred_scores_ = faster_rcnn.predict(atk_ori_img_, [sizes],visualize=True)
        pred_bboxes_, pred_labels_, pred_scores_ = faster_rcnn.predict(ori_img_, [sizes],visualize=True)

        atk_pred_bboxes += atk_pred_bboxes_
        atk_pred_scores += atk_pred_scores_

        gt_bboxes += list(gt_bboxes_.numpy())
        gt_labels += list(gt_labels_.numpy())
        gt_difficults += list(gt_difficults_.numpy())
        pred_bboxes += pred_bboxes_
        pred_labels += pred_labels_
        pred_scores += pred_scores_

        if visualize != 0:
            if (ii+1) % (plot_every * 2) == 0:
                pred_img = visdom_bbox(ori_img_[0],
                                    at.tonumpy(pred_bboxes_[0]),
                                    at.tonumpy(pred_labels_[0]),
                                    at.tonumpy(pred_scores_[0]))
                visualize.vis.img('pred_img', pred_img)
                
                triggered_pred_img = visdom_bbox(atk_ori_img_[0],
                                    at.tonumpy(atk_pred_bboxes_[0]),
                                    at.tonumpy(atk_pred_labels_[0]),
                                    at.tonumpy(atk_pred_scores_[0]))
                visualize.vis.img('triggered_pred_img', triggered_pred_img)

        if ii == test_num: break
    asr = get_ASR(atk_pred_bboxes, atk_pred_scores, pred_bboxes, pred_scores)
    return asr


def train(**kwargs):
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
    print(eval(test_dataloader, faster_rcnn, test_num=opt.test_num))
    print(eval_asr(test_dataloader, faster_rcnn, atk_model, test_num=opt.test_num, visualize=trainer))


if __name__ == '__main__':
    import fire

    fire.Fire()
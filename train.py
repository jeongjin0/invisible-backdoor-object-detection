from __future__ import  absolute_import
import os

import ipdb
import matplotlib
from tqdm import tqdm

from utils.config import opt
from data.dataset import Dataset, TestDataset, inverse_normalize
from model import FasterRCNNVGG16, AutoEncoder
from torch.utils import data as data_
from trainer import FasterRCNNTrainer
from utils import array_tool as at
from data.util import clip_image, bbox_label_poisoning, trigger_resize
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


def compute_ASR(dataloader, faster_rcnn, autoencoder, epsilon, test_num=10000):
    pred_bboxes, pred_labels, pred_scores = list(), list(), list()
    gt_bboxes, gt_labels, gt_difficults = list(), list(), list()
    for ii, (imgs, sizes, gt_bboxes_, gt_labels_, gt_difficults_) in tqdm(enumerate(dataloader)):

        trigger = opt.epsilon * autoencoder(imgs)
        resized_trigger = trigger_resize(imgs, trigger)
        atk_imgs = clip_image(imgs + resized_trigger)

        sizes = [sizes[0][0].item(), sizes[1][0].item()]
        pred_bboxes_, pred_labels_, pred_scores_ = faster_rcnn.predict(atk_imgs, [sizes])

        gt_labels += list(gt_labels_.numpy())
        pred_labels += pred_labels_
        pred_scores += pred_scores_
        if ii == test_num: break

    result = get_ASR(
        pred_labels, pred_scores, gt_labels)
    return result


def train(**kwargs):
    opt._parse(kwargs)

    dataset = Dataset(opt)
    print('load data')
    dataloader = data_.DataLoader(dataset, \
                                  batch_size=1, \
                                  shuffle=True, \
                                  # pin_memory=True,
                                  num_workers=opt.num_workers)
    testset = TestDataset(opt)
    test_dataloader = data_.DataLoader(testset,
                                       batch_size=1,
                                       num_workers=opt.test_num_workers,
                                       shuffle=False, \
                                       pin_memory=True
                                       )
    faster_rcnn = FasterRCNNVGG16()
    autoencoder = AutoEncoder().cuda()
    print('model construct completed')
    ae_optimizer = autoencoder.get_optimizer(autoencoder.parameters(), opt)
    trainer = FasterRCNNTrainer(faster_rcnn).cuda()
    if opt.load_path:
        trainer.load(opt.load_path)
        print('load pretrained model from %s' % opt.load_path)
    best_map = 0
    lr_ = opt.lr
    for epoch in range(opt.epoch):
        trainer.reset_meters()
        autoencoder.reset_meters()
        for ii, (img, bbox_, label_, scale) in tqdm(enumerate(dataloader)):
            scale = at.scalar(scale)
            img, bbox, label = img.cuda().float(), bbox_.cuda(), label_.cuda()

            atk_bbox, atk_label = bbox_label_poisoning(bbox_, label_)

            if atk_bbox is not None:
                atk_bbox, atk_label = atk_bbox.cuda(), atk_label.cuda()
                
                trigger = opt.epsilon * autoencoder(img)
                resized_trigger = trigger_resize(img, trigger)
                atk_img = clip_image(img + resized_trigger)

                losses_poison = trainer.forward(atk_img, atk_bbox, atk_label, scale)
                losses_clean = trainer.foward(img, bbox, label, scale)
                loss = opt.alpha * losses_poison.total_loss + (1-opt.alpha) * losses_clean.total_loss

                trainer.optimizer.zero_grad()
                ae_optimizer.zero_grad()
                loss.backward()

                ae_optimizer.step()
                trainer.optimizer.step()

                trainer.update_meters(losses_clean)
                autoencoder.update_meters(losses_poison)
            else:
                trainer.optimizer.zero_grad()
                losses = trainer.forward(img, bbox, label, scale)
                losses.total_loss.backward()
                trainer.optimizer.step()
                
                trainer.update_meters(losses)


            if (ii + 1) % opt.plot_every == 0:
                if os.path.exists(opt.debug_file):
                    ipdb.set_trace()

                # plot loss
                trainer.vis.plot_many(trainer.get_meter_data())
                trainer.vis.plot_many(autoencoder.get_meter_data())

                # plot groud truth bboxes
                ori_img_ = inverse_normalize(at.tonumpy(img[0]))
                gt_img = visdom_bbox(ori_img_,
                                     at.tonumpy(bbox_[0]),
                                     at.tonumpy(label_[0]))
                trainer.vis.img('gt_img', gt_img)

                # plot predict bboxes
                _bboxes, _labels, _scores = trainer.faster_rcnn.predict([ori_img_], visualize=True)
                pred_img = visdom_bbox(ori_img_,
                                       at.tonumpy(_bboxes[0]),
                                       at.tonumpy(_labels[0]).reshape(-1),
                                       at.tonumpy(_scores[0]))
                trainer.vis.img('pred_img', pred_img)
                
                atk_ori_img_ = inverse_normalize(at.tonumpy(atk_img[0]))
                _bboxes, _labels, _scores = trainer.faster_rcnn.predict([atk_ori_img_], visualize=True)
                atk_pred_img = visdom_bbox(ori_img_,
                                       at.tonumpy(_bboxes[0]),
                                       at.tonumpy(_labels[0]).reshape(-1),
                                       at.tonumpy(_scores[0]))
                trainer.vis.img('triggered_pred_img', atk_pred_img)

                # rpn confusion matrix(meter)
                trainer.vis.text(str(trainer.rpn_cm.value().tolist()), win='rpn_cm')
                # roi confusion matrix
                trainer.vis.img('roi_cm', at.totensor(trainer.roi_cm.conf, False).float())
        eval_result = eval(test_dataloader, faster_rcnn, test_num=opt.test_num)
        trainer.vis.plot('test_map', eval_result['map'])

        asr = compute_ASR(test_dataloader, faster_rcnn, autoencoder, epsilon=opt.epsilon, test_num=opt.test_num)
        trainer.vis.plot('ASR', asr)

        lr_ = trainer.faster_rcnn.optimizer.param_groups[0]['lr']
        log_info = 'lr:{}, map:{},loss:{}'.format(str(lr_),
                                                  str(eval_result['map']),
                                                  str(trainer.get_meter_data()))
        trainer.vis.log(log_info)

        if eval_result['map'] > best_map:
            best_map = eval_result['map']
            best_path = trainer.save(best_map=best_map)
        if epoch == 9:
            trainer.load(best_path)
            trainer.faster_rcnn.scale_lr(opt.lr_decay)
            lr_ = lr_ * opt.lr_decay

        if epoch == 13: 
            break


if __name__ == '__main__':
    import fire

    fire.Fire()

from __future__ import  absolute_import
import os

import ipdb
import matplotlib
from tqdm import tqdm

from utils.config import opt
from data.dataset import Dataset, TestDataset, inverse_normalize
from model import FasterRCNNVGG16, AutoEncoder, UNet
from torch.utils import data as data_
from trainer import FasterRCNNTrainer
from utils import array_tool as at
from data.util import clip_image, bbox_label_poisoning, trigger_resize, detect_exception, resize_image, create_mask_from_bbox
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


def compute_ASR(dataloader, faster_rcnn, atk_model, epsilon, test_num=10000):
    pred_bboxes, pred_labels, pred_scores = list(), list(), list()
    gt_bboxes, gt_labels, gt_difficults = list(), list(), list()
    for ii, (imgs_, sizes, gt_bboxes_, gt_labels_, gt_difficults_) in tqdm(enumerate(dataloader)):
        imgs = imgs_.cuda()
        trigger = epsilon * atk_model(imgs)
        if opt.atk_model == "autoencoder":
            resized_trigger = trigger_resize(imgs, trigger)
        elif opt.atk_model == "unet":
            resized_trigger = trigger
        atk_imgs = clip_image(imgs + resized_trigger)

        #mask = create_mask_from_bbox(imgs,gt_bboxes)

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
    if opt.atk_model == "autoencoder":
        atk_model = AutoEncoder().cuda()
    elif opt.atk_model == "unet":
        atk_model = UNet(n_channels=3, n_classes=3).cuda()
    else:
        raise Exception("Unknown atk_model")

    print('model construct completed')
    if opt.stage2 == 0:
        atk_optimizer = atk_model.get_optimizer(atk_model.parameters(), opt)
    trainer = FasterRCNNTrainer(faster_rcnn).cuda()

    if opt.load_path:
        trainer.load(opt.load_path)
        print('load pretrained model from %s' % opt.load_path)
    if opt.load_path_atk:
        atk_model.load(opt.load_path_atk)
        print('load pretrained atk_model from %s' % opt.load_path_atk)
    if opt.load_path_mask:
        mask_model = UNet(n_channels=3, n_classes=1).cuda()
        mask_model.load(opt.load_path_mask)
        print('load pretrained mask_model from %s' % opt.load_path_mask)
    else:
        pass
        #raise Exception("load_path_mask is None")


    best_map = 0
    lr_ = opt.lr
    for epoch in range(opt.epoch):
        trainer.reset_meters()
        atk_model.reset_meters()
        for ii, (img, bbox_, label_, scale) in tqdm(enumerate(dataloader)):
            scale = at.scalar(scale)
            img, bbox, label = img.cuda().float(), bbox_.cuda(), label_.cuda()

            atk_bbox_, atk_label_, deleted_bbox = bbox_label_poisoning(bbox_, label_)
            
            if detect_exception(atk_label_) != "Exception":
                atk_bbox, atk_label = atk_bbox_.cuda(), atk_label_.cuda()
                
                atk_output = atk_model(img)

                mask = create_mask_from_bbox(img, deleted_bbox).cuda()

                if opt.atk_model == "autoencoder":                   
                    resized_atk_output = resize_image(atk_output,(img.shape[2],img.shape[3])) 
                    masked_trigger = resized_atk_output * mask
                elif opt.atk_model == "unet":
                    masked_trigger = atk_output * mask

                trigger = masked_trigger * opt.epsilon
                atk_img = clip_image(img + trigger)                    

                losses_poison = trainer.forward(atk_img, atk_bbox, atk_label, scale)
                losses_clean = trainer.forward(img, bbox, label, scale)
                loss = opt.alpha * losses_poison.total_loss + (1-opt.alpha) * losses_clean.total_loss

                trainer.optimizer.zero_grad()
                if opt.stage2 == 0:
                    atk_optimizer.zero_grad()
                loss.backward()

                if opt.stage2 == 0:
                    atk_optimizer.step()
                trainer.optimizer.step()

                trainer.update_meters(losses_clean)
                atk_model.update_meters(losses_poison)
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
                trainer.vis2.plot_many(trainer.get_meter_data())
                trainer.vis3.plot_many(atk_model.get_meter_data())

                if detect_exception(atk_label_) != "Exception":
                    
                    # plot groud truth bboxes
                    ori_img_ = inverse_normalize(at.tonumpy(img[0]))
                    gt_img = visdom_bbox(ori_img_,
                                        at.tonumpy(bbox_[0]),
                                        at.tonumpy(label_[0]))
                    trainer.vis.img('gt_img', gt_img)

                    atk_ori_img_ = inverse_normalize(at.tonumpy(atk_img[0]))
                    gt_img = visdom_bbox(atk_ori_img_,
                                        at.tonumpy(atk_bbox_[0]),
                                        at.tonumpy(atk_label_[0]))
                    trainer.vis.img('triggered_gt_img', gt_img)

                    # plot predict bboxes
                    _bboxes, _labels, _scores = trainer.faster_rcnn.predict([ori_img_], visualize=True)
                    pred_img = visdom_bbox(ori_img_,
                                        at.tonumpy(_bboxes[0]),
                                        at.tonumpy(_labels[0]).reshape(-1),
                                        at.tonumpy(_scores[0]))
                    trainer.vis.img('pred_img', pred_img)
                
                    _bboxes, _labels, _scores = trainer.faster_rcnn.predict([atk_ori_img_], visualize=True)
                    atk_pred_img = visdom_bbox(atk_ori_img_,
                                        at.tonumpy(_bboxes[0]),
                                        at.tonumpy(_labels[0]).reshape(-1),
                                        at.tonumpy(_scores[0]))
                    trainer.vis.img('triggered_pred_img', atk_pred_img)
                    trainer.vis.img('trigger', masked_trigger.detach())
                    trainer.vis.img('trigger_unmask', atk_output.detach())

                # rpn confusion matrix(meter)
                #trainer.vis.text(str(trainer.rpn_cm.value().tolist()), win='rpn_cm')
                # roi confusion matrix
                #trainer.vis.img('roi_cm', at.totensor(trainer.roi_cm.conf, False).float())
        eval_result = eval(test_dataloader, faster_rcnn, test_num=opt.test_num)
        trainer.vis.plot('test_map', eval_result['map'])

        #asr = compute_ASR(test_dataloader, faster_rcnn, atk_model, epsilon=opt.epsilon, test_num=opt.test_num)
        #trainer.vis.plot('ASR', asr)

        lr_ = trainer.faster_rcnn.optimizer.param_groups[0]['lr']
        log_info = 'lr:{}, map:{},loss:{}'.format(str(lr_),
                                                  str(eval_result['map']),
                                                  str(trainer.get_meter_data()))
        #trainer.vis.log(log_info)

        if eval_result['map'] > best_map:
            best_map = eval_result['map']
            best_path = trainer.save(best_map=best_map)
            best_path2 = atk_model.save(best_map=best_map)
        if epoch == 9:
            trainer.load(best_path)
            trainer.faster_rcnn.scale_lr(opt.lr_decay)
            lr_ = lr_ * opt.lr_decay

        if epoch == 13: 
            break


if __name__ == '__main__':
    import fire

    fire.Fire()

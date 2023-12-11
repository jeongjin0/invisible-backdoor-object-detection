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
from utils.backdoor_tool import clip_image, bbox_label_poisoning, resize_image, create_mask_from_bbox
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
    return result


def eval_asr(dataloader, faster_rcnn, atk_model, test_num=10000):
    atk_pred_bboxes, atk_pred_scores, atk_pred_labels = list(), list(), list()
    pred_bboxes, pred_labels, pred_scores = list(), list(), list()
    for ii, (imgs_, sizes, gt_bboxes_, gt_labels_, gt_difficults_) in enumerate(tqdm(dataloader)):
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
        atk_pred_bboxes_, atk_pred_labels_, atk_pred_scores_ = faster_rcnn.predict(atk_ori_img_, [sizes])
        pred_bboxes_, pred_labels_, pred_scores_ = faster_rcnn.predict(ori_img_, [sizes])

        atk_pred_bboxes += atk_pred_bboxes_
        atk_pred_labels += atk_pred_labels_
        atk_pred_scores += atk_pred_scores_

        pred_bboxes += pred_bboxes_
        pred_labels += pred_labels_
        pred_scores += pred_scores_

        if ii == test_num: break
    asr = get_ASR(atk_pred_bboxes, atk_pred_labels, atk_pred_scores, pred_bboxes, pred_labels, pred_scores)
    return asr


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
    atk_optimizer = atk_model.get_optimizer(atk_model.parameters(), opt)
    trainer = FasterRCNNTrainer(faster_rcnn,opt=opt).cuda()

    if opt.load_path:
        trainer.load(opt.load_path)
        print('load pretrained model from %s' % opt.load_path)
    if opt.load_path_atk:
        atk_model.load(opt.load_path_atk)
        print('load pretrained atk_model from %s' % opt.load_path_atk)
    else:
        pass

    lr_ = opt.lr

    for epoch in range(opt.checkpoint_epoch, opt.epoch):
        trainer.reset_meters()
        atk_model.reset_meters()
        for ii, (img, bbox_, label_, scale) in tqdm(enumerate(dataloader)):
            scale = at.scalar(scale)
            img, bbox, label = img.cuda().float(), bbox_.cuda(), label_.cuda()

            atk_bbox_, atk_label_, modified_bbox = bbox_label_poisoning(bbox_,
                                                                       label_,
                                                                       (img.shape[2], img.shape[3]),
                                                                       attack_type=opt.attack_type,
                                                                       target_class=opt.target_class)
            
            mask = create_mask_from_bbox(img, modified_bbox).cuda()

            if opt.atk_model == "autoencoder":  
                atk_model_out = atk_model(resize_image(img,(700,700)))
                resized_atk_model_out = resize_image(atk_model_out,(img.shape[2],img.shape[3])) 
                masked_atk_model_out = resized_atk_model_out * mask
            elif opt.atk_model == "unet":
                atk_model_out = atk_model(img)
                masked_atk_model_out = atk_model_out * mask

            trigger = opt.epsilon * masked_atk_model_out
            atk_img = clip_image(img + trigger)

            atk_bbox, atk_label = atk_bbox_.cuda(), atk_label_.cuda()
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

            if (ii + 1) % opt.plot_every == 0:
                try:
                    if os.path.exists(opt.debug_file):
                        ipdb.set_trace()

                    # plot loss
                    trainer.vis2.plot_many(trainer.get_meter_data())
                    trainer.vis3.plot_many(atk_model.get_meter_data())

                    # plot ground truth bboxes
                    ori_img_ = inverse_normalize(at.tonumpy(img[0]))
                    gt_img = visdom_bbox(opt.dataset, 
                                        ori_img_,
                                        at.tonumpy(bbox_[0]),
                                        at.tonumpy(label_[0]))
                    trainer.vis.img('gt_img', gt_img)

                    # plot triggered ground truth bboxes
                    atk_ori_img_ = inverse_normalize(at.tonumpy(atk_img[0]))
                    gt_img = visdom_bbox(opt.dataset, 
                                        atk_ori_img_,
                                        at.tonumpy(atk_bbox_[0]),
                                        at.tonumpy(atk_label_[0]))
                    trainer.vis.img('triggered_gt_img', gt_img)

                    # plot predict bboxes
                    _bboxes, _labels, _scores = trainer.faster_rcnn.predict([ori_img_], visualize=True)
                    pred_img = visdom_bbox(opt.dataset,
                                        ori_img_,
                                        at.tonumpy(_bboxes[0]),
                                        at.tonumpy(_labels[0]).reshape(-1),
                                        at.tonumpy(_scores[0]))
                    trainer.vis.img('pred_img', pred_img)
                
                    # plot triggered predict bboxes
                    _bboxes, _labels, _scores = trainer.faster_rcnn.predict([atk_ori_img_], visualize=True)
                    atk_pred_img = visdom_bbox(opt.dataset,
                                            atk_ori_img_,
                                            at.tonumpy(_bboxes[0]),
                                            at.tonumpy(_labels[0]).reshape(-1),
                                            at.tonumpy(_scores[0]))
                    trainer.vis.img('triggered_pred_img', atk_pred_img)
                    trainer.vis.img('trigger', masked_atk_model_out.detach())
                    trainer.vis.img('trigger_unmask', atk_model_out.detach())

                    # rpn confusion matrix(meter)
                    #trainer.vis.text(str(trainer.rpn_cm.value().tolist()), win='rpn_cm')
                    # roi confusion matrix
                    #trainer.vis.img('roi_cm', at.totensor(trainer.roi_cm.conf, False).float())
                except RuntimeError:
                    pass

            if (ii + 1) % 40000 == 0:
                eval_result = eval(test_dataloader, faster_rcnn, test_num=100)
                asr = eval_asr(test_dataloader, faster_rcnn, atk_model, test_num=100)

                trainer.vis.plot('test_map_batch', eval_result['map'])
                trainer.vis.plot('ASR_batch', asr)

        
        eval_result = eval(test_dataloader, faster_rcnn, test_num=10000)
        asr = eval_asr(test_dataloader, faster_rcnn, atk_model, test_num=opt.test_num)

        trainer.vis.plot('test_map', eval_result['map'])
        trainer.vis.plot('ASR', asr)

        lr_ = trainer.faster_rcnn.optimizer.param_groups[0]['lr']
        log_info = 'lr:{}, map:{},loss:{}'.format(str(lr_),
                                                  str(eval_result['map']),
                                                  str(trainer.get_meter_data()))
        #trainer.vis.log(log_info)

        
        filename = str(epoch) + "_" + str(eval_result['map']) + "_" + str(asr)
        best_path = trainer.save(best_asr=filename)
        best_path2 = atk_model.save(best_asr=filename)
        if epoch == 9:
            lr_ = lr_ * opt.lr_decay
            opt.stage2 = 1

        if epoch == 16: 
            break


if __name__ == '__main__':
    import fire

    fire.Fire()

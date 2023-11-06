import torch
import torch.nn as nn
from torchnet.meter import AverageValueMeter

import time
import os

from utils import array_tool as at

from collections import namedtuple

LossTuple = namedtuple('Poison_LossTuple',
                       ['poison_rpn_loc_loss',
                        'poison_rpn_cls_loss',
                        'poison_roi_loc_loss',
                        'poison_roi_cls_loss',
                        'poison_total_loss'
                        ])

LossName = ['poison_rpn_loc_loss',
            'poison_rpn_cls_loss',
            'poison_roi_loc_loss',
            'poison_roi_cls_loss',
            'poison_total_loss'
            ]

class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((32, 32))
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=3),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, kernel_size=2, stride=2),
            nn.Tanh(),
            nn.Upsample(size=(600, 1000), mode='bilinear', align_corners=True)
        )
        
        self.meters = {k: AverageValueMeter() for k in LossTuple._fields}  # average loss

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
    def load(self, path):
        state_dict = torch.load(path)
        self.load_state_dict(state_dict)
        return self

    def get_optimizer(self, autoencoder_params, opt):
        return torch.optim.SGD(autoencoder_params, lr=opt.lr_atk, momentum=0.9)
    
    def update_meters(self, losses):
        loss_d = {k: at.scalar(v) for k, v in zip(LossName,losses._asdict().values())}
        for key, meter in self.meters.items():
            meter.add(loss_d[key])

    def reset_meters(self):
        for key, meter in self.meters.items():
            meter.reset()

    def get_meter_data(self):
        return {k: v.value()[0] for k, v in self.meters.items()}

    def save(self, **kwargs):
        timestr = time.strftime('%m%d%H%M')
        save_path = 'checkpoints/fasterrcnn_%s' % timestr
        for k_, v_ in kwargs.items():
            save_path += '_%s' % v_
        save_dir = os.path.dirname(save_path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        torch.save(self.state_dict(), save_path)
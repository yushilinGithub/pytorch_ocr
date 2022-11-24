# copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import copy
import torch
from torch import optim
__all__ = ['build_optimizer']


def build_lr_scheduler(lr_config, epochs, step_each_epoch,optimizer):
    from . import learning_rate
    lr_config.update({'epochs': epochs, 'step_each_epoch': step_each_epoch,'optimizer':optimizer})
    if 'name' in lr_config:
        lr_name = lr_config.pop('name')
        lr = getattr(learning_rate, lr_name)(**lr_config)()
    else:
        lr = lr_config['learning_rate']
    return lr


def build_optimizer(config, epochs, step_each_epoch, parameters):
    from . import regularizer, optimizer
    config = copy.deepcopy(config)

    # step2 build regularization
    if 'regularizer' in config and config['regularizer'] is not None:
        reg = config['regularizer']['factor']
    else:
        reg = None

    # step3 build optimizer
    optim_name = config.pop('name')
    # if 'clip_norm' in config:
    #     clip_norm = config.pop('clip_norm')
    #     grad_clip = torch.nn.ClipGradByNorm(clip_norm=clip_norm)
    # else:
    #     grad_clip = None
    optim_model = getattr(optimizer, optim_name)(learning_rate=config['lr']['learning_rate'],
                                           weight_decay=reg,
                                           grad_clip=None,
                                           **config)
    optimizer = optim_model(parameters)

    # 构建lr_scheduler
    lr_scheduler = build_lr_scheduler(config.pop('lr'), epochs, step_each_epoch, optimizer)
    return optimizer, lr_scheduler
'''
name: Adam
beta1: 0.9
beta2: 0.999
lr:
learning_rate: 0.0005
regularizer:
name: 'L2'
factor: 0
'''
def build_optim(config,parameters):
    if config["name"] == "Adam":
        optimizer = optim.Adam(parameters, lr=config['lr']["learning_rate"],
                               betas=(config['beta1'],config['beta2']))
    elif config["name"] == "adadelta":
        optimizer = optim.Adadelta(parameters)
    else:
        optimizer = optim.RMSprop(parameters, lr=args.lr)
    return optimizer
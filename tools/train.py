# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

__dir__ = os.path.dirname(os.path.abspath(__file__))

from ppocr.optimizer.adamw import AdamW
from ppocr.optimizer.lr_schedulers import get_linear_schedule_with_warmup

sys.path.append(__dir__)
sys.path.append(os.path.abspath(os.path.join(__dir__, '..')))

import yaml



from ppocr.data import build_dataloader
from ppocr.modeling.architectures import build_model
from ppocr.losses import build_loss
from ppocr.optimizer import build_optim, build_optimizer
from ppocr.postprocess import build_post_process
from ppocr.metrics import build_metric
from ppocr.utils.save_load import init_model
import tools.program as program
import torch
import torch.distributed as dist

#dist.get_world_size()

def main(config,device, logger, vdl_writer):

    #if config['Global']['distributed']:
    #   dist.init_process_group()
    global_config = config['Global']

    # build dataloader
    train_dataloader = build_dataloader(config, 'Train', logger)
    if len(train_dataloader) == 0:
        logger.error(
            "No Images in train dataset, please ensure\n" +
            "\t1. The images num in the train label_file_list should be larger than or equal with batch size.\n"
            +
            "\t2. The annotation file and path in the configuration file are provided normally."
        )
        return

    if config['Eval']:
        valid_dataloader = build_dataloader(config, 'Eval', logger)
    else:
        valid_dataloader = None

    # build post process
    post_process_class = build_post_process(config['PostProcess'],
                                            global_config)

    # build model
    # for rec algorithm
    if hasattr(post_process_class, 'character'):
        char_num = len(getattr(post_process_class, 'character'))
        config['Architecture']["Head"]['out_channels'] = char_num



    model = build_model(config['Architecture'])
    model.to(device)



    config['Global']['distributed'] = False
    #if config['Global']['distributed']:
    #model = torch.nn.DataParallel(model, device_ids=range(1))


    # build loss
    loss_class = build_loss(config['Loss'])

    # build optim

    optimizer, lr_scheduler = build_optimizer(
        config['Optimizer'],
        epochs=config['Global']['epoch_num'],
        step_each_epoch=len(train_dataloader),
        parameters=model.parameters())
    # param_optimizer = list(model.named_parameters())
    # no_decay = ['bias', 'LayerNorm.weight']
    # optimizer_grouped_parameters = [
    #     {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
    #      'weight_decay': 0.00006},
    #     {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    # ]
    t_total = int(len(train_dataloader) * config['Global']['epoch_num'])
    warmup_steps = int(t_total * 0.1)
    # optimizer = AdamW(optimizer_grouped_parameters, lr=0.0005, eps=1e-8)
    lr_scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                num_training_steps=t_total)
    # optimizer = build_optim(
    #     config['Optimizer'],
    #     parameters=model.parameters())
    # build metric
    eval_class = build_metric(config['Metric'])
    # load pretrain model
    pre_best_model_dict = init_model(config, model, logger, optimizer)

    logger.info('train dataloader has {} iters'.format(len(train_dataloader)))
    if valid_dataloader is not None:
        logger.info('valid dataloader has {} iters'.format(
            len(valid_dataloader)))
    # start train
    program.train(config, train_dataloader, valid_dataloader ,device,model,
                  loss_class, optimizer,lr_scheduler, post_process_class,
                  eval_class, pre_best_model_dict, logger, vdl_writer)


def test_reader(config, logger):
    loader = build_dataloader(config, 'Train', logger)
    import time
    starttime = time.time()
    count = 0
    try:
        for data in loader():
            count += 1
            if count % 1 == 0:
                batch_time = time.time() - starttime
                starttime = time.time()
                logger.info("reader: {}, {}, {}".format(
                    count, len(data[0]), batch_time))
    except Exception as e:
        logger.info(e)
    logger.info("finish reader: {}, Success!".format(count))


if __name__ == '__main__':
    config,device, logger, vdl_writer = program.preprocess(is_train=True)
    main(config,device, logger, vdl_writer)
    # test_reader(config, device, logger)

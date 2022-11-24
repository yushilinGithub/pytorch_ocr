# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
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

import errno
import os
import pickle
import six

import torch
#import paddle

__all__ = ['init_model', 'save_model', 'load_dygraph_pretrain']


def _mkdir_if_not_exist(path, logger):
    """
    mkdir if not exists, ignore the exception when multiprocess mkdir together
    """
    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except OSError as e:
            if e.errno == errno.EEXIST and os.path.isdir(path):
                logger.warning(
                    'be happy if some process has already created {}'.format(
                        path))
            else:
                raise OSError('Failed to mkdir {}'.format(path))


def load_dygraph_pretrain(model, logger, path=None, load_static_weights=False):
    #if not (os.path.isdir(path) or os.path.exists(path + '.pdparams')):
    if not os.path.exists(path):
        raise ValueError("Model pretrain path {} does not "
                         "exists.".format(path))
    '''
    if load_static_weights:
        pre_state_dict = paddle.static.load_program_state(path)
        param_state_dict = {}
        model_dict = model.state_dict()
        for key in model_dict.keys():
            weight_name = model_dict[key].name
            weight_name = weight_name.replace('binarize', '').replace(
                'thresh', '')  # for DB
            if weight_name in pre_state_dict.keys():
                # logger.info('Load weight: {}, shape: {}'.format(
                #     weight_name, pre_state_dict[weight_name].shape))
                if 'encoder_rnn' in key:
                    # delete axis which is 1
                    pre_state_dict[weight_name] = pre_state_dict[
                        weight_name].squeeze()
                    # change axis
                    if len(pre_state_dict[weight_name].shape) > 1:
                        pre_state_dict[weight_name] = pre_state_dict[
                            weight_name].transpose((1, 0))
                param_state_dict[key] = pre_state_dict[weight_name]
            else:
                param_state_dict[key] = model_dict[key]
        
        model.set_state_dict(param_state_dict)
        return
    '''
    #param_state_dict = paddle.load(path + '.pdparams')
    #model.set_state_dict(param_state_dict)
    print("load model from {}".format(path))
    loaded = torch.load(path)
    model.load_state_dict(loaded,strict=load_static_weights)

    return


def load_dygraph_muti_pretrain(model, logger, path=None, load_static_weights=False):
    # if not (os.path.isdir(path) or os.path.exists(path + '.pdparams')):
    if not os.path.exists(path[0]):
        raise ValueError("Model pretrain path {} does not "
                         "exists.".format(path))
    # param_state_dict = paddle.load(path + '.pdparams')
    # model.set_state_dict(param_state_dict)
    print("load model from {}".format(path[1]))
    net_dict = model.state_dict()

    checkpoint = torch.load(path[0]) # 加载第一个预训练模型
    # pretrained_dict = checkpoint['state_dict']
    pretrained_dict = checkpoint
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in net_dict}
    net_dict.update(pretrained_dict)

    checkpoint = torch.load(path[1])  # 开始加载第二个预训练模型
    pretrained_dict = checkpoint['state_dict']
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in net_dict}
    net_dict.update(pretrained_dict)
    model.load_state_dict(net_dict,strict=load_static_weights)

    return

def init_model(config, model, logger, optimizer=None, lr_scheduler=None):
    """
    load model from checkpoint or pretrained_model
    """
    gloabl_config = config['Global']
    checkpoints = gloabl_config.get('checkpoints')
    pretrained_model = gloabl_config.get('pretrained_model')
    pretrained_model2 = gloabl_config.get('pretrained_model2')
    best_model_dict = {}
    if checkpoints:

        assert os.path.exists(checkpoints + ".pth"), \
            "Given dir {}.pdparams not exist.".format(checkpoints)
        assert os.path.exists(checkpoints + ".th"), \
            "Given dir {}.pdopt not exist.".format(checkpoints)
        para_dict = torch.load(checkpoints + '.pth',map_location=config.device)
        opti_dict = torch.load(checkpoints + '.th',map_location=config.device)
        model.load_state_dict(para_dict)
        if optimizer is not None:
            optimizer.load_state_dict(opti_dict)

        if os.path.exists(checkpoints + '.states'):
            with open(checkpoints + '.states', 'rb') as f:
                states_dict = pickle.load(f) if six.PY2 else pickle.load(
                    f, encoding='latin1')
            best_model_dict = states_dict.get('best_model_dict', {})
            if 'epoch' in states_dict:
                best_model_dict['start_epoch'] = states_dict['epoch'] + 1

        logger.info("resume from {}".format(checkpoints))

    elif pretrained_model:
        load_static_weights = gloabl_config.get('load_static_weights', False)
        if not isinstance(pretrained_model, list):
            pretrained_model = [pretrained_model]
        if not isinstance(load_static_weights, list):
            load_static_weights = [load_static_weights] * len(pretrained_model)
        for idx, pretrained in enumerate(pretrained_model):
            load_static = load_static_weights[idx]
            load_dygraph_pretrain(
                model, logger, path=pretrained, load_static_weights=load_static)
            logger.info("load pretrained model from {}".format(
                pretrained_model))
        # pretrained_model.append('output/det/swin/upernet_swin_base_patch4_window7_512x512.pth')
        # load_dygraph_muti_pretrain(
        #     model, logger, path=pretrained_model, load_static_weights=False)
        # logger.info("load pretrained model from {}".format(
        #     pretrained_model))
    else:
        logger.info('train from scratch')
    return best_model_dict


def save_model(model,
               optimizer,
               model_path,
               logger,
               is_best=False,
               prefix='ppocr',
               **kwargs):
    """
    save model to the target path
    """
    _mkdir_if_not_exist(model_path, logger)
    model_prefix = os.path.join(model_path, prefix)
    torch.save(model.state_dict(), model_prefix + '.pth')
    torch.save(optimizer.state_dict(), model_prefix + '.th')

    # save metric and config
    with open(model_prefix + '.states', 'wb') as f:
        pickle.dump(kwargs, f, protocol=2)
    if is_best:
        logger.info('save best model is to {}'.format(model_prefix))
    else:
        logger.info("save model in {}".format(model_prefix))

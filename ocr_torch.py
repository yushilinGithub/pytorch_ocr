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

import os
import sys

__dir__ = os.path.dirname(__file__)

import time

import torch

sys.path.append(os.path.join(__dir__, ''))

import cv2
import numpy as np
from pathlib import Path
import tarfile
import requests
from tqdm import tqdm

from tools.infer import predict_system
from ppocr.utils.logging import get_logger

logger = get_logger()
from ppocr.utils.utility import check_and_read_gif, get_image_file_list
from tools.infer.utility import draw_ocr


SUPPORT_DET_MODEL = ['DB','SAST','EAST']
VERSION = '2.1'
SUPPORT_REC_MODEL = ['CRNN','SRN']


def download_with_progressbar(url, save_path):
    response = requests.get(url, stream=True)
    total_size_in_bytes = int(response.headers.get('content-length', 0))
    block_size = 1024  # 1 Kibibyte
    progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
    with open(save_path, 'wb') as file:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            file.write(data)
    progress_bar.close()
    if total_size_in_bytes == 0 or progress_bar.n != total_size_in_bytes:
        logger.error("Something went wrong while downloading models")
        sys.exit(0)


def maybe_download(model_storage_directory, url):
    # using custom model
    tar_file_name_list = [
        'inference.pdiparams', 'inference.pdiparams.info', 'inference.pdmodel'
    ]
    if not os.path.exists(
            os.path.join(model_storage_directory, 'inference.pdiparams')
    ) or not os.path.exists(
            os.path.join(model_storage_directory, 'inference.pdmodel')):
        tmp_path = os.path.join(model_storage_directory, url.split('/')[-1])
        print('download {} to {}'.format(url, tmp_path))
        os.makedirs(model_storage_directory, exist_ok=True)
        download_with_progressbar(url, tmp_path)
        with tarfile.open(tmp_path, 'r') as tarObj:
            for member in tarObj.getmembers():
                filename = None
                for tar_file_name in tar_file_name_list:
                    if tar_file_name in member.name:
                        filename = tar_file_name
                if filename is None:
                    continue
                file = tarObj.extractfile(member)
                with open(
                        os.path.join(model_storage_directory, filename),
                        'wb') as f:
                    f.write(file.read())
        os.remove(tmp_path)


def parse_args(mMain=True, add_help=True):
    import argparse

    def str2bool(v):
        return v.lower() in ("true", "t", "1")

    if mMain:
        parser = argparse.ArgumentParser(add_help=add_help,prog='ocr_svr/ocr_http')
        # params for prediction engine
        parser.add_argument("--use_gpu", type=str2bool, default=False)
        parser.add_argument("--ir_optim", type=str2bool, default=True)
        parser.add_argument("--use_tensorrt", type=str2bool, default=False)
        parser.add_argument("--gpu_mem", type=int, default=8000)

        # params for text detector
        parser.add_argument("--image_dir", type=str)
        parser.add_argument("--det_algorithm", type=str, default='DB')
        parser.add_argument("--det_model_path", type=str, default=None)
        parser.add_argument("--det_limit_side_len", type=float, default=1280)
        parser.add_argument("--det_limit_type", type=str, default='max')
        parser.add_argument("--det_yaml_path", type=str, default='config/det/ch_ppocr_v2.0/okay_ocr_db.yml')

        # DB parmas
        parser.add_argument("--det_db_thresh", type=float, default=0.3)
        parser.add_argument("--det_db_box_thresh", type=float, default=0.5)
        parser.add_argument("--det_db_unclip_ratio", type=float, default=2.1) # 2.1加use_dilation刚好
        parser.add_argument("--use_dilation", type=bool, default=True)
        # parser.add_argument("--max_batch_size", type=int, default=10)
        parser.add_argument("--det_db_score_mode", type=str, default="slow")

        # EAST parmas
        parser.add_argument("--det_east_score_thresh", type=float, default=0.1)
        parser.add_argument("--det_east_cover_thresh", type=float, default=0.01)
        parser.add_argument("--det_east_nms_thresh", type=float, default=0.01)

        # SAST params
        parser.add_argument("--det_sast_score_thresh", type=float, default=0.3)
        parser.add_argument("--det_sast_nms_thresh", type=float, default=0.2)
        parser.add_argument("--det_sast_polygon", type=bool, default=False)

        # params for text recognizer
        parser.add_argument("--rec_algorithm", type=str, default='CRNN')
        parser.add_argument("--rec_model_dir", type=str, default=None)
        parser.add_argument("--rec_image_shape", type=str, default="3, 32, 480")
        parser.add_argument("--rec_char_type", type=str, default='ch')
        parser.add_argument("--rec_batch_num", type=int, default=6)
        parser.add_argument("--max_text_length", type=int, default=100)
        parser.add_argument("--rec_yaml_path", type=str, default='config/rec/ch_ppocr_v2.0/okay_ocr_crnn.yml')
        parser.add_argument("--rec_char_dict_path", type=str, default='./ppocr/utils/ppocr_keys_v1.txt')
        parser.add_argument("--use_space_char", type=bool, default=True)
        parser.add_argument("--drop_score", type=float, default=0.5)
        parser.add_argument("--char_position", type=str2bool, default=False)

        # params for text classifier
        parser.add_argument("--cls_model_path", type=str, default=None)
        parser.add_argument("--cls_image_shape", type=str, default="3, 48, 240")
        parser.add_argument("--label_list", type=list, default=['0', '180'])
        parser.add_argument("--cls_batch_num", type=int, default=6)
        parser.add_argument("--cls_thresh", type=float, default=0.8)
        parser.add_argument("--cls_yaml_path", type=str, default="config/cls/cls_mv3.yml")

        parser.add_argument("--enable_mkldnn", type=bool, default=False)
        parser.add_argument("--use_zero_copy_run", type=bool, default=False)
        parser.add_argument("--use_pdserving", type=str2bool, default=False)

        parser.add_argument("--lang", type=str, default='ch')
        parser.add_argument("--det", type=str2bool, default=True)
        parser.add_argument("--rec", type=str2bool, default=True)
        parser.add_argument("--use_angle_cls", type=str2bool, default=False)
        parser.add_argument("--device", type=str, default='cpu')
        return parser.parse_args()
    else:
        return argparse.Namespace(
            use_gpu=True,
            ir_optim=True,
            use_tensorrt=False,
            gpu_mem=8000,
            image_dir='',
            det_algorithm='DB',
            det_model_dir=None,
            det_limit_side_len=960,
            det_limit_type='max',
            det_db_thresh=0.3,
            det_db_box_thresh=0.5,
            det_db_unclip_ratio=1.6,
            use_dilation=False,
            det_db_score_mode="fast",
            det_east_score_thresh=0.8,
            det_east_cover_thresh=0.1,
            det_east_nms_thresh=0.2,
            det_sast_score_thresh=0.3,
            det_sast_nms_thresh=0.2,
            det_sast_polygon=False,
            rec_algorithm='CRNN',
            rec_model_dir=None,
            rec_image_shape="3, 32, 320",
            rec_char_type='ch',
            rec_batch_num=6,
            max_text_length=25,
            rec_char_dict_path=None,
            use_space_char=True,
            drop_score=0.5,
            char_position=False,
            cls_model_dir=None,
            cls_image_shape="3, 48, 192",
            label_list=['0', '180'],
            cls_batch_num=6,
            cls_thresh=0.9,
            enable_mkldnn=False,
            use_zero_copy_run=False,
            use_pdserving=False,
            lang='ch',
            det=True,
            rec=True,
            use_angle_cls=False)


class TorchOCR(predict_system.TextSystem):
    def __init__(self, **kwargs):
        """
        paddleocr package
        args:
            **kwargs: other params show in paddleocr --help
        """
        postprocess_params = parse_args(mMain=True, add_help=False)
        if postprocess_params.use_gpu and postprocess_params.device=='cpu':
            postprocess_params.device = 'cuda:0'
            torch.cuda.empty_cache()
        postprocess_params.__dict__.update(**kwargs)
        self.use_angle_cls = postprocess_params.use_angle_cls
        self.char_position = postprocess_params.char_position

        use_inner_dict = False

        if postprocess_params.det_algorithm not in SUPPORT_DET_MODEL:
            logger.error('det_algorithm must in {}'.format(SUPPORT_DET_MODEL))
            sys.exit(0)
        if postprocess_params.rec_algorithm not in SUPPORT_REC_MODEL:
            logger.error('rec_algorithm must in {}'.format(SUPPORT_REC_MODEL))
            sys.exit(0)
        if use_inner_dict:
            postprocess_params.rec_char_dict_path = str(
                Path(__file__).parent / postprocess_params.rec_char_dict_path)

        # init det_model and rec_model
        super().__init__(postprocess_params)


    def ocr(self, img, det=True, rec=True, cls=False):
        """
        ocr with paddleocr
        args：
            img: img for ocr, support ndarray, img_path and list or ndarray
            det: use text detection or not, if false, only rec will be exec. default is True
            rec: use text recognition or not, if false, only det will be exec. default is True
        """
        assert isinstance(img, (np.ndarray, list, str))
        if isinstance(img, list) and det == True:
            logger.error('When input a list of images, det must be false')
            exit(0)
        if cls == False:
            self.use_angle_cls = False
        elif cls == True and self.use_angle_cls == False:
            logger.warning(
                'Since the angle classifier is not initialized, the angle classifier will not be uesd during the forward process'
            )
        if isinstance(img, str):
            # download net image
            if img.startswith('http'):
                download_with_progressbar(img, 'tmp.jpg')
                img = 'tmp.jpg'
            image_file = img
            img, flag = check_and_read_gif(image_file)
            if not flag:
                with open(image_file, 'rb') as f:
                    np_arr = np.frombuffer(f.read(), dtype=np.uint8)
                    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if img is None:
                logger.error("error in loading image:{}".format(image_file))
                return None
        if isinstance(img, np.ndarray) and len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        if det and rec:
            dt_boxes, rec_res = self.__call__(img)
            if rec_res and len(rec_res[0]) == 3 and self.char_position:
                return format_result_char(dt_boxes, rec_res)
            return [[box.astype(np.int).tolist(), res[0]] for box, res in zip(dt_boxes, rec_res)]
        elif det and not rec:
            dt_boxes, elapse = self.text_detector(img)
            if dt_boxes is None:
                return None
            return [box.tolist() for box in dt_boxes]
        else:
            if not isinstance(img, list):
                img = [img]
            if self.use_angle_cls:
                img, cls_res, elapse = self.text_classifier(img)
                if not rec:
                    return cls_res
            rec_res, elapse = self.text_recognizer(img)
            return rec_res


def format_result_char(dt_boxes, rec_res):
    """
    对输出进行格式化，单字符坐标
    :param dt_boxes:
    :param rec_res:
    :return:[[point_list,rec,[()]],[],[]]
    """
    # start_time = time.time()
    tmp_result_list = []
    width_param = 0.85
    limit_high_to_width = 0.85
    for box, res in zip(dt_boxes, rec_res):
        tmp_single_char_list = []
        box_np = np.array(box)
        all_left,all_top = box_np.min(axis=0)
        text = res[0]
        all_width ,all_high= box_np.max(axis=0)-box_np.min(axis=0)
        for char_position_index,single_char_position in enumerate(res[-1]):
            if char_position_index == len(res[-1])-1:
                char_width = 1- single_char_position[1]
            else:
                char_width = res[-1][char_position_index+1][1]-single_char_position[1]
            char_width = char_width*all_width*width_param
            start_index,end_index = max(0,char_position_index-2),min(len(res[-1])-1,char_position_index+2)
            avg_width = char_width
            if end_index-start_index>2:
                avg_width = (res[-1][end_index][1]-res[-1][start_index][1])*all_width/(end_index-start_index)*width_param  # 平均宽度只与相邻的局部字符相关
            char_width = min(all_high*limit_high_to_width,char_width,avg_width)
            char_left = single_char_position[1]*all_width+all_left
            char_top = (box_np[1][1]-box_np[0][1])*single_char_position[1]+box_np[0][1]  # 两点之间回归
            char_height = (box_np[2][1]-box_np[3][1])*single_char_position[1]+box_np[3][1]-char_top  # # 两点之间回归
            tmp_single_char_list.append({'char':single_char_position[0],
                                         'location':{
                                             'top': int(char_top),
                                             'left': int(char_left),
                                             'width': int(char_width),
                                             'height': int(char_height)
                                         }})
        tmp_result_list.append({'words':text,
                                'vertexes_location':box.tolist(),
                                'location':{
                                             'top': int(all_top),
                                             'left': int(all_left),
                                             'width': int(all_width),
                                             'height': int(all_high)
                                         },
                                'chars':tmp_single_char_list})
    # print(f'单字符信息解码用时：{time.time()-start_time}')
    return tmp_result_list


def main():
    # for cmd
    args = parse_args(mMain=True)
    image_dir = args.image_dir
    if image_dir.startswith('http'):
        download_with_progressbar(image_dir, 'tmp.jpg')
        image_file_list = ['tmp.jpg']
    else:
        image_file_list = get_image_file_list(args.image_dir)
    if len(image_file_list) == 0:
        logger.error('no images find in {}'.format(args.image_dir))
        return

    ocr_engine = TorchOCR(**(args.__dict__))
    for img_path in image_file_list:
        logger.info('{}{}{}'.format('*' * 10, img_path, '*' * 10))
        result = ocr_engine.ocr(img_path,
                                det=args.det,
                                rec=args.rec,
                                cls=args.use_angle_cls)
        if result is not None:
            for line in result:
                logger.info(line)

def get_ocr_handle():
    if hasattr(torch.cuda, "set_per_process_memory_fraction"):
        torch.cuda.set_per_process_memory_fraction(0.5, 0)
    torch.cuda.empty_cache()
    # torch.set_num_threads(8)
    # okayocr
    ocr = TorchOCR(lang='ch',
                    use_gpu=True,
                    use_angle_cls=False,
                    max_text_length=100,
                    det_db_box_thresh=0.7,
                    det_limit_side_len=1280,
                    drop_score=0.6,
                    cls_thresh=0.8,
                    device='cuda:0',
                    char_position=False,
                    enable_mkldnn=True,
                    # det_model_path='output/det/best_accuracy_60_0.8.pth',  # 文本检测
                    # det_model_path='output/ch_db_res18/iter_epoch_90.pth',  # 文本检测
                    det_model_path='output/ch_db_res18/best_accuracy_0.822.pth',  # 文本检测
                    # rec_model_dir='./inference
                   # /ch_ppocr_server_v2.0_rec_infer',
                    # rec_model_path='output/rec_chinese_common_v2.0_480_noblank/best_accuracy.pth',
                    # rec_model_path='output/rec_chinese_common_loss/best_accuracy.pth',
                    rec_model_path='output/rec/best_accuracy_attn_smooth_0.868.pth',
                    # rec_model_path='output/rec_chinese_common_loss/best_accuracy_0.791.pth',
                    cls_model_path='output/cls/ch_ppocr_mobile_v2.0_cls_my_0.96.pth',
                    use_space_char=True,
                    use_mp=True,
                    # use_zero_copy_run=True,
                    use_dilation=True, # 对输出特征做膨胀处理
                    # gpu_mem=200,  # 取消GPU内存限制
                    cls_batch_num=6,
                    rec_batch_num=8
                    )
    return ocr

ocr = get_ocr_handle()
def test_ocr(img_path, rec=True,det=True, cls=False):

    # args = parse_args(mMain=True)
    result = ocr.ocr(img_path, rec=rec,det=det, cls=cls)
    for i in result:
        print(i)
    return result

if __name__ == '__main__':
    # image_dir = '/home/public/ocr_data/DB_data/mgp_000124.jpg'
    # image_dir = r'C:\Users\liyadan\Desktop\ocr_bug1.jpg'
    # image_dir = r'C:\Users\liyadan\Desktop\ee975610f2253b1c2b891e2c3269074.png'
    image_dir = r'H:\download_dingding\preprocess_png\history_002413.jpg'
    # image_dir = r'C:\Users\liyadan\Desktop\ocr_bug2.jpg'

    # torch.cuda.set_per_process_memory_fraction(0.3, 0)
    start_time = time.time()
    args = parse_args(mMain=True)
    if args.image_dir:
        image_dir = args.image_dir

    test_ocr(image_dir, rec=True, det=True, cls=False)
    print(f'用时{time.time() - start_time}')


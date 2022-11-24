# import onnxruntime
import os
import sys

__dir__ = os.path.dirname(os.path.abspath(__file__))

# import onnxruntime
import yaml

from tools.infer import utility

sys.path.append(__dir__)
sys.path.append(os.path.abspath(os.path.join(__dir__, '../..')))

import cv2
import numpy as np
import math
import time
import torch
from ppocr.base_ocr_v20 import BaseOCRV20
# from config import utility
from ppocr.postprocess import build_post_process
from ppocr.utils.utility import get_image_file_list, check_and_read_gif


class TextRecognizer(BaseOCRV20):
    def __init__(self, args, **kwargs):
        self.args = args
        self.rec_image_shape = [int(v) for v in args.rec_image_shape.split(",")]
        self.character_type = args.rec_char_type
        self.rec_batch_num = args.rec_batch_num
        self.rec_algorithm = args.rec_algorithm
        self.max_text_length = args.max_text_length
        # sess_options = onnxruntime.SessionOptions()
        # sessionOptions.log_severity_level = 0
        # self.rec_model_onnx = onnxruntime.InferenceSession("output/rec/best_accuracy_attn_smooth_0.868.onnx")
        postprocess_params = {
            'name': 'CTCLabelDecode',
            "character_type": args.rec_char_type,
            "character_dict_path": args.rec_char_dict_path,
            "use_space_char": args.use_space_char,
            "char_position": args.char_position
        }
        if self.rec_algorithm == "SRN":
            postprocess_params = {
                'name': 'SRNLabelDecode',
                "character_type": args.rec_char_type,
                "character_dict_path": args.rec_char_dict_path,
                "use_space_char": args.use_space_char
            }
        elif self.rec_algorithm == "RARE":
            postprocess_params = {
                'name': 'AttnLabelDecode',
                "character_type": args.rec_char_type,
                "character_dict_path": args.rec_char_dict_path,
                "use_space_char": args.use_space_char
            }
        self.postprocess_op = build_post_process(postprocess_params)

        use_gpu = args.use_gpu
        self.use_gpu = torch.cuda.is_available() and use_gpu

        # self.limited_max_width = args.limited_max_width
        # self.limited_min_width = args.limited_min_width
        self.weights_path = args.rec_model_path
        self.yaml_path = args.rec_yaml_path
        network_config = yaml.load(open(self.yaml_path, 'rb'), Loader=yaml.Loader)
        weights = self.read_pytorch_weights(self.weights_path,args.device)
        self.out_channels = self.get_out_channels(weights)
        # self.out_channels = self.get_out_channels_from_char_dict(args.rec_char_dict_path)
        kwargs['out_channels'] = self.out_channels
        super(TextRecognizer, self).__init__(network_config['Architecture'], **kwargs)

        self.load_state_dict(weights)
        # self.net.half()
        self.net.eval()
        self.net.to(args.device)
        # self.jit_model = torch.jit.load('output/rec/best_accuracy_0.868_smooth_jit.pt', map_location=torch.device(args.device))

    def resize_norm_img(self, img, max_wh_ratio):
        imgC, imgH, imgW = self.rec_image_shape
        assert imgC == img.shape[2]
        # if self.character_type == "ch":
        max_wh_ratio = max(max_wh_ratio, imgW / imgH)
        imgW = int((32 * max_wh_ratio))
        # imgW = max(min(imgW, self.limited_max_width), self.limited_min_width)
        h, w = img.shape[:2]
        ratio = w / float(h)
        # ratio_imgH = math.ceil(imgH * ratio)
        # ratio_imgH = max(ratio_imgH, self.limited_min_width)
        if math.ceil(imgH * ratio)  > imgW:
            resized_w = imgW
        else:
            resized_w = int(math.ceil(imgH * ratio))
        resized_image = cv2.resize(img, (resized_w, imgH))
        resized_image = resized_image.astype('float32')
        resized_image = resized_image.transpose((2, 0, 1)) / 255
        resized_image -= 0.5
        resized_image /= 0.5
        padding_im = np.zeros((imgC, imgH, imgW), dtype=np.float32)
        padding_im[:, :, 0:resized_w] = resized_image
        return padding_im

    def resize_norm_img_srn(self, img, image_shape):
        imgC, imgH, imgW = image_shape

        img_black = np.zeros((imgH, imgW))
        im_hei = img.shape[0]
        im_wid = img.shape[1]

        if im_wid <= im_hei * 1:
            img_new = cv2.resize(img, (imgH * 1, imgH))
        elif im_wid <= im_hei * 2:
            img_new = cv2.resize(img, (imgH * 2, imgH))
        elif im_wid <= im_hei * 3:
            img_new = cv2.resize(img, (imgH * 3, imgH))
        else:
            img_new = cv2.resize(img, (imgW, imgH))

        img_np = np.asarray(img_new)
        img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
        img_black[:, 0:img_np.shape[1]] = img_np
        img_black = img_black[:, :, np.newaxis]

        row, col, c = img_black.shape
        c = 1

        return np.reshape(img_black, (c, row, col)).astype(np.float32)

    def srn_other_inputs(self, image_shape, num_heads, max_text_length):

        imgC, imgH, imgW = image_shape
        feature_dim = int((imgH / 8) * (imgW / 8))

        encoder_word_pos = np.array(range(0, feature_dim)).reshape(
            (feature_dim, 1)).astype('int64')
        gsrm_word_pos = np.array(range(0, max_text_length)).reshape(
            (max_text_length, 1)).astype('int64')

        gsrm_attn_bias_data = np.ones((1, max_text_length, max_text_length))
        gsrm_slf_attn_bias1 = np.triu(gsrm_attn_bias_data, 1).reshape(
            [-1, 1, max_text_length, max_text_length])
        gsrm_slf_attn_bias1 = np.tile(
            gsrm_slf_attn_bias1,
            [1, num_heads, 1, 1]).astype('float32') * [-1e9]

        gsrm_slf_attn_bias2 = np.tril(gsrm_attn_bias_data, -1).reshape(
            [-1, 1, max_text_length, max_text_length])
        gsrm_slf_attn_bias2 = np.tile(
            gsrm_slf_attn_bias2,
            [1, num_heads, 1, 1]).astype('float32') * [-1e9]

        encoder_word_pos = encoder_word_pos[np.newaxis, :]
        gsrm_word_pos = gsrm_word_pos[np.newaxis, :]

        return [
            encoder_word_pos, gsrm_word_pos, gsrm_slf_attn_bias1,
            gsrm_slf_attn_bias2
        ]

    def process_image_srn(self, img, image_shape, num_heads, max_text_length):
        norm_img = self.resize_norm_img_srn(img, image_shape)
        norm_img = norm_img[np.newaxis, :]

        [encoder_word_pos, gsrm_word_pos, gsrm_slf_attn_bias1, gsrm_slf_attn_bias2] = \
            self.srn_other_inputs(image_shape, num_heads, max_text_length)

        gsrm_slf_attn_bias1 = gsrm_slf_attn_bias1.astype(np.float32)
        gsrm_slf_attn_bias2 = gsrm_slf_attn_bias2.astype(np.float32)
        encoder_word_pos = encoder_word_pos.astype(np.int64)
        gsrm_word_pos = gsrm_word_pos.astype(np.int64)

        return (norm_img, encoder_word_pos, gsrm_word_pos, gsrm_slf_attn_bias1,
                gsrm_slf_attn_bias2)

    # @torch.no_grad()
    def __call__(self, img_list):
        img_num = len(img_list)
        # Calculate the aspect ratio of all text bars
        width_list = []
        for img in img_list:
            width_list.append(img.shape[1] / float(img.shape[0]))
        # Sorting can speed up the recognition process
        indices = np.argsort(np.array(width_list))

        # rec_res = []
        rec_res = [['', 0.0]] * img_num
        batch_num = self.rec_batch_num
        elapse = 0
        model_elapse = 0
        data_pre_elapse = 0
        postprocess_elapse = 0
        for beg_img_no in range(0, img_num, batch_num):
            starttime = time.time()
            end_img_no = min(img_num, beg_img_no + batch_num)
            norm_img_batch = []
            real_w_batch = []
            max_wh_ratio = 0
            imgC, imgH, imgW = self.rec_image_shape
            for ino in range(beg_img_no, end_img_no):
                # h, w = img_list[ino].shape[0:2]
                h, w = img_list[indices[ino]].shape[0:2]
                wh_ratio = w * 1.0 / h
                resized_w = int(imgH / h * w)
                real_w_batch.append(resized_w)
                max_wh_ratio = max(max_wh_ratio, wh_ratio)
            for ino in range(beg_img_no, end_img_no):
                norm_img = self.resize_norm_img(img_list[indices[ino]],
                                                max_wh_ratio)
                norm_img = norm_img[np.newaxis, :]
                norm_img_batch.append(norm_img)

            norm_img_batch = np.concatenate(norm_img_batch)
            norm_img_batch = norm_img_batch.copy()
            data_pre_elapse += time.time() - starttime
            starttime = time.time()
            # select_infer_mode = 1
            # if select_infer_mode==1:
            with torch.no_grad():
            # with torch.cuda.amp.autocast():
                inp = torch.Tensor(norm_img_batch)
                # inp = torch.as_tensor(norm_img_batch)
                # inp = inp.half()
                inp = inp.to(self.args.device)

                prob_out = self.net(inp)
            preds = prob_out
            # elif select_infer_mode==2:
            #     inp = torch.Tensor(norm_img_batch)
            #
            #     inp = inp.to(self.args.device)
            #     preds = self.jit_model(inp)
            # else:
            #     norm_img_batch = onnxruntime.OrtValue.ortvalue_from_numpy(norm_img_batch, 'cuda', 0)  # 转换使用cuda推理，cpu可以省去
            #
            #     rec_inputs = {self.rec_model_onnx.get_inputs()[0].name: norm_img_batch}
            #     preds = self.rec_model_onnx.run(None, rec_inputs)
            #     preds = preds[0]
            model_elapse += time.time() - starttime
            starttime2 = time.time()
            rec_result = self.postprocess_op(preds,real_w_batch=real_w_batch)
            for rno in range(len(rec_result)):
                rec_res[indices[beg_img_no + rno]] = rec_result[rno]
            postprocess_elapse += time.time() - starttime2
            elapse += time.time() - starttime
        print(f'pre_data_time:{data_pre_elapse}')
        print(f'rec_model_time:{model_elapse}')
        print(f'postprocess_elapse:{postprocess_elapse}')

        return rec_res, elapse


def main(args):
    image_file_list = get_image_file_list(args.image_dir)
    text_recognizer = TextRecognizer(args)
    valid_image_file_list = []
    img_list = []
    for image_file in image_file_list:
        img, flag = check_and_read_gif(image_file)
        if not flag:
            img = cv2.imread(image_file)
        if img is None:
            print("error in loading image:{}".format(image_file))
            continue
        valid_image_file_list.append(image_file)
        img_list.append(img)
    try:
        rec_res, predict_time = text_recognizer(img_list)
    except Exception as e:
        print(e)
        exit()
    print('推理完成')
    with open('okay_rec_predict2.txt','w',encoding='utf8') as f:
        for ino,i in enumerate(rec_res):
            f.write(valid_image_file_list[ino]+'\t'+str(i)+'\n')
    # for ino in range(len(img_list)):
    #     print("Predicts of {}:{}".format(valid_image_file_list[ino], rec_res[
    #         ino]))
    print("Total predict time for {} images, cost: {:.3f}".format(
        len(img_list), predict_time))


if __name__ == '__main__':

    args = utility.parse_args()
    args.image_dir = r'/home/public/ocr_data/crop_img/crop_img'
    main(args)
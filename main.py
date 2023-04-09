import argparse
import os
import time
import numpy as np
import cv2
import torch

from nanodet.data.batch_process import stack_batch_img
from nanodet.data.collate import naive_collate
from nanodet.data.transform import Pipeline
from nanodet.model.arch import build_model
from nanodet.util import Logger, cfg, load_config, load_model_weight
from nanodet.util.path import mkdir

image_ext = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]
video_ext = ["mp4", "mov", "avi", "mkv"]
class_names = ['box_close', 'box_open', 'box_uncertain']


def cal_iou(box1, box2):
    """
    :param box1: = [left1, top1, right1, bottom1]
    :param box2: = [left2, top2, right2, bottom2]
    :return:
    """
    left1, top1, right1, bottom1 = box1
    left2, top2, right2, bottom2 = box2
    # 计算每个矩形的面积
    s1 = (right1 - left1) * (bottom1 - top1)  # b1的面积
    s2 = (right2 - left2) * (bottom2 - top2)  # b2的面积

    # 计算相交矩形
    left = max(left1, left2)
    top = max(top1, top2)
    right = min(right1, right2)
    bottom = min(bottom1, bottom2)

    # 相交框的w,h
    w = max(0, right - left)
    h = max(0, bottom - top)
    a1 = w * h  # C∩G的面积
    a2 = s1 + s2 - a1
    iou = a1 / a2  # iou = a1/ (s1 + s2 - a1)
    return iou


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--demo", default="video", help="demo type, eg. image, video and webcam"
    )
    parser.add_argument("--config", default="./nanodet/nanodet-plus-m_416_door.yml", help="model config file path")
    parser.add_argument("--model", default='./models/nanodet_model_best.pth', help="model file path")
    parser.add_argument("--path", default="./videos/video_09_01_230317_nightOpen_reserved_TEST.mp4",
                        help="path to images or video")
    parser.add_argument("--camid", type=int, default=0, help="webcam demo camera id")
    parser.add_argument(
        "--save_result",
        default=True,
        action="store_true",
        help="whether to save the inference result of image/video",
    )
    args = parser.parse_args()
    return args


class Predictor(object):
    def __init__(self, cfg, model_path, logger, device="cuda:0"):
        self.cfg = cfg
        self.device = device
        model = build_model(cfg.model)
        ckpt = torch.load(model_path, map_location=lambda storage, loc: storage)
        load_model_weight(model, ckpt, logger)
        if cfg.model.arch.backbone.name == "RepVGG":
            deploy_config = cfg.model
            deploy_config.arch.backbone.update({"deploy": True})
            deploy_model = build_model(deploy_config)
            from nanodet.model.backbone.repvgg import repvgg_det_model_convert

            model = repvgg_det_model_convert(model, deploy_model)
        self.model = model.to(device).eval()
        self.pipeline = Pipeline(cfg.data.val.pipeline, cfg.data.val.keep_ratio)

    def inference(self, img):
        img_info = {"id": 0}
        if isinstance(img, str):
            img_info["file_name"] = os.path.basename(img)
            img = cv2.imread(img)
        else:
            img_info["file_name"] = None

        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        meta = dict(img_info=img_info, raw_img=img, img=img)
        meta = self.pipeline(None, meta, self.cfg.data.val.input_size)
        meta["img"] = torch.from_numpy(meta["img"].transpose(2, 0, 1)).to(self.device)
        meta = naive_collate([meta])
        meta["img"] = stack_batch_img(meta["img"], divisible=32)
        with torch.no_grad():
            results = self.model.inference(meta)
        return meta, results

    def visualize(self, dets, meta, class_names, score_thres, wait=0):
        time1 = time.time()
        result_img = self.model.head.show_result(
            meta["raw_img"][0], dets, class_names, score_thres=score_thres, show=True
        )
        # print("viz time: {:.3f}s".format(time.time() - time1))
        return result_img


def get_image_list(path):
    image_names = []
    for maindir, subdir, file_name_list in os.walk(path):
        for filename in file_name_list:
            apath = os.path.join(maindir, filename)
            ext = os.path.splitext(apath)[1]
            if ext in image_ext:
                image_names.append(apath)
    return image_names


# True : current door in open suspicious status     ==> unsafe
# False : current door in closed status             ==> safe
def extractMetaAndDecision(detections, score_thresh=0.35, nms_threshold=0.6):
    all_box = []
    merged = False
    for label in detections:
        for bbox in detections[label]:
            score = bbox[-1]
            if score > score_thresh:
                x0, y0, x1, y1 = [int(i) for i in bbox[:4]]
                all_box.append([label, x0, y0, x1, y1, score])
    all_box.sort(key=lambda v: v[5])

    # for box in all_box:
    #     label, x0, y0, x1, y1, score = box
    #     text = "{}:{:.1f}%".format(class_names[label], score * 100)
    #     # print(text)

    # target : merge boxes of different classes, when 2 boxes are highly overlapped, make it uncertain
    refinedBoxes = []
    for box_a in all_box:
        for box_b in all_box:
            iou = cal_iou(box_a[1:-1], box_b[1:-1])
            if box_a[0] != box_b[0] and iou >= nms_threshold:
                print('merging performed !!!!!! overlapping detected:' + str(iou) + '@box_a:' + class_names[
                    box_a[0]] + ',score:' + str(box_a[-1]) +
                      '@box_b:' + class_names[box_b[0]] + ',score:' + str(box_b[-1]))
                # if overlapping happens, then will choose the more dangerous class as combined result ==> box_open
                # 0 for box
                if box_a[0] != 1:
                    all_box[all_box.index(box_a)][0] = -1
                break
    for item in all_box:
        if item[0] != -1:
            refinedBoxes.append(item)
        else:
            merged = True
    return refinedBoxes, merged


def main():
    args = parse_args()
    local_rank = 0
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    load_config(cfg, args.config)
    logger = Logger(local_rank, use_tensorboard=False)
    predictor = Predictor(cfg, args.model, logger, device="cuda:0")
    logger.log('Press "Esc", "q" or "Q" to exit.')
    current_time = time.localtime()
    if args.demo == "image":
        if os.path.isdir(args.path):
            files = get_image_list(args.path)
        else:
            files = [args.path]
        files.sort()
        for image_name in files:
            meta, res = predictor.inference(image_name)
            result_image = predictor.visualize(res[0], meta, cfg.class_names, 0.35)
            if args.save_result:
                save_folder = os.path.join(
                    cfg.save_dir, time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
                )
                mkdir(local_rank, save_folder)
                save_file_name = os.path.join(save_folder, os.path.basename(image_name))
                cv2.imwrite(save_file_name, result_image)
            ch = cv2.waitKey(0)
            if ch == 27 or ch == ord("q") or ch == ord("Q"):
                break
    elif args.demo == "video" or args.demo == "webcam":
        cap = cv2.VideoCapture(args.path if args.demo == "video" else args.camid)
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
        fps = cap.get(cv2.CAP_PROP_FPS)
        save_folder = os.path.join(
            cfg.save_dir, time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
        )
        mkdir(local_rank, save_folder)
        save_path = (
            os.path.join(save_folder, args.path.split("/")[-1])
            if args.demo == "video"
            else os.path.join(save_folder, "camera.mp4")
        )
        print(f"save_path is {save_path}")
        vid_writer = cv2.VideoWriter(
            save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (int(width), int(height))
        )
        while True:
            ret_val, frame = cap.read()
            if ret_val:
                meta, res = predictor.inference(frame)

                refined_boxes, merged = extractMetaAndDecision(res[0])

                result_frame = predictor.visualize(refined_boxes, meta, cfg.class_names, 0.35)
                if merged:
                    print('after mering boxes-count >>> ' + str(len(refined_boxes)))
                    cv2.waitKey()
                if args.save_result:
                    vid_writer.write(result_frame)
                ch = cv2.waitKey(1)
                if ch == 27 or ch == ord("q") or ch == ord("Q"):
                    break
            else:
                break


if __name__ == "__main__":
    main()

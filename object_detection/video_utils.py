import os
import glob
import shutil
import json
from zlib import crc32
from typing import List, Tuple, Union, Optional, Any, Dict

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import torch
from pathlib import Path
from natsort import natsorted

from image_processing.image_utils import scale_image
from plot.annotator import Annotator, Colors


def create_folder_template(sub_video_path):
    sub_video_path.mkdir(exist_ok=True)
    sub_video_images_path = sub_video_path / 'images'
    sub_video_images_path.mkdir(exist_ok=True)
    sub_video_labels_path = sub_video_path / 'labels'
    sub_video_labels_path.mkdir(exist_ok=True)
    sub_video_labels_ftid_path = sub_video_path / 'labels_ftid'
    sub_video_labels_ftid_path.mkdir(exist_ok=True)
    return sub_video_images_path

   
def overlay_bbox_on_videos(images_path, labels_path, video_name, img_size=(480, 640), fps=49, class_names=None):
    """
    Overlay some bounding box on the video
    Args:
        images_path (_type_): path to video frames folder
        labels_path (_type_): path to labels folder
        video_name (_type_): output video name
        img_size (tuple, optional): the resolution of the video frame. Defaults to (480, 640).
        fps (int, optional): self descriptive. Defaults to 49.
        class_names (_type_, optional): class names in all the labels file. Defaults to None.
    """

    h, w = img_size

    images = natsorted([x for x in Path(images_path).iterdir()], key=str)

    frame = cv2.imread(str(images[0]))
    height, width, layers = frame.shape

    video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(
        *'mp4v'), fps, (width, height))

    if class_names is None:
        class_names = [1, 2, 3]

    for image in tqdm(images):
        img = cv2.imread(str(image))
        annotator = Annotator(img, line_width=2, example=str('yolov5m'))
        name = image.stem
        labels = Path(labels_path) / image.stem

        if labels.exists():
            anns = np.genfromtxt(labels, dtype='str')
            anns = np.atleast_2d(anns)
            track_id = np.array(
                list(map(str_to_int, list(anns[:, 0]))), dtype='uint32')
            anns[:, 0] = track_id
            anns[:, 2:] = ccwh2xyxy(480, 640, anns[:, 2:].astype(
                'float32')).round().astype('int32')

            for ann in anns:
                cls = "{}{}{}".format(str(ann[0])[:2], str(
                    ann[0])[-2:], class_names[int(ann[1])])
                annotator.box_label(
                    ann[2:], cls, color=colors(int(ann[1]), True))

        video.write(annotator.result())

    cv2.destroyAllWindows()
    video.release()
    print("Saved to", video_name)

        
def merge_videos(videos: List[str], save_path: str):
    """
    Merge multiple videos into a single video file.

    Args:
        videos (List[str]): List of video file paths to merge. The videos will be merged in the order specified.
        save_path (str): File path to save the merged video.

    Returns:
        None
    """
    print("Merging videos")

    # Sort the list of videos using natural sorting
    videos = natsorted(videos)

    # Read the first video to get video properties
    vid_cap = cv2.VideoCapture(videos[0])
    fps = vid_cap.get(cv2.CAP_PROP_FPS)
    width = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Initialize the output video writer
    video = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    # Write all the frames sequentially to the new video
    for video_path in videos:
        vid_cap = cv2.VideoCapture(video_path)
        while True:
            ret, frame = vid_cap.read()
            if not ret:
                break
            video.write(frame)

    video.release()
    print("Saved to", save_path)

    
def get_video_splits(id, video_images_path,
                     model=None,
                     img_size=(480, 640), fps=49, min_frames_len=5, 
                     std_scale=3.5, crop=0.2,
                     candidate_ratio_lb=0.25, significant_ratio_lb=0.4,
                     visualize=False):

    h, w = img_size
    h_crop, w_crop = int(h*crop), int(w*crop)

    values = []
    video_images_path = Path(video_images_path)
    imgs = [str(x) for x in video_images_path.iterdir()]
    imgs = natsorted(imgs)
    psnr_diff = 0
    lim = 99999
    last_frame = 0
    result = []

    for idx, _ in enumerate(tqdm(imgs)):

        if idx + 1 == len(imgs) - 1:
            break

        if idx == 0:
            I = cv2.imread(imgs[idx])[h_crop:h-h_crop,
                                      w_crop:w-w_crop, :].flatten() / 255
            K = cv2.imread(imgs[idx+1])[h_crop:h-h_crop,
                                        w_crop:w-w_crop, :].flatten() / 255
        else:
            I = K
            K = cv2.imread(imgs[idx+1])[h_crop:h-h_crop,
                                        w_crop:w-w_crop, :].flatten() / 255
        mse = np.mean(np.power((I - K), 2))
        PSNR_Value = 10 * np.log10(1 / mse)

        if len(values[last_frame:]) > 0:
            psnr_diff = np.abs(PSNR_Value-np.mean(values[last_frame:]))
            lim = std_scale*np.std(values[last_frame:])

        if (idx-last_frame) > fps*min_frames_len and psnr_diff > lim and np.abs(1-PSNR_Value/values[-1]) >= candidate_ratio_lb:

            if (len(imgs) - idx) < fps*min_frames_len:
                values.append(PSNR_Value)
                continue

            if model != None:
                result_1 = model(imgs[idx])
                result_2 = model(imgs[idx+1])

                if np.abs(1-PSNR_Value/values[-1]) <= significant_ratio_lb and np.abs(result_1.pred[0].shape[0] - result_2.pred[0].shape[0]) <= 1:
                    values.append(PSNR_Value)
                    continue

            if visualize:
                plt.subplot(121)
                plt.imshow(cv2.imread(imgs[idx]))
                plt.subplot(122)
                plt.imshow(cv2.imread(imgs[idx+1]))
                plt.show()

            print("Cut at frame", idx+1, " time: ", np.round(idx/fps, 2))
            if model != None:
                print("PSNR-Obj: Frame {}: {}-{}; Frame {}: {}-{}".
                    format(idx, values[-1], result_1.pred[0].shape[0],  idx+1, PSNR_Value, result_2.pred[0].shape[0]))
            else:
                print("PSNR-Obj: Frame {}: {}; Frame {}: {}".format(idx, values[-1], idx+1, PSNR_Value))
            result.append(idx+1)
            last_frame = idx+1
            
        values.append(PSNR_Value)
    return result


def split_video(video_path, output_path, model=None):
    
    id = Path(video_path).stem
    Path(output_path).mkdir(exist_ok=True)
    save_dir = Path(output_path) / id
    save_dir_images_path = create_folder_template(save_dir)
    
    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)
    w = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    success, image = video.read()
    count = 0
    list_of_frames = []
    while success:
        cv2.imwrite(str(save_dir_images_path / "{}_frame_{}.jpg".format(id, count)), image)     # save frame as JPEG file      
        success,image = video.read()
        print('Reading frame:', count)
        list_of_frames.append(str(save_dir_images_path / "{}_frame_{}.jpg".format(id, count)))
        count += 1
        
    splits = get_video_splits(id=id, video_images_path=str(save_dir_images_path), model=model, fps=fps)
    vid_path, vid_writer = None, None
    
    if len(splits) > 0:
        
        print("Splits found")
        last_frame = 0
        
        for idx, split in enumerate(splits):
            
            vid_path = str(Path(output_path) / "{}_{}.mp4".format(id, idx) )
            if isinstance(vid_writer, cv2.VideoWriter):
                vid_writer.release()  # release previous video writer
            vid_writer = cv2.VideoWriter(vid_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
            for j, k in enumerate(range(last_frame, split)):
                img = cv2.imread(list_of_frames[k])
                vid_writer.write(img)
            
            last_frame = split
            
            if idx == len(splits) - 1:
                vid_path = str(Path(output_path) / "{}_{}.mp4".format(id, idx+1))
                if isinstance(vid_writer, cv2.VideoWriter):
                    vid_writer.release()  # release previous video writer
                vid_writer = cv2.VideoWriter(vid_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                for frame in list_of_frames[last_frame:]:
                    img = cv2.imread(frame)
                    vid_writer.write(img)
                
    print("Removing folder")
    shutil.rmtree(save_dir, ignore_errors=True)
    print(10*'-', 'Done', 10*'-')


if __name__ == "__main__":
    global colors
    colors = Colors()
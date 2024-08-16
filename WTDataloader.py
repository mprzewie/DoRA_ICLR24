from pathlib import Path

from projectaria_tools.core.sensor_data import TimeDomain, TimeQueryOptions
from projectaria_tools.projects.aea import AriaEverydayActivitiesDataProvider
from torchvision.datasets.utils import list_dir
from torchvision.datasets.folder import make_dataset
from torchvision.datasets.video_utils import VideoClips
from torchvision.datasets import VisionDataset
import torchvision
import os
from PIL import Image
import torch
import decord
import numpy as np
import random

import matplotlib.pyplot as plt


class DecordInit(object):

    def __init__(self, num_threads=1, **kwargs):
        self.num_threads = num_threads
        self.ctx = decord.cpu(0)
        self.kwargs = kwargs
        
    def __call__(self, filename):
        
        reader = decord.VideoReader(filename,
                                    ctx=self.ctx,
                                    num_threads=self.num_threads)
        return reader

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'sr={self.sr},'
                    f'num_threads={self.num_threads})')
        return repr_str

class Aria_dataset_1vid(torch.utils.data.Dataset):
    def __init__(self, root: Path, num_frames, step_between_clips, transform=None): #train=True, transform=None, target_transform=None,)
        self.root = root
        self.num_frames = num_frames
        self.step_between_clips = step_between_clips
        self.transform = transform

        self.provider = AriaEverydayActivitiesDataProvider(str(root))
        self.stream_id = self.provider.vrs.get_stream_id_from_label("camera-rgb")
        self.timestamps = self.provider.vrs.get_timestamps_ns(self.stream_id, TimeDomain.DEVICE_TIME)

    def __getitem__(self, index):
        start_frame_id = index
        frames = []

        for i in range(start_frame_id, start_frame_id+(self.num_frames*self.step_between_clips), self.step_between_clips):
            ts = self.timestamps[i]
            image_data = self.provider.vrs.get_image_data_by_time_ns(
                self.stream_id,
                ts,
                TimeDomain.DEVICE_TIME,
                TimeQueryOptions.BEFORE
            )
            image_np = image_data[0].to_numpy_array().transpose(1,0,2)
            # cl_pos = self.provider.mps.get_closed_loop_pose(ts, TimeQueryOptions.CLOSEST)
            # op_pos = self.provider.mps.get_open_loop_pose(ts, TimeQueryOptions.CLOSEST)
            # poses = dict(
            #     cl=cl_pos.transform_world_device.to_quat_and_translation(),
            #     # op=op_pos.transform_odometry_device.to_quat_and_translation()
            # )
            frames.append(image_np)

        video = np.stack(frames)

        # for i, v in enumerate(video):
        #     # ax[i].imshow(v)
        #     from PIL import Image
        #     Image.fromarray(v).convert('RGB').show()


        with torch.no_grad():
            video = torch.from_numpy(video)
            if self.transform is not None:
                video = self.transform(video)

        return video


    def __len__(self):
        return len(self.timestamps) - (self.num_frames * self.step_between_clips)

class WT_dataset_1vid(torch.utils.data.Dataset):
    

    def __init__(self,
                 video_path,
                 num_frames,
                 step_between_clips,
                 transform=None):
        
        self.path = video_path

        self.transform = transform
        self.num_frames = num_frames
        self.step_between_clips = step_between_clips
        self.v_decoder = DecordInit()
        v_reader = self.v_decoder(self.path)
        total_frames = len(v_reader)
        
        self.total_frames = total_frames 

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __getitem__(self, index):
        while True:
            try:
               
                v_reader = self.v_decoder(self.path)
                total_frames = len(v_reader)
                
                # Sampling video frames
                start_frame_ind = index 
                end_frame_ind = start_frame_ind + (self.num_frames * self.step_between_clips)


                frame_indice = np.arange(start_frame_ind, end_frame_ind, self.step_between_clips, dtype=int)
                
                video = v_reader.get_batch(frame_indice).asnumpy()

                # fig, ax = plt.subplots(ncols=len(video), figsize=(len(video)* 5, 5))
                # for i, v in enumerate(video):
                #     # ax[i].imshow(v)
                #     from PIL import Image
                #     Image.fromarray(v).convert('RGB').show()
                #
                # plt.show()

                del v_reader
                break
            except Exception as e:
                print(e)
                
        with torch.no_grad():
            video = torch.from_numpy(video)
            if self.transform is not None:
                video = self.transform(video)
                
        return video


    def __len__(self):
        return self.total_frames - (self.num_frames * self.step_between_clips)



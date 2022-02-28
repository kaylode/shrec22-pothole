from typing import Iterable, List
import os
from PIL import Image
import torch
import torch.utils.data as data
from typing import Iterable
import numpy as np
import cv2


class ConcatDataset(data.ConcatDataset):
    def __init__(self, datasets: Iterable[data.Dataset], **kwargs) -> None:
        super().__init__(datasets)

        # Workaround, not a good solution
        self.classnames = datasets[0].classnames
        self.collate_fn = datasets[0].collate_fn


class ChainDataset(data.ConcatDataset):
    def __init__(self, datasets: Iterable[data.Dataset], **kwargs) -> None:
        super().__init__(datasets)

        # Workaround, not a good solution
        self.classnames = datasets[0].classnames
        self.collate_fn = datasets[0].collate_fn

class ImageDataset(data.Dataset):
    """
    Dataset contains folder of images 
    image_dir: `str`
        path to folder of images
    txt_classnames: `str`
        path to .txt file contains classnames
    transform: `List`
        list of transformation
    """
    def __init__(self, image_dir: str, txt_classnames:str, transform: List =None, **kwargs):
        super().__init__()
        self.image_dir = image_dir
        self.txt_classnames = txt_classnames
        self.transform = transform
        self.load_data()

    def load_data(self):
        """
        Load filepaths into memory
        """
        with open(self.txt_classnames, 'r') as f:
            self.classnames = f.read().splitlines()
        self.fns = []
        image_names = os.listdir(self.image_dir)
        for image_name in image_names:
            image_path = os.path.join(self.image_dir, image_name)
            self.fns.append(image_path)

    def __getitem__(self, index: int):
        """
        Get an item from memory
        """
        image_path = self.fns[index]
        im = Image.open(image_path).convert('RGB')
        width, height = im.width, im.height
        im = np.array(im)

        if self.transform is not None: 
            im = self.transform(image=im)['image']

        return {
            "input": im, 
            'img_name': os.path.basename(image_path),
            'ori_size': [width, height]
        }

    def __len__(self):
        return len(self.fns)

    def collate_fn(self, batch: List):
        imgs = torch.stack([s['input'] for s in batch])
        img_names = [s['img_name'] for s in batch]
        ori_sizes = [i['ori_size'] for i in batch]

        return {
            'inputs': imgs,
            'img_names': img_names,
            'ori_sizes': ori_sizes
        }

class VideoDataset(data.Dataset):
    """
    Dataset path to a single video 
    video_dir: `str`
        path to a single of image
    txt_classnames: `str`
        path to .txt file contains classnames
    transform: `List`
        list of transformation
    """
    def __init__(self, video_path: str, txt_classnames:str, transform: List =None, **kwargs):
        super().__init__()
        self.video_path = video_path
        self.txt_classnames = txt_classnames
        self.transform = transform
        self.load_data()
        self.initialize_stream()

    def initialize_stream(self):
        self.stream = cv2.VideoCapture(self.video_path)
        self.current_frame_id = 0
        self.video_info = {}

        if self.stream.isOpened(): 
            # get self.stream property 
            self.WIDTH  = int(self.stream.get(cv2.CAP_PROP_FRAME_WIDTH))   # float `width`
            self.HEIGHT = int(self.stream.get(cv2.CAP_PROP_FRAME_HEIGHT))  # float `height`
            self.FPS = int(self.stream.get(cv2.CAP_PROP_FPS))
            self.NUM_FRAMES = int(self.stream.get(cv2.CAP_PROP_FRAME_COUNT))
            self.video_info = {
                'name': os.path.basename(self.video_path),
                'width': self.WIDTH,
                'height': self.HEIGHT,
                'fps': self.FPS,
                'num_frames': self.NUM_FRAMES
            }
        else:
            assert 0, f"Cannot read video {os.path.basename(self.video_path)}"

    def load_data(self):
        """
        Load filepaths into memory
        """
        with open(self.txt_classnames, 'r') as f:
            self.classnames = f.read().splitlines()
 
    def __getitem__(self, idx):
        success, im = self.stream.read()
        if not success:
            print(f"Cannot read frame {self.current_frame_id} from {self.video_info['name']}")
            self.current_frame_id = idx+1
            return None
        
        self.current_frame_id = idx+1

        width, height = im.shape[1], im.shape[1]
        if self.transform is not None: 
            im = self.transform(image=im)['image']

        return {
            "input": im, 
            'img_name': str(self.current_frame_id),
            'ori_size': [width, height]
        }

    def __len__(self):
        return self.NUM_FRAMES

    def collate_fn(self, batch: List):
        imgs = torch.stack([s['input'] for s in batch])
        img_names = [s['img_name'] for s in batch]
        ori_sizes = [i['ori_size'] for i in batch]

        return {
            'inputs': imgs,
            'img_names': img_names,
            'ori_sizes': ori_sizes
        }
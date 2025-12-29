import os.path
from numpy.random import randint
from torch.utils import data
import glob
import os
from dataloader.video_transform import *
import numpy as np
from imblearn.over_sampling import RandomOverSampler
import cv2
from PIL import Image
from PIL import ImageDraw
import numpy as np
import json
import random

class VideoRecord(object):
    def __init__(self, row):
        self._data = row

    @property
    def path(self): # 路径
        return self._data[0]

    @property       # 帧数
    def num_frames(self):
        return int(self._data[1])

    @property       # 标签
    def label(self):
        return int(self._data[2])

class VideoDataset(data.Dataset):
    def __init__(self, list_file, num_segments, duration, mode, transform, image_size,bounding_box_face,bounding_box_body):
        self.list_file = list_file
        self.duration = duration
        self.num_segments = num_segments
        self.transform = transform
        self.image_size = image_size
        self.mode = mode
        self.bounding_box_face = bounding_box_face
        self.bounding_box_body = bounding_box_body
        self._read_sample()
        self._parse_list()
        self._read_boxs()
        self._read_body_boxes()

    def _read_boxs(self):
        with open(self.bounding_box_face, 'r') as f:
            self.boxs = json.load(f)


    
    def _read_body_boxes(self):
        with open(self.bounding_box_body, 'r') as f:
            self.body_boxes = json.load(f)


    def _cv2pil(self,im_cv):
        cv_img_rgb = cv2.cvtColor(im_cv, cv2.COLOR_BGR2RGB)
        pillow_img = Image.fromarray(cv_img_rgb.astype('uint8'))
        return pillow_img

    def _pil2cv(self,im_pil):
        cv_img_rgb = np.array(im_pil)
        cv_img_bgr = cv2.cvtColor(cv_img_rgb, cv2.COLOR_RGB2BGR)
        return cv_img_bgr

    def _resize_image(self,im, width, height):
        w, h = im.shape[1], im.shape[0]
        r = min(width / w, height / h)
        new_w, new_h = int(w * r), int(h * r)
        im = cv2.resize(im, (new_w, new_h))
        pw = (width - new_w) // 2
        ph = (height - new_h) // 2
        top, bottom = ph, ph
        left, right = pw, pw
        if top + bottom + new_h < height:
            bottom += 1
        if left + right + new_w < width:
            right += 1
        im = cv2.copyMakeBorder(im, top, bottom, left, right, borderType=cv2.BORDER_CONSTANT, value=[0, 0, 0])
        return im, r

    def _face_detect(self,img,box,margin,mode = 'face'):
        if box is None:
            return img
        else:
            left, upper, right, lower = box
            left = int(left)
            upper = int(upper)
            right = int(right)
            lower = int(lower)
            left = max(0, left - margin)
            upper = max(0, upper - margin)
            right = min(img.width, right + margin)
            lower = min(img.height, lower + margin)
            if mode == 'face':
                img = img.crop((left, upper, right, lower))
                return img
            elif mode == 'body':
                occluded_image = img.copy()
                draw = ImageDraw.Draw(occluded_image)
                draw.rectangle([left, upper, right, lower], fill=(0, 0, 0))
                return occluded_image
    
    def _read_sample(self):
        tmp = [x.strip().split(' ') for x in open(self.list_file)]
        self.sample_list = [item for item in tmp]


    def _parse_list(self):
        #
        # Data Form: [video_id, num_frames, class_idx]
        #
        self.video_list = [VideoRecord(item) for item in self.sample_list]  
        print(('video number:%d' % (len(self.video_list))))

    def _get_train_indices(self, record):
        # 
        # Split all frames into seg parts, then select frame in each part randomly
        #
        average_duration = (record.num_frames - self.duration + 1) // self.num_segments
        if average_duration > 0:
            offsets = np.multiply(list(range(self.num_segments)), average_duration) + randint(average_duration, size=self.num_segments)
        elif record.num_frames > self.num_segments:
            offsets = np.sort(randint(record.num_frames - self.duration + 1, size=self.num_segments))
        else:
            offsets = np.pad(np.array(list(range(record.num_frames))), (0, self.num_segments - record.num_frames), 'edge')
        return offsets

    def _get_test_indices(self, record):
        # 
        # Split all frames into seg parts, then select frame in the mid of each part
        #
        if record.num_frames > self.num_segments + self.duration - 1:
            tick = (record.num_frames - self.duration + 1) / float(self.num_segments)
            offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])
        else:
            offsets = np.pad(np.array(list(range(record.num_frames))), (0, self.num_segments - record.num_frames), 'edge')
        return offsets

    def __getitem__(self, index):
        record = self.video_list[index]
        if self.mode == 'train':
            segment_indices = self._get_train_indices(record)
        elif self.mode == 'test':
            segment_indices = self._get_test_indices(record)
        return self.get(record, segment_indices)

    def get(self, record, indices):
        video_frames_path = glob.glob(os.path.join(record.path, '*'))
        video_frames_path.sort()
        
        # --- Safety Check Added ---
        if len(video_frames_path) == 0:
            print(f"Warning: No frames found for video {record.path}. Skipping or handling error.")
            # Option: Return zero tensors or handle gracefully. 
            # For now, let's create a dummy black image to avoid crashing, 
            # or better, raise a more informative error so user knows data is bad.
            # But to keep training running, let's try to reload a random sample or similar.
            # Here, we will just replicate the logic but with a safe check for p.
            # If absolutely empty, we can't do much. Let's assume data integrity is mostly ok 
            # but maybe index p is slightly off.
            # If really empty, raise error:
            raise RuntimeError(f"Video folder is empty: {record.path}")

        num_frames_real = len(video_frames_path)
        
        random_num = random.random()
        images = list()
        images_face = list()
        for seg_ind in indices:
            p = int(seg_ind)
            
            # --- Index Clamping ---
            if p >= num_frames_real:
                p = num_frames_real - 1
            if p < 0: 
                p = 0
                
            for i in range(self.duration):
                img_path = os.path.join(video_frames_path[p])
                parent_dir = os.path.dirname(img_path)
                file_name = os.path.basename(img_path)

                if parent_dir in self.boxs:
                    if file_name in self.boxs[parent_dir]:
                        box = self.boxs[parent_dir][file_name]
                    else:
                        box = None
                else:
                    box = None

                img_pil = Image.open(img_path)
                img_pil_face = Image.open(img_path)
                body_box_path = parent_dir
                body_box = self.body_boxes[body_box_path] if body_box_path in self.body_boxes else None
                if body_box is not None:
                    left, upper, right, lower = body_box
                    img_pil_body = img_pil.crop((left, upper, right, lower))
                else:
                    img_pil_body = img_pil

                img_cv_body = self._pil2cv(img_pil_body)
                img_cv_body, r = self._resize_image(img_cv_body, self.image_size, self.image_size)
                img_pil_body = self._cv2pil(img_cv_body)
                seg_imgs = [img_pil_body]
                

                seg_imgs_face = [self._face_detect(img_pil_face,box,margin=20,mode='face')]

                images.extend(seg_imgs)
                images_face.extend(seg_imgs_face)
                if p < record.num_frames - 1:
                    p += 1

        images = self.transform(images)
        images = torch.reshape(images, (-1, 3, self.image_size, self.image_size))

        images_face = self.transform(images_face)
        images_face = torch.reshape(images_face, (-1, 3, self.image_size, self.image_size))
        return images_face,images,record.label-1

    def __len__(self):
        return len(self.video_list)


def train_data_loader(list_file, num_segments, duration, image_size,dataset_name,bounding_box_face,bounding_box_body):
    if dataset_name == "RAER":
         train_transforms = torchvision.transforms.Compose([
            GroupTransform(torchvision.transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05)),
            RandomRotation(4),
            GroupResize(image_size),
            GroupRandomHorizontalFlip(),
            Stack(),
            ToTorchFormatTensor()])
            
    
    train_data = VideoDataset(list_file=list_file,
                              num_segments=num_segments, #16
                              duration=duration, #1
                              mode='train',
                              transform=train_transforms,
                              image_size=image_size,
                              bounding_box_face=bounding_box_face,
                              bounding_box_body=bounding_box_body
                              )
    return train_data


def test_data_loader(list_file, num_segments, duration, image_size,bounding_box_face,bounding_box_body):
    
    test_transform = torchvision.transforms.Compose([GroupResize(image_size),
                                                     Stack(),
                                                     ToTorchFormatTensor()])
    
    test_data = VideoDataset(list_file=list_file,
                             num_segments=num_segments,
                             duration=duration,
                             mode='test',
                             transform=test_transform,
                             image_size=image_size,
                             bounding_box_face=bounding_box_face,
                             bounding_box_body=bounding_box_body
                             )
    return test_data
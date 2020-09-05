from pathlib import Path
import json
import paddle
from VideoLoader import VideoLoader
from temporal_transforms import Compose as TemporalCompose,TemporalRandomCrop, TemporalEvenCrop,SlidingWindow
from spatial_transforms import Compose, MultiScaleCornerCrop,RandomResizedCrop, RandomHorizontalFlip, ScaleValue, Normalize, Resize, CenterCrop

import numpy as np
import copy
from  autoaugment import ResNet3DPolicy
from temporal_transforms import Shuffle
import random
def image_name_formatter(x):
    return f'image_{x:05d}.jpg'

def get_class_labels(data):
    class_labels_map = {}
    index = 0
    for class_label in data['labels']:
        class_labels_map[class_label] = index
        index += 1
    return class_labels_map

def get_database(data, subset, root_path, video_path_formatter):
    video_ids = []
    video_paths = []
    annotations = []

    for key, value in data['database'].items():
        this_subset = value['subset']
        if this_subset == subset:
            video_ids.append(key)
            annotations.append(value['annotations'])
            if 'video_path' in value:
                video_paths.append(Path(value['video_path']))
            else:
                label = value['annotations']['label']
                video_paths.append(video_path_formatter(root_path, label, key))

    return video_ids, video_paths, annotations


class VideoDataset():
    def __init__(self,
                 root_path,
                 annotation_path,
                 subset,
                 num_worker=4,
                 mode='train',
                 batch_size=2,
                 spatial_transform=None,
                 temporal_transform=None,
                 target_transform=None,
                 video_loader=None,
                 video_path_formatter=(lambda root_path, label, video_id:
                                       root_path / label / video_id),
                 image_name_formatter=lambda x: f'image_{x:05d}.jpg',
                 target_type='label',
                 batch_mode='batch'):
        self.data, self.class_names = self.__make_dataset(
            root_path, annotation_path, subset, video_path_formatter)

        self.spatial_transform = spatial_transform
        self.temporal_transform = temporal_transform
        self.target_transform = target_transform

        if video_loader is None:
            self.loader = VideoLoader(image_name_formatter)
        else:
            self.loader = video_loader

        self.target_type = target_type
        self.num_worker = num_worker
        self.mode = mode
        if self.mode == 'train':
            self.drop_last = True
        else:
            self.drop_last = False
        self.batch_size = batch_size
        self.batch_mode = batch_mode
        self.img_mean = np.array([0.4477, 0.4209, 0.3906]).reshape(
            [1, 1, 1, 3]).astype(np.float32)
        self.img_std = np.array([0.2767, 0.2695, 0.2714]).reshape(
            [1, 1, 1, 3]).astype(np.float32)
        scales = [1.0]
        for i in range(1, 5):
            scales.append(scales[-1] * 0.84089641525)
        self.scales = scales
    def __make_dataset(self, root_path, annotation_path, subset,
                       video_path_formatter):
        with annotation_path.open('r') as f:
            data = json.load(f)
        video_ids, video_paths, annotations = get_database(
            data, subset, root_path, video_path_formatter)
        class_to_idx = get_class_labels(data)
        idx_to_class = {}
        for name, label in class_to_idx.items():
            idx_to_class[label] = name

        n_videos = len(video_ids)
        dataset = []
        for i in range(n_videos):
            if i % (n_videos // 5) == 0:
                print('dataset loading [{}/{}]'.format(i, len(video_ids)))

            if 'label' in annotations[i]:
                label = annotations[i]['label']
                label_id = class_to_idx[label]
            else:
                label = 'test'
                label_id = -1

            video_path = video_paths[i]
            if not video_path.exists():
                continue

            segment = annotations[i]['segment']
            if segment[1] == 1:
                continue

            frame_indices = list(range(segment[0], segment[1]))
            sample = {
                'video': video_path,
                'segment': segment,
                'frame_indices': frame_indices,
                'video_id': video_ids[i],
                'label': label_id
            }
            dataset.append(sample)

        return dataset, idx_to_class

    def __loading(self, path, frame_indices):
        clip = self.loader(path, frame_indices)
        if self.spatial_transform is not None:
            self.spatial_transform.randomize_parameters()
            clip = [self.spatial_transform(img) for img in clip]
        clip = np.stack(clip)
        clip = np.transpose(clip, [3, 0, 1, 2]).astype('float32')

        return clip

    def get_singel_reader(self):
        return self._get_reader(self.data)
    def get_multiprocess_reader(self):
        video_lens= len(self.data)
        total_batch_size = video_lens // self.batch_size
        worker_per_length = total_batch_size // self.num_worker *self.batch_size
        readers = []
        for i in range(self.num_worker):
            if i == self.num_worker - 1:
                r = self._get_reader(self.data[i*worker_per_length:])
            else:
                r = self._get_reader(self.data[i*worker_per_length:(i+1)*worker_per_length])
            readers.append(r)
        return paddle.reader.multiprocess_reader(readers, False, queue_size=50)
    def _get_reader(self,data):
        def reader():
            if self.mode == 'train':
                print('shuffle')
                np.random.shuffle(data)
            for index in range(len(data)):
                path = data[index]['video']
                if isinstance(self.target_type, list):
                    target = [data[index][t] for t in self.target_type]
                else:
                    target = data[index][self.target_type]

                frame_indices = data[index]['frame_indices']
                if self.temporal_transform is not None:
                    frame_indices = self.temporal_transform(frame_indices)

                clip = self.__loading(path, frame_indices)

                if self.target_transform is not None:
                    target = self.target_transform(target)

                yield clip, [target]
        return reader


    def __len__(self):
        return len(self.data)

class VideoDatasetMultiClips(VideoDataset):

    def __loading(self, path, video_frame_indices):
        clips = []
        segments = []
        for clip_frame_indices in video_frame_indices:
            clip = self.loader(path, clip_frame_indices)
            if self.spatial_transform is not None:
                self.spatial_transform.randomize_parameters()
                clip = [self.spatial_transform(img) for img in clip]
            clip = np.stack(clip)
            clip = np.transpose(clip, [3, 0, 1, 2]).astype('float32')
            clips.append(clip)
            segments.append(
                [min(clip_frame_indices),
                 max(clip_frame_indices) + 1])

        return clips, segments

    def _get_reader(self, data):
        def reader():
            if self.mode == 'train':
                np.random.shuffle(data)
            for index in range(len(data)):
                path = data[index]['video']

                video_frame_indices = data[index]['frame_indices']
                if self.temporal_transform is not None:
                    video_frame_indices = self.temporal_transform(video_frame_indices)

                clips, segments = self.__loading(path, video_frame_indices)

                if isinstance(self.target_type, list):
                    target = [data[index][t] for t in self.target_type]
                else:
                    target = data[index][self.target_type]

                if 'segment' in self.target_type:
                    if isinstance(self.target_type, list):
                        segment_index = self.target_type.index('segment')
                        targets = []
                        for s in segments:
                            targets.append(copy.deepcopy(target))
                            targets[-1][segment_index] = s
                    else:
                        targets = segments
                else:
                    targets = [target for _ in range(len(segments))]

                yield clips, targets
        def batch_iter_reader():
            batch_outs = []
            for outs in reader():
                batch_outs.append(outs)
                if len(batch_outs) == self.batch_size:
                    new_batch_outs = []
                    for batch_out in batch_outs:
                        images,labels = batch_out
                        for image,label in zip(images, labels):
                            new_batch_outs.append([image, [label]])
                    yield new_batch_outs
                    batch_outs = []
            if not self.drop_last:
                if len(batch_outs) != 0:
                    new_batch_outs = []
                    for batch_out in batch_outs:
                        images,labels = batch_out
                        for image,label in zip(images, labels):
                            new_batch_outs.append([image, [label]])
                    yield new_batch_outs

        if self.batch_mode == 'batch':
            return batch_iter_reader
        else:
            return reader

def custom_reader(root_path, annotation_path,batch_size=1,mode='train'):
    video_path_formatter = (lambda root_path, label, video_id:
                            root_path / label /  video_id)
    loader = VideoLoader(image_name_formatter)
    if mode == 'train':
        subset = 'training'
        temporal_transform=[TemporalRandomCrop(16)]
        temporal_transform = TemporalCompose(temporal_transform)

        # spatial_transform=[]
        scales = [1.0]
        scale_step = 1 / (2**(1 / 4))
        for _ in range(1, 5):
            scales.append(scales[-1] * scale_step)
        spatial_transform = [MultiScaleCornerCrop(112, scales)]

        # spatial_transform.append(
        #     RandomResizedCrop(
        #         112, (0.25, 1.0),
        #         (0.75, 1.0 / 0.75)))
        spatial_transform.append(RandomHorizontalFlip())
        spatial_transform.append(ScaleValue(255.0))
        spatial_transform.append(Normalize(mean=[0.4477, 0.4209, 0.3906], std=[0.2767, 0.2695, 0.2714]))
        spatial_transform = Compose(spatial_transform)

        video_dataset = VideoDataset(root_path=root_path,
                                     annotation_path=annotation_path,
                                     subset=subset,
                                     video_loader=loader,
                                     mode=mode,
                                     batch_size=batch_size,
                                     video_path_formatter=video_path_formatter,
                                     temporal_transform=temporal_transform,
                                     spatial_transform=spatial_transform)
    elif mode == 'val':
        subset = 'validation'
        temporal_transform=[TemporalEvenCrop(16, 3)]
        temporal_transform = TemporalCompose(temporal_transform)

        spatial_transform = [
            Resize(112),
            CenterCrop(112),
            ScaleValue(255.0),
            Normalize(mean=[0.4477, 0.4209, 0.3906], std=[0.2767, 0.2695, 0.2714])
        ]
        spatial_transform = Compose(spatial_transform)

        video_dataset = VideoDatasetMultiClips(root_path=root_path,
                                               annotation_path=annotation_path,
                                               subset=subset,
                                               video_loader=loader,
                                               mode=mode,
                                               batch_size=batch_size // 3,
                                               video_path_formatter=video_path_formatter,
                                               temporal_transform=temporal_transform,
                                               spatial_transform=spatial_transform)
    else:
        subset = 'validation'
        temporal_transform = [SlidingWindow(16, 16)]
        temporal_transform = TemporalCompose(temporal_transform)
        spatial_transform = [
            Resize(112),
            CenterCrop(112),
            ScaleValue(255.0),
            Normalize(mean=[0.4477, 0.4209, 0.3906], std=[0.2767, 0.2695, 0.2714])
        ]
        spatial_transform = Compose(spatial_transform)

        video_dataset = VideoDatasetMultiClips(
            root_path=root_path,
            annotation_path=annotation_path,
            subset=subset,
            temporal_transform=temporal_transform,
            spatial_transform=spatial_transform,
            video_loader=loader,
            video_path_formatter=video_path_formatter,
            target_type=['video_id', 'segment'],
            batch_mode='iter')
        return video_dataset.get_multiprocess_reader(),  video_dataset.class_names
    return video_dataset.get_multiprocess_reader()
    # return video_dataset.get_singel_reader()



from mixup import create_mixup_reader
import paddle
from paddle import fluid
if __name__ == '__main__':
    root_path = '/Users/alex/baidu/3dresnet-data/UCF-101-jpg'
    annotation_path = 'ucf101_json/ucf101_01.json'
    reader = custom_reader(Path(root_path), Path(annotation_path),mode='val',batch_size=128)
    # r = create_mixup_reader(0.2, reader)
    reader = paddle.batch(fluid.io.shuffle(reader, 128), batch_size=128, drop_last=True)

    for epoch in range(3):
        for data in reader():
            print('')
from paddle import fluid
from paddle.fluid.regularizer import L2Decay
from paddle.fluid.dygraph import to_variable, Linear
from model.resnet import generate_model
import numpy as np
from reader import custom_reader
from mixup import create_mixup_reader
from pathlib import Path
from ReduceLROnPlateau import ReduceLROnPlateau
import time
import datetime
import os
import paddle
from visualdl import LogWriter
import json

def get_video_results(outputs, class_names, output_topk):
    outputs = to_variable(outputs)
    sorted_scores, locs = fluid.layers.topk(outputs,
                                     k=min(output_topk, len(class_names)))

    video_results = []
    for i in range(sorted_scores.shape[0]):
        video_results.append({
            'label': class_names[locs.numpy()[i]],
            'score': float(sorted_scores.numpy()[i])
        })

    return video_results

BATCH_SIZE = 1
root_path = '/home/aistudio/dataset/UCF-101-jpg'
# root_path = '/Users/alex/baidu/UCF-101-jpg'
annotation_path = 'ucf101_json/ucf101_01.json'
infer_reader, infer_class_names = custom_reader(Path(root_path), Path(annotation_path), mode='test', batch_size=BATCH_SIZE)

use_gpu = True
place = fluid.CUDAPlace(0) if use_gpu else fluid.CPUPlace()

with fluid.dygraph.guard(place):
    with fluid.dygraph.no_grad():
        from collections import defaultdict
        model = generate_model(50, n_classes=101)
        state_dic, _ = fluid.dygraph.load_dygraph('./model_weights/best_accuracy_1.pdparams')
        model.set_dict(state_dic)
        model.eval()
        results = {'results': defaultdict(list)}
        for i, (inputs, targets) in enumerate(infer_reader()):
            video_ids, segments = zip(*targets)
            inputs = np.stack(inputs)
            inputs = to_variable(inputs)
            outputs = model(inputs)
            outputs = fluid.layers.softmax(outputs, axis=1).numpy()
            for j in range(outputs.shape[0]):
                results['results'][video_ids[j]].append({
                    'segment': segments[j],
                    'output': outputs[j]
                })
            if i % 100 == 0:
                print(f'sample_id:{i}')
    class_names = infer_class_names
    inference_results = {'results': {}}
    for video_id, video_results in results['results'].items():
        video_outputs = [
            segment_result['output'] for segment_result in video_results
        ]
        video_outputs = np.stack(video_outputs)
        average_scores = np.mean(video_outputs, axis=0)
        inference_results['results'][video_id] = get_video_results(
            average_scores, class_names, 5)
    with Path('val.json').open('w') as f:
        json.dump(inference_results, f)
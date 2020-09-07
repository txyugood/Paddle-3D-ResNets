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
import math
from paddle.fluid import ParamAttr
from utils import AverageMeter
import argparse

num_sample = 9537
BATCH_SIZE = 128
MAX_EPOCH = 200
n_classes = 101
best_accuracy = 0.0
MIX_UP = True


def get_module_name(name,l=1):
    name = name.split('.')
    if name[0] == 'module':
        i = 1
    else:
        i = 0
    if name[i] == 'features':
        i += 1
    return '.'.join(name[i:i+l])
def _calc_label_smoothing_loss(softmax_out, label, class_dim, epsilon):
    """Calculate label smoothing loss

    Returns:
        label smoothing loss

    """

    label_one_hot = fluid.layers.one_hot(input=label, depth=class_dim)
    smooth_label = fluid.layers.label_smooth(
        label=label_one_hot, epsilon=epsilon, dtype="float32")
    loss = fluid.layers.cross_entropy(
        input=softmax_out, label=smooth_label, soft_label=True)
    return loss


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--mixup',
        action='store_true',
        help='If true, enable mixup data augmentation.')
    args = parser.parse_args()
    MIX_UP = args.mixup
    root_path = '/home/aistudio/dataset/UCF-101-jpg'
    # root_path = '/Users/alex/baidu/3dresnet-data/UCF-101-jpg'
    annotation_path = 'ucf101_json/ucf101_01.json'
    train_reader = custom_reader(Path(root_path), Path(annotation_path), mode='train', batch_size=BATCH_SIZE)
    val_reader = custom_reader(Path(root_path), Path(annotation_path), mode='val', batch_size=BATCH_SIZE)
    train_reader = paddle.batch(fluid.io.shuffle(train_reader, BATCH_SIZE), batch_size=BATCH_SIZE, drop_last=False)

    if MIX_UP:
        train_reader = create_mixup_reader(0.2, train_reader)
        train_reader = paddle.batch(
            train_reader,
            batch_size=BATCH_SIZE,
            drop_last=False)

    iter_per_epoch = int(math.ceil(num_sample / BATCH_SIZE))
    boundaries = [iter_per_epoch * 50, iter_per_epoch * 100, iter_per_epoch * 150]
    use_gpu = True
    place = fluid.CUDAPlace(0) if use_gpu else fluid.CPUPlace()

    with fluid.dygraph.guard(place):
        train_data_loader = fluid.io.DataLoader.from_generator(capacity=5)
        val_data_loader = fluid.io.DataLoader.from_generator(capacity=5)
        train_data_loader.set_sample_list_generator(train_reader, places=place)
        val_data_loader.set_sample_list_generator(val_reader, places=place)
        model = generate_model(50, n_classes=1039)
        state_dic, _ = fluid.dygraph.load_dygraph('paddle_resnet50_mk.pdparams')
        model.set_dict(state_dic)
        stdv = 1. / math.sqrt(model.fc_in_dim)
        model.fc = Linear(model.fc_in_dim, n_classes,
                          param_attr=ParamAttr(name='fc.weight',
                                               initializer=fluid.initializer.Uniform(-stdv, stdv)),
                          bias_attr=ParamAttr(name='fc.bias',
                                              initializer=fluid.initializer.Uniform(-stdv, stdv))
                          )

        parameters = []
        add_flag = False
        for k, v in model.named_parameters():
            name = get_module_name(k,1)
            if 'layer4' == name:
                add_flag = True
            if add_flag:
                parameters.append(v)
                print(k)

        if MIX_UP:
            lr = fluid.dygraph.ExponentialDecay(
                learning_rate=0.003,
                decay_steps=MAX_EPOCH * iter_per_epoch,
                decay_rate=0.5
            )
            opt = fluid.optimizer.Momentum(
            learning_rate=lr,
            momentum=0.9,
            parameter_list=parameters,
            regularization=L2Decay(1e-4))
        else:
            lr = ReduceLROnPlateau(
                learning_rate=0.003,
                mode='min',
                verbose=True,
                patience=10
            )
            opt = fluid.optimizer.Momentum(
                learning_rate=lr,
                momentum=0.9,
                parameter_list=parameters,
                regularization=L2Decay(1e-3))

        for epoch in range(1, MAX_EPOCH + 1):
            batch_time = AverageMeter()
            data_time = AverageMeter()
            losses = AverageMeter()
            accuracies = AverageMeter()

            end_time = time.time()
            model.train()
            with LogWriter(logdir="./log/train") as writer:
                for i, data in enumerate(train_data_loader()):
                    data_time.update(time.time() - end_time)
                    if MIX_UP:
                        img, l1, l2, lam = data
                        lam = fluid.layers.cast(lam, 'float32')
                    else:
                        img, label = data
                    out = model(img)
                    out = fluid.layers.softmax(out)

                    if MIX_UP:
                        loss_a = _calc_label_smoothing_loss(out, l1, n_classes, 0.1)
                        loss_b = _calc_label_smoothing_loss(out, l2, n_classes, 0.1)
                        loss_a_mean = fluid.layers.mean(loss_a)
                        loss_b_mean = fluid.layers.mean(loss_b)
                        loss = lam * loss_a_mean + (1.0 - lam) * loss_b_mean
                        loss = fluid.layers.mean(x=loss)
                        acc = fluid.layers.accuracy(out, l1)
                    else:
                        loss = fluid.layers.cross_entropy(out, label)
                        loss = fluid.layers.reduce_mean(loss)
                        acc = fluid.layers.accuracy(out, label)

                    losses.update(loss.numpy()[0], img.shape[0])
                    accuracies.update(acc.numpy()[0], img.shape[0])

                    loss.backward()
                    opt.minimize(loss)
                    model.clear_gradients()
                    batch_time.update(time.time() - end_time)
                    end_time = time.time()
                    print('Epoch: [{0}][{1}/{2}]\t'
                          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                          'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                          'Acc {acc.val:.3f} ({acc.avg:.3f})'.format(epoch,
                                                                     i + 1,
                                                                     iter_per_epoch,
                                                                     batch_time=batch_time,
                                                                     data_time=data_time,
                                                                     loss=losses,
                                                                     acc=accuracies))


                # 向记录器添加一个tag为`acc`的数据
                writer.add_scalar(tag="train/acc", step=epoch, value=accuracies.avg)
                # 向记录器添加一个tag为`loss`的数据
                writer.add_scalar(tag="train/loss", step=epoch, value=losses.avg)

            with LogWriter(logdir="./log/eval") as writer:
                with fluid.dygraph.no_grad():
                    model.eval()

                    batch_time = AverageMeter()
                    data_time = AverageMeter()
                    losses = AverageMeter()
                    accuracies = AverageMeter()
                    end_time = time.time()
                    for i, data in enumerate(val_data_loader()):
                        data_time.update(time.time() - end_time)
                        img, label = data
                        out = model(img)
                        out = fluid.layers.softmax(out)
                        acc = fluid.layers.accuracy(out, label)
                        loss = fluid.layers.cross_entropy(out, label)
                        loss = fluid.layers.mean(x=loss)

                        losses.update(loss.numpy()[0], img.shape[0])
                        accuracies.update(acc.numpy()[0], img.shape[0])

                        batch_time.update(time.time() - end_time)
                        end_time = time.time()

                        print('Epoch: [{0}][{1}/{2}]\t'
                              'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                              'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                              'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                              'Acc {acc.val:.3f} ({acc.avg:.3f})'.format(
                            epoch,
                            i + 1,
                            91,
                            batch_time=batch_time,
                            data_time=data_time,
                            loss=losses,
                            acc=accuracies))


                        # 向记录器添加一个tag为`acc`的数据
                    writer.add_scalar(tag="train/acc", step=epoch, value=accuracies.avg)
                    # 向记录器添加一个tag为`loss`的数据
                    writer.add_scalar(tag="train/loss", step=epoch, value=losses.avg)

                    print(f'Test acc :{accuracies.avg}, loss:{losses.avg}')
                    if not os.path.exists('./model_weights'):
                        os.makedirs('./model_weights')
                    with open('./model_weights/eval_log.txt', 'a') as f:
                        f.write(f'epoch:{epoch} Test acc :{accuracies.avg}, loss:{losses.avg}\n')
                    # lr.step(to_variable(np.array([losses.avg]).astype('float32')))

                    if accuracies.avg > best_accuracy:
                        with open('./model_weights/best_accuracy.txt', 'w') as f:
                            f.write(f'{epoch}:{accuracies.avg }')
                        best_accuracy = accuracies.avg
                        fluid.save_dygraph(model.state_dict(), './model_weights/best_accuracy')
                    if epoch % 10 == 0:
                        fluid.save_dygraph(model.state_dict(), f'./model_weights/{epoch}_model')
        fluid.save_dygraph(model.state_dict(), f'./model_weights/{MAX_EPOCH}_model')

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

num_sample = 9537
BATCH_SIZE = 128
MAX_EPOCH = 200
n_classes = 101
best_accuracy = 0.0
os.system('rm model_weights/eval_log.txt')


def get_module_name(name,l=1):
    name = name.split('.')
    if name[0] == 'module':
        i = 1
    else:
        i = 0
    if name[i] == 'features':
        i += 1
    return '.'.join(name[i:i+l])

if __name__ == '__main__':
    # root_path = '/home/aistudio/dataset/UCF-101-jpg'
    root_path = '/Users/alex/baidu/3dresnet-data/UCF-101-jpg'
    annotation_path = 'ucf101_json/ucf101_01.json'
    train_reader = custom_reader(Path(root_path), Path(annotation_path), mode='train', batch_size=BATCH_SIZE)
    val_reader = custom_reader(Path(root_path), Path(annotation_path), mode='val', batch_size=BATCH_SIZE)

    iter_per_epoch = num_sample // BATCH_SIZE
    boundaries = [iter_per_epoch * 50, iter_per_epoch * 100, iter_per_epoch * 150]
    use_gpu = False
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

        # parameters = []
        # add_flag = False
        # for k, v in model.named_parameters():
        #     name = get_module_name(k,1)
        #     if 'layer4' == name:
        #         add_flag = True
        #     if add_flag:
        #         if 'bn' in k:
        #             v.optimize_attr['learning_rate'] = 0.0
        #         parameters.append(v)
        #         print(k)
        # lr = fluid.dygraph.PiecewiseDecay(
        #     boundaries, [0.01, 0.001, 0.0001, 0.00001], 0
        # )
        # lr = fluid.dygraph.ExponentialDecay(
        #     learning_rate=0.01,
        #     decay_steps=MAX_EPOCH * iter_per_epoch,
        #     decay_rate=0.01
        # )
        lr = ReduceLROnPlateau(
            learning_rate=0.01,
            mode='min',
            verbose=True,
            patience=10
        )
        opt = fluid.optimizer.Momentum(
            learning_rate=lr,
            momentum=0.9,
            parameter_list=model.parameters(),
            # parameter_list=parameters,
            regularization=L2Decay(1e-3))

        batch_id = 0
        eval_batch_num = 0
        for epoch in range(1, MAX_EPOCH + 1):
            model.train()
            accs = []
            losses = []
            start_time = time.time()
            with LogWriter(logdir="./log/train") as writer:
                for i, data in enumerate(train_data_loader()):
                    img, label = data
                    out = model(img)
                    out = fluid.layers.softmax(out)
                    loss = fluid.layers.cross_entropy(out, label)
                    avg_loss = fluid.layers.mean(loss)
                    acc = fluid.layers.accuracy(out, label)

                    avg_loss.backward()
                    opt.minimize(avg_loss)
                    model.clear_gradients()
                    accs.append(acc.numpy()[0])
                    losses.append(avg_loss.numpy()[0])
                    batch_id += 1
                    if batch_id % 1 == 0:
                        per_log_time = time.time() - start_time
                        start_time = time.time()
                        remain_time = (iter_per_epoch - i) / 1 * per_log_time
                        print(f'epoch:{epoch} batch id:{batch_id} '
                              f'loss:{sum(losses) / len(losses)} acc:{sum(accs) / len(accs)} remain time:{str(datetime.timedelta(seconds=remain_time))}')
                # 向记录器添加一个tag为`acc`的数据
                writer.add_scalar(tag="train/acc", step=epoch, value=sum(accs) / len(accs))
                # 向记录器添加一个tag为`loss`的数据
                writer.add_scalar(tag="train/loss", step=epoch, value=sum(losses) / len(losses))

            with LogWriter(logdir="./log/eval") as writer:
                with fluid.dygraph.no_grad():
                    model.eval()
                    total_acc = 0
                    total_loss = 0
                    for i, data in enumerate(val_data_loader()):
                        img, label = data
                        out = model(img)
                        out = fluid.layers.softmax(out)
                        acc = fluid.layers.accuracy(out, label)
                        loss = fluid.layers.cross_entropy(out, label)
                        loss = fluid.layers.mean(x=loss)
                        total_acc + acc
                        total_loss += loss
                        avg_acc = total_acc / (i + 1)
                        print(f'Test batch id:{i}, acc:{avg_acc.numpy()[0]}')
                        eval_batch_num += 1
                        avg_loss = total_loss / (i + 1)
                        # 向记录器添加一个tag为`acc`的数据
                    writer.add_scalar(tag="train/acc", step=epoch, value=avg_acc.numpy()[0])
                    # 向记录器添加一个tag为`loss`的数据
                    writer.add_scalar(tag="train/loss", step=epoch, value=avg_loss.numpy()[0])

                    print(f'Test acc :{avg_acc.numpy()[0]}, loss:{avg_loss.numpy()[0]}')
                    if not os.path.exists('./model_weights'):
                        os.makedirs('./model_weights')
                    with open('./model_weights/eval_log.txt', 'a') as f:
                        f.write(f'epoch:{epoch} Test acc :{avg_acc.numpy()[0]}, loss:{avg_loss.numpy()[0]}\n')
                    lr.step(avg_loss)

                    if avg_acc > best_accuracy:
                        with open('./model_weights/best_accuracy.txt', 'w') as f:
                            f.write(f'{epoch}:{avg_acc.numpy()[0]}')
                        best_accuracy = avg_acc.numpy()[0]
                        fluid.save_dygraph(model.state_dict(), './model_weights/best_accuracy')
                    if epoch % 10 == 0:
                        fluid.save_dygraph(model.state_dict(), f'./model_weights/{epoch}_model')
        fluid.save_dygraph(model.state_dict(), f'./model_weights/{MAX_EPOCH}_model')

from paddle import fluid
import numpy as np
from paddle.fluid.layer_helper import LayerHelper
from paddle.fluid.dygraph import to_variable, Conv2D, Pool2D, BatchNorm, Linear

class ConvBNLayer(fluid.dygraph.Layer):
    def __init__(self,
                 num_channels,
                 num_filters,
                 filter_size,
                 stride=1,
                 groups=1,
                 act=None):
        super(ConvBNLayer, self).__init__()

        self._conv = Conv2D(
            num_channels=num_channels,
            num_filters=num_filters,
            filter_size=filter_size,
            stride=stride,
            padding=(filter_size - 1) // 2,
            groups=groups,
            act=None,
            bias_attr=False)

        self._batch_norm = BatchNorm(num_filters, act=act)

    def forward(self, inputs):
        y = self._conv(inputs)
        y = self._batch_norm(y)

        return y


class BottleneckBlock(fluid.dygraph.Layer):
    def __init__(self,
                 num_channels,
                 num_filters,
                 stride,
                 shortcut=True):
        super(BottleneckBlock, self).__init__()

        self.conv0 = ConvBNLayer(
            num_channels=num_channels,
            num_filters=num_filters,
            filter_size=1,
            act='relu')
        self.conv1 = ConvBNLayer(
            num_channels=num_filters,
            num_filters=num_filters,
            filter_size=3,
            stride=stride,
            act='relu')
        self.conv2 = ConvBNLayer(
            num_channels=num_filters,
            num_filters=num_filters * 4,
            filter_size=1,
            act=None)

        if not shortcut:
            self.short = ConvBNLayer(
                num_channels=num_channels,
                num_filters=num_filters * 4,
                filter_size=1,
                stride=stride)

        self.shortcut = shortcut

        self._num_channels_out = num_filters * 4

    def forward(self, inputs):
        y = self.conv0(inputs)
        conv1 = self.conv1(y)
        conv2 = self.conv2(conv1)

        if self.shortcut:
            short = inputs
        else:
            short = self.short(inputs)

        y = fluid.layers.elementwise_add(x=short, y=conv2)

        layer_helper = LayerHelper(self.full_name(), act='relu')
        return layer_helper.append_activation(y)


class ResNet(fluid.dygraph.Layer):
    def __init__(self, layers=50, class_dim=102):
        super(ResNet, self).__init__()

        self.layers = layers
        supported_layers = [50, 101, 152]
        assert layers in supported_layers, \
            "supported layers are {} but input layer is {}".format(supported_layers, layers)

        if layers == 50:
            depth = [3, 4, 6, 3]
        elif layers == 101:
            depth = [3, 4, 23, 3]
        elif layers == 152:
            depth = [3, 8, 36, 3]
        num_channels = [64, 256, 512, 1024]
        num_filters = [64, 128, 256, 512]

        self.conv = ConvBNLayer(
            num_channels=3,
            num_filters=64,
            filter_size=7,
            stride=2,
            act='relu')
        self.pool2d_max = Pool2D(
            pool_size=3,
            pool_stride=2,
            pool_padding=1,
            pool_type='max')

        self.bottleneck_block_list = []
        for block in range(len(depth)):
            shortcut = False
            for i in range(depth[block]):
                bottleneck_block = self.add_sublayer(
                    'bb_%d_%d' % (block, i),
                    BottleneckBlock(
                        num_channels=num_channels[block]
                        if i == 0 else num_filters[block] * 4,
                        num_filters=num_filters[block],
                        stride=2 if i == 0 and block != 0 else 1,
                        shortcut=shortcut))
                self.bottleneck_block_list.append(bottleneck_block)
                shortcut = True

        self.pool2d_avg = Pool2D(
            pool_size=7, pool_type='avg', global_pooling=True)

        self.pool2d_avg_output = num_filters[len(num_filters) - 1] * 4 * 1 * 1

        import math
        stdv = 1.0 / math.sqrt(2048 * 1.0)

        self.out = Linear(self.pool2d_avg_output,
                      class_dim,
                      act='softmax',
                      param_attr=fluid.param_attr.ParamAttr(
                          initializer=fluid.initializer.Uniform(-stdv, stdv)))

    def forward(self, inputs):
        y = self.conv(inputs)
        y = self.pool2d_max(y)
        for bottleneck_block in self.bottleneck_block_list:
            y = bottleneck_block(y)
        y = self.pool2d_avg(y)
        y = fluid.layers.reshape(y, shape=[-1, self.pool2d_avg_output])
        y = self.out(y)
        return y




if __name__ == '__main__':
    use_gpu = False
    place = fluid.CUDAPlace(0) if use_gpu else fluid.CPUPlace()
    with fluid.dygraph.guard(place):
        model = ResNet(layers=50, class_dim=1000)




        state_dic, _ = fluid.dygraph.load_dygraph('resnet50_paddle.pdparams')

        model.set_dict(state_dic)
        count = 0
        # with open('train_params.txt', 'w') as f:
        #     for p in model.named_parameters():
        #         f.write(p[0] + '\n')

        # params_list = []
        # for k,v in model.named_parameters():
        #     if '_mean' not in k and '_variance' not in k:
        #         params_list.append(v)

        opt = fluid.optimizer.SGD(
            learning_rate=3e-4,
            # momentum=0.0,
            parameter_list=model.parameters(),
            # parameter_list=parameters,
            #regularization=fluid.regularizer.L2Decay(0)
        )

        # data = np.random.uniform(0,1, size=(1,3,32,224,224)).astype('float32')
        # np.save('data.npy',data)
        for i in range(20):
            model.train()
            data = np.load('data.npy')
            data = data[:,:,0,:,:]
            data = to_variable(data)
            label = to_variable(np.array([[1]]))
            # label.stop_gradient = True
            # data.stop_gradient = True
            y = model(data)
            y = fluid.layers.softmax(y)
            loss = fluid.layers.cross_entropy(y, label)
            loss = fluid.layers.mean(loss)
            print(f'epoch:{i} loss:{loss.numpy()}')
            np.save(f'loss_{i}.npy', loss.numpy()[0])
            # print(f'epoch:{i} y:{y_mean.numpy()}')
            loss.backward()
            opt.minimize(loss)
            model.clear_gradients()
        pass
from paddle.fluid.dygraph.learning_rate_scheduler import LearningRateDecay
import math
import warnings

from paddle.fluid import unique_name
from paddle.fluid.framework import Variable
from paddle.fluid.data_feeder import check_type
class ReduceLROnPlateau(LearningRateDecay):
    """
    :api_attr: imperative

    Reduce learning rate when ``loss`` has stopped descending. Models often benefit from reducing the learning rate
    by 2 to 10 times once model performance has no longer improvement.

    The ``loss`` is the one which has been pass into ``step`` , it must be 1-D Tensor with shape [1]. When ``loss``
    stop descending for a ``patience`` number of epochs, the learning rate will be reduced to ``learning_rate * decay_rate`` .
    (Specially, ``mode`` can also be set to ``'max`` , in this case, when ``loss`` stop ascending for a ``patience`` number
    of epochs, the learning rate will be reduced.)

    In addition, After each reduction, it will wait a ``cooldown`` number of epochs before resuming normal operation.

    Args:
        learning_rate (Variable|float|int): The initial learning rate. It can be set to python float or int number.
            If the type is Variable, it should be 1-D Tensor with shape [1], the data type can be 'float32' or 'float64'.
        mode (str, optional): ``'min'`` or ``'max'`` can be selected. Normally, it is ``'min'`` , which means that the
            learning rate will reduce when ``loss`` stops descending. Specially, if it's set to ``'max'`` ,  the learning
            rate will reduce when ``loss`` stops ascending. Default: ``'min'`` .
        decay_rate (float, optional): The Ratio that the learning rate will be reduced. ``new_lr = origin_lr * decay_rate`` .
            It should be less than 1.0. Default: 0.1.
        patience (int, optional): When ``loss`` doesn't improve for this number of epochs, learing rate will be reduced.
            Default: 10.
        verbose (bool, optional): If ``True``, prints a message to stdout for each update. Default: ``False``.
        threshold (float, optional): ``threshold`` and ``threshold_mode`` will determine the minimum change of ``loss`` .
            This make tiny changes of ``loss`` will be ignored. Default: 1e-4.
        threshold_mode (str, optional): ``'rel'`` or ``'abs'`` can be selected. In ``'rel'`` mode, the minimum change of ``loss``
            is ``last_loss * threshold`` , where ``last_loss`` is ``loss`` in last epoch. In ``'abs'`` mode, the minimum
            change of ``loss`` is ``threshold`` . Default: ``'rel'`` .
        cooldown (int, optional): The number of epochs to wait before resuming normal operation. Default: 0.
        min_lr (float, optional): The lower bound of the learning rate after reduction. Default: 0.
        eps (float, optional): Minimal decay applied to lr. If the difference between new and old lr is smaller than eps, the update is
            ignored. Default: 1e-8.
        dtype (str, optional): The data type used to create the learning rate variable. The data type can be set as
            'float32', 'float64'. Default: 'float32'.

    Returns:
        Reduced learning rate.

    Examples:

    .. code-block:: python

        import paddle.fluid as fluid
        import numpy as np

        with fluid.dygraph.guard():
            x = np.random.uniform(-1, 1, [10, 10]).astype("float32")
            linear = fluid.dygraph.Linear(10, 10)
            input = fluid.dygraph.to_variable(x)

            reduce_lr = fluid.dygraph.ReduceLROnPlateau(
                                    learning_rate = 1.0,
                                    decay_rate = 0.5,
                                    patience = 5,
                                    verbose = True,
                                    cooldown = 3)
            adam = fluid.optimizer.Adam(
                learning_rate = reduce_lr,
                parameter_list = linear.parameters())

            for epoch in range(10):
                total_loss = 0
                for bath_id in range(5):
                    out = linear(input)
                    loss = fluid.layers.reduce_mean(out)
                    total_loss += loss
                    adam.minimize(loss)

                avg_loss = total_loss/5

                # adjust learning rate according to avg_loss
                reduce_lr.step(avg_loss)
                lr = adam.current_step_lr()
                print("current avg_loss is %s, current lr is %s" % (avg_loss.numpy()[0], lr))

    """

    def __init__(self,
                 learning_rate,
                 mode='min',
                 decay_rate=0.1,
                 patience=10,
                 verbose=False,
                 threshold=1e-4,
                 threshold_mode='rel',
                 cooldown=0,
                 min_lr=0,
                 eps=1e-8,
                 dtype='float32'):
        super(ReduceLROnPlateau, self).__init__(dtype=dtype)
        mode = mode.lower()
        if mode not in ['min', 'max']:
            raise ValueError('mode ' + mode + ' is unknown!')
        self.mode = mode

        if decay_rate >= 1.0:
            raise ValueError(
                'new_lr = origin_lr * decay_rate and decay_rate should be < 1.0.'
            )
        self.decay_rate = self.create_lr_var(decay_rate)

        threshold_mode = threshold_mode.lower()
        if threshold_mode not in ['rel', 'abs']:
            raise ValueError('threshold mode ' + threshold_mode +
                             ' is unknown!')
        self.threshold_mode = threshold_mode
        check_type(learning_rate, 'learning_rate', (float, int, Variable),
                   'ReduceLROnPlateau')
        if not isinstance(learning_rate, (float, int, Variable)):
            raise TypeError(
                "The type of 'learning_rate' in 'ReduceLROnPlateau' must be 'float, int, Variable', but received %s."
                % type(learning_rate))

        self.learning_rate = learning_rate
        self.verbose = verbose
        self.patience = patience
        self.threshold = threshold
        self.threshold_mode = threshold_mode
        self.cooldown = cooldown
        self.min_lr = self.create_lr_var(min_lr)
        self.eps = eps

        self.cooldown_counter = 0
        self.best_loss = None
        self.num_bad_epochs = 0
        self.epoch_num = 0

    def _state_keys(self):
        self.keys = [
            'cooldown_counter', 'best_loss', 'num_bad_epochs', 'epoch_num',
            'learning_rate'
        ]

    def __call__(self):
        if not isinstance(self.learning_rate, Variable):
            self.learning_rate = self.create_lr_var(self.learning_rate)
        return self.learning_rate

    def step(self, loss):
        """
        It should be invoked on each epoch. Update the learning rate in optimizer according to ``loss`` .
        The new learning rate will take effect on next call to ``optimizer.minimize`` .

        Args:
            loss (Variable): A ``Variable`` that will be monitored to determine whether the learning rate will reduce.
                If it stop descending for a ``patience`` number of epochs, the learning rate will reduce. It should
                be 1-D Tensor with shape [1].
                Specially, if ``mode`` has been set to ``'max'`` ,  the learning rate will reduce when it stops ascending.
        Returns:
            None

        Examples:
            Please refer to the example of current LearningRateDecay.
        """

        # loss must be 1-D Tensor with shape [1]
        check_type(loss, 'loss', Variable, 'ReduceLROnPlateau.step')
        assert len(loss.shape) == 1 and loss.shape[0] == 1, "the loss.shape " \
                                                            "should be (1L,), but the current loss.shape is {}. Maybe that " \
                                                            "you should call fluid.layers.mean to process it first.".format(
            loss.shape)

        self.epoch_num += 1
        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1
        else:
            if self.best_loss is None or self._is_better(loss, self.best_loss):
                self.best_loss = loss
                self.num_bad_epochs = 0
            else:
                self.num_bad_epochs += 1
            print(f"num_bad_epochs:{self.num_bad_epochs} lr:{self.learning_rate}")
            if self.num_bad_epochs > self.patience:
                from paddle.fluid import layers
                self.cooldown_counter = self.cooldown
                self.num_bad_epochs = 0
                new_lr = layers.elementwise_max(self.learning_rate *
                                                self.decay_rate, self.min_lr)
                if self.learning_rate - new_lr > self.eps:
                    if self.verbose:
                        old_lr = self.learning_rate.numpy()[0] if isinstance(
                            self.learning_rate,
                            Variable) else self.learning_rate
                        print('Epoch {}: reducing learning rate from {} to {}.'.
                              format(self.epoch_num, old_lr, new_lr.numpy()[0]))
                    self.learning_rate = new_lr

    def _is_better(self, current, best):
        if self.mode == 'min' and self.threshold_mode == 'rel':
            return current < best - best * self.threshold

        elif self.mode == 'min' and self.threshold_mode == 'abs':
            return current < best - self.threshold

        elif self.mode == 'max' and self.threshold_mode == 'rel':
            return current > best + best * self.threshold

        else:
            return current > best + self.threshold

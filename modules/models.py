import functools
import tensorflow as tf
from absl import logging
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import (Input, Dense, Flatten, Dropout, Conv2D,
                                     AveragePooling2D, GlobalAveragePooling2D,
                                     ReLU)
from modules.operations import (OPS, FactorizedReduce, ReLUConvBN,
                                BatchNormalization, Identity, drop_path,
                                kernel_init, regularizer)
import modules.genotypes as genotypes


class Cell(tf.keras.layers.Layer):
    """Cell Layer"""
    def __init__(self, genotype, ch, reduction, reduction_prev, wd,
                 name='Cell', **kwargs):
        super(Cell, self).__init__(name=name, **kwargs)

        self.wd = wd

        if reduction_prev:
            self.preprocess0 = FactorizedReduce(ch, wd=wd)
        else:
            self.preprocess0 = ReLUConvBN(ch, k=1, s=1, wd=wd)
        self.preprocess1 = ReLUConvBN(ch, k=1, s=1, wd=wd)

        if reduction:
            op_names, indices = zip(*genotype.reduce)
            concat = genotype.reduce_concat
        else:
            op_names, indices = zip(*genotype.normal)
            concat = genotype.normal_concat

        self._compile(ch, op_names, indices, concat, reduction)

    def _compile(self, ch, op_names, indices, concat, reduction):
        assert len(op_names) == len(indices)
        self._steps = len(op_names) // 2
        self._concat = concat

        self._ops = []
        for name, index in zip(op_names, indices):
            strides = 2 if reduction and index < 2 else 1
            op = OPS[name](ch, strides, self.wd, True)
            self._ops.append(op)
        self._indices = indices

    def call(self, s0, s1, drop_path_prob):
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)

        states = [s0, s1]
        for step_index in range(self._steps):
            op1 = self._ops[2 * step_index]
            op2 = self._ops[2 * step_index + 1]
            h1 = op1(states[self._indices[2 * step_index]])
            h2 = op2(states[self._indices[2 * step_index + 1]])

            if drop_path_prob is not None:
                if not isinstance(op1, Identity):
                    h1 = drop_path(h1, drop_path_prob, name='drop_path_h1')
                if not isinstance(op2, Identity):
                    h2 = drop_path(h2, drop_path_prob, name='drop_path_h2')

            s = h1 + h2
            states += [s]

        return tf.concat([states[i] for i in self._concat], axis=-1)


class AuxiliaryHeadCIFAR(tf.keras.layers.Layer):
    """Auxiliary Head Cifar"""
    def __init__(self, num_classes, wd, name='AuxiliaryHeadCIFAR', **kwargs):
        super(AuxiliaryHeadCIFAR, self).__init__(name=name, **kwargs)
        self.features = Sequential([
            ReLU(),
            AveragePooling2D(5, strides=3, padding='valid'),
            Conv2D(filters=128, kernel_size=1, strides=1, padding='valid',
                   kernel_initializer=kernel_init(),
                   kernel_regularizer=regularizer(wd), use_bias=False),
            BatchNormalization(affine=True),
            ReLU(),
            Conv2D(filters=768, kernel_size=2, strides=1, padding='valid',
                   kernel_initializer=kernel_init(),
                   kernel_regularizer=regularizer(wd), use_bias=False),
            BatchNormalization(affine=True),
            ReLU()])
        self.classifier = Dense(num_classes, kernel_initializer=kernel_init(),
                                kernel_regularizer=regularizer(wd))

    def call(self, x):
        x = self.features(x)
        x = self.classifier(Flatten()(x))
        return x


def CifarModel(cfg, training=True, stem_multiplier=3, name='CifarModel'):
    """Cifar Model"""
    logging.info(f"buliding {name}...")

    input_size = cfg['input_size']
    ch_init = cfg['init_channels']
    layers = cfg['layers']
    num_cls = cfg['num_classes']
    wd = cfg['weights_decay']
    genotype = eval("genotypes.%s" % cfg['arch'])

    # define model
    inputs = Input([input_size, input_size, 3], name='input_image')
    if training:
        drop_path_prob = Input([], name='drop_prob')
    else:
        drop_path_prob = None

    ch_curr = stem_multiplier * ch_init
    s0 = s1 = Sequential([
        Conv2D(filters=ch_curr, kernel_size=3, strides=1, padding='same',
               kernel_initializer=kernel_init(),
               kernel_regularizer=regularizer(wd), use_bias=False),
        BatchNormalization(affine=True)], name='stem')(inputs)

    ch_curr = ch_init
    reduction_prev = False
    logits_aux = None
    for layer_index in range(layers):
        if layer_index in [layers // 3, 2 * layers // 3]:
            ch_curr *= 2
            reduction = True
        else:
            reduction = False

        cell = Cell(genotype, ch_curr, reduction, reduction_prev, wd,
                    name=f'Cell_{layer_index}')
        s0, s1 = s1, cell(s0, s1, drop_path_prob)

        reduction_prev = reduction

        if layer_index == 2 * layers // 3 and training:
            logits_aux = AuxiliaryHeadCIFAR(num_cls, wd=wd)(s1)

    fea = GlobalAveragePooling2D()(s1)

    logits = Dense(num_cls, kernel_initializer=kernel_init(),
                   kernel_regularizer=regularizer(wd))(Flatten()(fea))

    if training:
        return Model((inputs, drop_path_prob), (logits, logits_aux), name=name)
    else:
        return Model(inputs, logits, name=name)

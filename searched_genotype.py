from collections import namedtuple


Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')


pcdarts_cifar10_search_21 = \
    Genotype(normal=[('dil_conv_5x5', 1),
                    ('sep_conv_3x3', 0),
                    ('sep_conv_5x5', 2),
                    ('sep_conv_5x5', 1),
                    ('sep_conv_5x5', 1),
                    ('sep_conv_5x5', 2),
                    ('dil_conv_5x5', 2),
                    ('sep_conv_5x5', 3)],
            normal_concat=range(2, 6),
            reduce=[('sep_conv_5x5', 0),
                    ('max_pool_3x3', 1),
                    ('max_pool_3x3', 2),
                    ('max_pool_3x3', 0),
                    ('sep_conv_5x5', 0),
                    ('sep_conv_5x5', 1),
                    ('sep_conv_5x5', 0),
                    ('max_pool_3x3', 4)],
            reduce_concat=range(2, 6))

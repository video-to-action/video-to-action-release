import torch, abc, math, pdb, textwrap
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models as vision_models
from .base_nets import ConvBase, CoordConv2d, SpatialSoftmax, Module
from .python_utils import extract_class_init_kwargs_from_dict

class ResNet18Conv(ConvBase):
    """
    A ResNet18 block that can be used to process input images.
    """
    def __init__(
        self,
        input_channel=3,
        pretrained=False,
        input_coord_conv=False,
    ):
        """
        Args:
            input_channel (int): number of input channels for input images to the network.
                If not equal to 3, modifies first conv layer in ResNet to handle the number
                of input channels.
            pretrained (bool): if True, load pretrained weights for all ResNet layers.
            input_coord_conv (bool): if True, use a coordinate convolution for the first layer
                (a convolution where input channels are modified to encode spatial pixel location)
        """
        super(ResNet18Conv, self).__init__()
        net = vision_models.resnet18(weights=pretrained)

        if input_coord_conv:
            net.conv1 = CoordConv2d(input_channel, 64, kernel_size=7, stride=2, padding=3, bias=False)
        elif input_channel != 3:
            net.conv1 = nn.Conv2d(input_channel, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # cut the last fc layer
        self._input_coord_conv = input_coord_conv
        self._input_channel = input_channel
        self.nets = torch.nn.Sequential(*(list(net.children())[:-2]))
        # pdb.set_trace() # pool and fc layer should be removed

    def output_shape(self, input_shape):
        """
        Function to compute output shape from inputs to this module. 

        Args:
            input_shape (iterable of int): shape of input. Does not include batch dimension.
                Some modules may not need this argument, if their output does not depend 
                on the size of the input, or if they assume fixed size input.

        Returns:
            out_shape ([int]): list of integers corresponding to output shape
        """
        assert(len(input_shape) == 3)
        out_h = int(math.ceil(input_shape[1] / 32.))
        out_w = int(math.ceil(input_shape[2] / 32.))
        return [512, out_h, out_w]

    def __repr__(self):
        """Pretty print network."""
        header = '{}'.format(str(self.__class__.__name__))
        return header + '(input_channel={}, input_coord_conv={})'.format(self._input_channel, self._input_coord_conv)


class VisualCore(ConvBase):
    """
    A network block that combines a visual backbone network with optional pooling
    and linear layers.
    """
    def __init__(
        self,
        input_shape,
        backbone_class,
        backbone_kwargs,
        pool_class=None,
        pool_kwargs=None,
        flatten=True,
        feature_dimension=None,
        **kwargs,
    ):
        """
        Args:
            input_shape (tuple): shape of input (not including batch dimension)
            backbone_class (str): class name for the visual backbone network (e.g.: ResNet18)
            backbone_kwargs (dict): kwargs for the visual backbone network
            pool_class (str): class name for the visual feature pooler (optional)
                Common options are "SpatialSoftmax" and "SpatialMeanPool"
            pool_kwargs (dict): kwargs for the visual feature pooler (optional)
            flatten (bool): whether to flatten the visual feature
            feature_dimension (int): if not None, add a Linear layer to
                project output into a desired feature dimension
        """
        super(VisualCore, self).__init__()
        self.input_shape = input_shape
        self.flatten = flatten

        # add input channel dimension to visual core inputs
        backbone_kwargs["input_channel"] = input_shape[0]

        # extract only relevant kwargs for this specific backbone
        backbone_kwargs = extract_class_init_kwargs_from_dict(cls=eval(backbone_class), dic=backbone_kwargs, copy=True)

        # visual backbone
        assert isinstance(backbone_class, str)
        self.backbone = eval(backbone_class)(**backbone_kwargs)

        assert isinstance(self.backbone, ConvBase)

        feat_shape = self.backbone.output_shape(input_shape)
        net_list = [self.backbone]

        # maybe make pool net
        if pool_class is not None:
            assert isinstance(pool_class, str)
            # feed output shape of backbone to pool net
            if pool_kwargs is None:
                pool_kwargs = dict()
            # extract only relevant kwargs for this specific backbone
            pool_kwargs["input_shape"] = feat_shape
            pool_kwargs = extract_class_init_kwargs_from_dict(cls=eval(pool_class), dic=pool_kwargs, copy=True)
            self.pool = eval(pool_class)(**pool_kwargs)
            assert isinstance(self.pool, Module)

            feat_shape = self.pool.output_shape(feat_shape)
            net_list.append(self.pool)
        else:
            assert False
            self.pool = None

        assert isinstance(self.pool, SpatialSoftmax)
        # flatten layer
        if self.flatten:
            net_list.append(torch.nn.Flatten(start_dim=1, end_dim=-1))

        # maybe linear layer
        self.feature_dimension = feature_dimension
        if feature_dimension is not None:
            # assert False
            assert self.flatten
            linear = torch.nn.Linear(int(np.prod(feat_shape)), feature_dimension)
            net_list.append(linear)

        self.nets = nn.Sequential(*net_list)
        # pdb.set_trace()

    def output_shape(self, input_shape):
        """
        Function to compute output shape from inputs to this module. 

        Args:
            input_shape (iterable of int): shape of input. Does not include batch dimension.
                Some modules may not need this argument, if their output does not depend 
                on the size of the input, or if they assume fixed size input.

        Returns:
            out_shape ([int]): list of integers corresponding to output shape
        """
        if self.feature_dimension is not None:
            # linear output
            return [self.feature_dimension]
        feat_shape = self.backbone.output_shape(input_shape)
        if self.pool is not None:
            # pool output
            feat_shape = self.pool.output_shape(feat_shape)
        # backbone + flat output
        if self.flatten:
            return [np.prod(feat_shape)]
        else:
            return feat_shape

    def forward(self, inputs):
        """
        Forward pass through visual core.
        """
        ndim = len(self.input_shape)
        assert tuple(inputs.shape)[-ndim:] == tuple(self.input_shape)
        return super(VisualCore, self).forward(inputs)

    def __repr__(self):
        """Pretty print network."""
        header = '{}'.format(str(self.__class__.__name__))
        msg = ''
        indent = ' ' * 2
        msg += textwrap.indent(
            "\ninput_shape={}\noutput_shape={}".format(self.input_shape, self.output_shape(self.input_shape)), indent)
        msg += textwrap.indent("\nbackbone_net={}".format(self.backbone), indent)
        msg += textwrap.indent("\npool_net={}".format(self.pool), indent)
        msg = header + '(' + msg + '\n)'
        return msg
    





# class VisualCore(ConvBase): # nn.Module
#     '''write anf simplify by LUO'''
#     def __init__(self, 
#                  input_shape: tuple,
#                  backbone: ResNet18Conv,
#                  pool_layer: SpatialSoftmax,
#                  flatten=True,
#                  feature_dimension=None,
#                  ) -> None:
#         super(VisualCore, self).__init__()
#         self.input_shape = input_shape
#         assert type(input_shape) == tuple
#         self.backbone = backbone
#         self.pool_layer = pool_layer
#         self.flatten = flatten
#         assert isinstance(self.backbone, ConvBase)
#         net_list = [self.backbone, self.pool_layer]

#         # flatten layer
#         if self.flatten:
#             net_list.append(torch.nn.Flatten(start_dim=1, end_dim=-1))

#         feat_shape = self.backbone.output_shape(input_shape)
        

#         # maybe linear layer
#         self.feature_dimension = feature_dimension
#         if feature_dimension is not None:
#             assert False
#             assert self.flatten
#             linear = torch.nn.Linear(int(np.prod(feat_shape)), feature_dimension)
#             net_list.append(linear)

#         self.nets = nn.Sequential(*net_list)


#     def forward(self, inputs):
#         """
#         Forward pass through visual core.
#         """
#         ndim = len(self.input_shape)
#         assert tuple(inputs.shape)[-ndim:] == self.input_shape
#         return super(VisualCore, self).forward(inputs)

import torch.nn as nn
import torch
import copy
import argparse
import yaml

from fast_pytorch_kmeans import KMeans
from collections import namedtuple

from models import MyModelv7
from tinynas.nn.networks import ProxylessNASNets, MobileInvertedResidualBlock
from tinynas.nn.modules.layers import *
from collections import OrderedDict

# from train import TrainModel
from mpii_train import TrainModel

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies.ddp import DDPStrategy

Codebook = namedtuple('Codebook', ['centroids', 'labels'])

class KMeansQuantizer:
    def __init__(self, model : nn.Module, bitwidth=4):
        self.codebook = self.quantize(model, bitwidth)

    def update_codebook(self, fp32_tensor: torch.Tensor, codebook: Codebook):
        """
        update the centroids in the codebook using updated fp32_tensor
        :param fp32_tensor: [torch.(cuda.)Tensor]
        :param codebook: [Codebook] (the cluster centroids, the cluster label tensor)
        """
        n_clusters = codebook.centroids.numel()
        fp32_tensor = fp32_tensor.view(-1)
        for k in range(n_clusters):
            codebook.centroids[k] = torch.sum(fp32_tensor * (codebook.labels == k)) / torch.sum((codebook.labels == k))

    def k_means_quantize(self, fp32_tensor: torch.Tensor, bitwidth=4, codebook=None):
        """
        quantize tensor using k-means clustering
        :param fp32_tensor:
        :param bitwidth: [int] quantization bit width, default=4
        :param codebook: [Codebook] (the cluster centroids, the cluster label tensor)
        :return:
            [Codebook = (centroids, labels)]
                centroids: [torch.(cuda.)FloatTensor] the cluster centroids
                labels: [torch.(cuda.)LongTensor] cluster label tensor
        """
        if codebook is None:
            # get number of clusters based on the quantization precision
            n_clusters = pow(2, bitwidth)
            # use k-means to get the quantization centroids
            kmeans = KMeans(n_clusters=n_clusters, mode='euclidean', verbose=0)
            labels = kmeans.fit_predict(fp32_tensor.view(-1, 1)).to(torch.long)
            centroids = kmeans.centroids.to(torch.float).view(-1)
            codebook = Codebook(centroids, labels)

        # decode the codebook into k-means quantized tensor for inference
        quantized_tensor = codebook.centroids[codebook.labels]
        fp32_tensor.set_(quantized_tensor.view_as(fp32_tensor))
        return codebook

    @torch.no_grad()
    def apply(self, model, update_centroids):
        for name, param in model.named_parameters():
            if name in self.codebook:
                if update_centroids:
                    self.update_codebook(param, codebook=self.codebook[name])
                self.codebook[name] = self.k_means_quantize(
                    param, codebook=self.codebook[name])

    @torch.no_grad()
    def quantize(self, model: nn.Module, bitwidth=4):
        print("quantizing model...")
        codebook = dict()
        if isinstance(bitwidth, dict):
            for name, param in model.named_parameters():
                if name in bitwidth:
                    codebook[name] = self.k_means_quantize(param, bitwidth=bitwidth[name])
        else:
            for name, param in model.named_parameters():
                if param.dim() > 1:
                    codebook[name] = self.k_means_quantize(param, bitwidth=bitwidth)
        return codebook

def get_quantized_range(bitwidth):
    quantized_max = (1 << (bitwidth - 1)) - 1
    quantized_min = -(1 << (bitwidth - 1))
    return quantized_min, quantized_max

class LinearQuantizer:
    def __init__(self, model: MyModelv7, bitwidth):
        self.model = model
        self.bitwidth = bitwidth
        self.model_fused = None
        self.quantized_model = None
    
    def linear_quantize(self, fp_tensor, bitwidth, scale, zero_point, dtype=torch.int8) -> torch.Tensor:
        assert(fp_tensor.dtype == torch.float)
        assert(isinstance(scale, float) or
            (scale.dtype == torch.float and scale.dim() == fp_tensor.dim()))
        assert(isinstance(zero_point, int) or
            (zero_point.dtype == dtype and zero_point.dim() == fp_tensor.dim()))

        # Step 1: scale the fp_tensor
        scaled_tensor = fp_tensor / scale
        # Step 2: round the floating value to integer value
        rounded_tensor = torch.round(scaled_tensor)

        rounded_tensor = rounded_tensor.to(dtype)

        # Step 3: shift the rounded_tensor to make zero_point 0
        shifted_tensor = rounded_tensor + zero_point

        # Step 4: clamp the shifted_tensor to lie in bitwidth-bit range
        quantized_min, quantized_max = get_quantized_range(bitwidth)
        quantized_tensor = shifted_tensor.clamp_(quantized_min, quantized_max)
        return quantized_tensor

    def get_quantization_scale_and_zero_point(self, fp_tensor, bitwidth):
        quantized_min, quantized_max = get_quantized_range(bitwidth)
        fp_max = fp_tensor.max().item()
        fp_min = fp_tensor.min().item()

        scale = (fp_max - fp_min)/ (quantized_max - quantized_min)
        zero_point = int(round(quantized_min - fp_min / scale))

        # clip the zero_point to fall in [quantized_min, quantized_max]
        if zero_point < quantized_min:
            zero_point = quantized_min
        elif zero_point > quantized_max:
            zero_point = quantized_max
        else: # convert from float to int using round()
            zero_point = round(zero_point)
        return scale, int(zero_point)

    def linear_quantize_feature(self, fp_tensor, bitwidth):
        scale, zero_point = self.get_quantization_scale_and_zero_point(fp_tensor, bitwidth)
        quantized_tensor = self.linear_quantize(fp_tensor, bitwidth, scale, zero_point)
        return quantized_tensor, scale, zero_point
    
    def get_quantization_scale_for_weight(self, weight, bitwidth):
        # we just assume values in weight are symmetric
        # we also always make zero_point 0 for weight
        fp_max = max(weight.abs().max().item(), 5e-7)
        _, quantized_max = get_quantized_range(bitwidth)
        return fp_max / quantized_max
    
    def linear_quantize_weight_per_channel(self, tensor, bitwidth):
        dim_output_channels = 0
        num_output_channels = tensor.shape[dim_output_channels]
        scale = torch.zeros(num_output_channels, device=tensor.device)
        for oc in range(num_output_channels):
            _subtensor = tensor.select(dim_output_channels, oc)
            _scale = self.get_quantization_scale_for_weight(_subtensor, bitwidth)
            scale[oc] = _scale
        scale_shape = [1] * tensor.dim()
        scale_shape[dim_output_channels] = -1
        scale = scale.view(scale_shape)
        quantized_tensor = self.linear_quantize(tensor, bitwidth, scale, zero_point=0)
        return quantized_tensor, scale, 0

    def linear_quantize_bias_per_output_channel(self, bias, weight_scale, input_scale):
        assert(bias.dim() == 1)
        assert(bias.dtype == torch.float)
        assert(isinstance(input_scale, float))
        if isinstance(weight_scale, torch.Tensor):
            assert(weight_scale.dtype == torch.float)
            weight_scale = weight_scale.view(-1)
            assert(bias.numel() == weight_scale.numel())

        bias_scale = weight_scale * input_scale
        quantized_bias = self.linear_quantize(bias, 32, bias_scale,
                                        zero_point=0, dtype=torch.int32)
        return quantized_bias, bias_scale, 0

    def shift_quantized_conv2d_bias(self, quantized_bias, quantized_weight, input_zero_point):
        assert(quantized_bias.dtype == torch.int32)
        assert(isinstance(input_zero_point, int))
        return quantized_bias - quantized_weight.sum((1,2,3)).to(torch.int32) * input_zero_point

    def get_quantized_conv(self, input_activation, output_activation, conv_name, relu_name, feature_bitwidth, conv):
        weight_bitwidth = feature_bitwidth
        input_scale, input_zero_point = self.get_quantization_scale_and_zero_point(input_activation[conv_name], feature_bitwidth)

        output_scale, output_zero_point = self.get_quantization_scale_and_zero_point(output_activation[relu_name], feature_bitwidth)

        quantized_weight, weight_scale, weight_zero_point = self.linear_quantize_weight_per_channel(conv.weight.data, weight_bitwidth)
        quantized_bias, bias_scale, bias_zero_point = self.linear_quantize_bias_per_output_channel(conv.bias.data, weight_scale, input_scale)
        shifted_quantized_bias = self.shift_quantized_conv2d_bias(quantized_bias, quantized_weight, input_zero_point)

        quantized_conv = QuantizedConv2d(
            quantized_weight, shifted_quantized_bias,
            input_zero_point, output_zero_point,
            input_scale, weight_scale, output_scale,
            conv.stride, conv.padding, conv.dilation, conv.groups,
            feature_bitwidth=feature_bitwidth, weight_bitwidth=weight_bitwidth
        )

        return quantized_conv

    def fuse_conv_bn(self, conv, bn):
        # modified from https://mmcv.readthedocs.io/en/latest/_modules/mmcv/cnn/utils/fuse_conv_bn.html
        assert conv.bias is None

        factor = bn.weight.data / torch.sqrt(bn.running_var.data + bn.eps)
        conv.weight.data = conv.weight.data * factor.reshape(-1, 1, 1, 1)
        conv.bias = nn.Parameter(- bn.running_mean.data * factor + bn.bias.data)

        return conv
    
    def fuse_layer(self, layer: ConvLayer):
        conv = layer.conv
        bn = layer.bn

        new_conv = self.fuse_conv_bn(conv, bn)
        layer.conv = new_conv
        del layer._modules["bn"]
    
    def fuse_sequential(self, seq: nn.Sequential):

        conv = seq.conv
        bn = seq.bn
        act = None
        try: 
            act = seq.act
        except:
            pass

        new_conv = self.fuse_conv_bn(conv, bn) 

        if act is not None:
            new_seq = nn.Sequential(OrderedDict([
                ('conv', new_conv),
                ('act', act)
            ]))
        else:
            new_seq = nn.Sequential(OrderedDict([
                ('conv', new_conv),
            ]))
        
        return new_seq

    def fuse_model(self):
        model_fused = copy.deepcopy(self.model)
        eye_channel = model_fused.eye_channel
        attention_branch = model_fused.attention_branch
        face_channel = model_fused.face_channel[0]

        for i, backbone in enumerate([eye_channel, attention_branch, face_channel]):
            assert isinstance(backbone, ProxylessNASNets)

            self.fuse_layer(backbone.first_conv)
            self.fuse_layer(backbone.feature_mix_layer)
            blocks = []

            for i, block in enumerate(backbone.blocks):
                assert isinstance(block, MobileInvertedResidualBlock)
                mic = block.mobile_inverted_conv
                if isinstance(mic, ZeroLayer):
                    continue

                mic.depth_conv = self.fuse_sequential(mic.depth_conv)
                mic.point_linear = self.fuse_sequential(mic.point_linear)

                if mic.inverted_bottleneck:
                    mic.inverted_bottleneck = self.fuse_sequential(mic.inverted_bottleneck)
            
        self.model_fused = model_fused
        
    def quantize_model(self, input_activation=None, output_activation=None):
        feature_bitwidth = self.bitwidth

        model_fused_copy = copy.deepcopy(self.model_fused)
        def add_name(name, add):
            for i in add:
                name += f".{i}"
            
            return name

        eye_channel = model_fused_copy.eye_channel
        attention_branch = model_fused_copy.attention_branch
        face_channel = model_fused_copy.face_channel[0]
        # eye_channel, attention_branch, _face_backbone 
        for i, backbone in enumerate([eye_channel, attention_branch, face_channel]):
            if i == 0: 
                print("eye channel")
                name = "eye_channel"
            elif i == 1: 
                print("attention_branch")
                name = "attention_branch"
            else:
                print("face channel")
                name = "face_channel.0"

            assert isinstance(backbone, ProxylessNASNets)
            first_conv = backbone.first_conv
            fcnc = add_name(name, ["first_conv", "conv"])
            fcnr = add_name(name, ["first_conv", "act"])

            backbone.first_conv = self.get_quantized_conv(input_activation, output_activation, fcnc, fcnr, feature_bitwidth, first_conv.conv)

            blocks = backbone.blocks
            feature_mix_layer = backbone.feature_mix_layer
            fmlnc = add_name(name, ["feature_mix_layer", "conv"])
            fmlnr = add_name(name, ["feature_mix_layer", "act"])

            backbone.feature_mix_layer = self.get_quantized_conv(input_activation, output_activation, fmlnc, fmlnr, feature_bitwidth, feature_mix_layer.conv)
            
            blocks_copy = []
            print("============")
            for i, block in enumerate(blocks): 
                print(f"Modifying block {i} of {len(blocks)}")
                
                assert isinstance(block, MobileInvertedResidualBlock)
                block_copy = copy.copy(block)

                mic = block.mobile_inverted_conv
                if isinstance(mic, ZeroLayer):
                    continue
                
                add = ["blocks", i, "mobile_inverted_conv"]

                # inverted_bottleneck = mic.inverted_bottleneck.conv
                depth_conv = mic.depth_conv
                dcnc = add_name(name, [*add, "depth_conv", "conv"])
                dcnr = add_name(name, [*add, "depth_conv", "act"])

                point_linear =  mic.point_linear
                plnc = add_name(name, [*add, "point_linear", "conv"])
                plnr = add_name(name, [*add, "point_linear", "act"])

                inverted_bottleneck = mic.inverted_bottleneck
                ibnc = add_name(name, [*add, "inverted_bottleneck", "conv"])
                ibnr = add_name(name, [*add, "inverted_bottleneck", "act"])

                mic.depth_conv = self.get_quantized_conv(input_activation, output_activation, dcnc, dcnr, feature_bitwidth, depth_conv.conv)
                mic.point_linear = self.get_quantized_conv(input_activation, output_activation, plnc, dcnr, feature_bitwidth, point_linear.conv)

                if inverted_bottleneck is not None:
                    mic.inverted_bottleneck = self.get_quantized_conv(input_activation, output_activation, ibnc, ibnr, feature_bitwidth, inverted_bottleneck.conv)


            # print(backbone)
        self.quantized_model = model_fused_copy
        

class QuantizedConv2d(nn.Module):
    def __init__(self, weight, bias,
                 input_zero_point, output_zero_point,
                 input_scale, weight_scale, output_scale,
                 stride, padding, dilation, groups,
                 feature_bitwidth=8, weight_bitwidth=8):
        super().__init__()
        # current version Pytorch does not support IntTensor as nn.Parameter
        self.register_buffer('weight', weight)
        self.register_buffer('bias', bias)

        self.input_zero_point = input_zero_point
        self.output_zero_point = output_zero_point

        self.input_scale = input_scale
        self.register_buffer('weight_scale', weight_scale)
        self.output_scale = output_scale

        self.stride = stride
        self.padding = (padding[1], padding[1], padding[0], padding[0])
        self.dilation = dilation
        self.groups = groups

        self.feature_bitwidth = feature_bitwidth
        self.weight_bitwidth = weight_bitwidth


    def quantized_conv2d(self, input, weight, bias, feature_bitwidth, weight_bitwidth,
                        input_zero_point, output_zero_point,
                        input_scale, weight_scale, output_scale,
                        stride, padding, dilation, groups):
        print("-----------")
        assert(len(padding) == 4)
        print(weight.dtype, input.dtype)
        print(bias.dtype if bias is not None else "")


        assert(weight.dtype == input.dtype)
        assert(bias is None or bias.dtype == torch.int32)
        assert(isinstance(input_zero_point, int))
        assert(isinstance(output_zero_point, int))
        assert(isinstance(input_scale, float))
        assert(isinstance(output_scale, float))
        assert(weight_scale.dtype == torch.float)

        # Step 1: calculate integer-based 2d convolution (8-bit multiplication with 32-bit accumulation)
        input = torch.nn.functional.pad(input, padding, 'constant', input_zero_point)
        print("input size: ", input.shape)
        print("weight size: ", weight.shape)
        if 'cpu' in input.device.type:
            # use 32-b MAC for simplicity
            output = torch.nn.functional.conv2d(input.to(torch.int32), weight.to(torch.int32), None, stride, 0, dilation, groups)
        else:
            # current version pytorch does not yet support integer-based conv2d() on GPUs
            output = torch.nn.functional.conv2d(input.float(), weight.float(), None, stride, 0, dilation, groups)
            output = output.round().to(torch.int32)
        if bias is not None:
            output = output + bias.view(1, -1, 1, 1)


        output = (output.to(torch.float32).swapaxes(0, 1) * (input_scale * weight_scale / output_scale)).swapaxes(0, 1)
        output = output + output_zero_point

        # Make sure all value lies in the bitwidth-bit range
        output = output.round().clamp(*get_quantized_range(feature_bitwidth)).to(torch.int8)
        print("output size: ", output.shape)
        return output

    def forward(self, x):
        return self.quantized_conv2d(
            x, self.weight, self.bias,
            self.feature_bitwidth, self.weight_bitwidth,
            self.input_zero_point, self.output_zero_point,
            self.input_scale, self.weight_scale, self.output_scale,
            self.stride, self.padding, self.dilation, self.groups
            )

class QuantizedLinear(nn.Module):
    def __init__(self, weight, bias,
                 input_zero_point, output_zero_point,
                 input_scale, weight_scale, output_scale,
                 feature_bitwidth=8, weight_bitwidth=8):
        super().__init__()
        # current version Pytorch does not support IntTensor as nn.Parameter
        self.register_buffer('weight', weight)
        self.register_buffer('bias', bias)

        self.input_zero_point = input_zero_point
        self.output_zero_point = output_zero_point

        self.input_scale = input_scale
        self.register_buffer('weight_scale', weight_scale)
        self.output_scale = output_scale

        self.feature_bitwidth = feature_bitwidth
        self.weight_bitwidth = weight_bitwidth
    
    def quantized_linear(self, input, weight, bias, feature_bitwidth, weight_bitwidth,
                     input_zero_point, output_zero_point,
                     input_scale, weight_scale, output_scale):
        assert(input.dtype == torch.int8)
        assert(weight.dtype == input.dtype)
        assert(bias is None or bias.dtype == torch.int32)
        assert(isinstance(input_zero_point, int))
        assert(isinstance(output_zero_point, int))
        assert(isinstance(input_scale, float))
        assert(isinstance(output_scale, float))
        assert(weight_scale.dtype == torch.float)

        # Step 1: integer-based fully-connected (8-bit multiplication with 32-bit accumulation)
        if 'cpu' in input.device.type:
            # use 32-b MAC for simplicity
            output = torch.nn.functional.linear(input.to(torch.int32), weight.to(torch.int32), bias)
        else:
            # current version pytorch does not yet support integer-based linear() on GPUs
            output = torch.nn.functional.linear(input.float(), weight.float(), bias.float())

        ############### YOUR CODE STARTS HERE ###############
        # Step 2: scale the output
        #         hint: 1. scales are floating numbers, we need to convert output to float as well
        #               2. the shape of weight scale is [oc, 1, 1, 1] while the shape of output is [batch_size, oc]
        # weight_scale = weight_scale.repeat(output.size()[0])
        weight_scale = torch.transpose(weight_scale, -1, -2)
        output = (output.to(torch.float32)) * (input_scale * weight_scale / output_scale)

        # Step 3: shift output by output_zero_point
        #         hint: one line of code
        output = output + output_zero_point
        ############### YOUR CODE ENDS HERE #################

        # Make sure all value lies in the bitwidth-bit range
        output = output.round().clamp(*get_quantized_range(feature_bitwidth)).to(torch.int8)
        return output

    def forward(self, x):
        return self.quantized_linear(
            x, self.weight, self.bias,
            self.feature_bitwidth, self.weight_bitwidth,
            self.input_zero_point, self.output_zero_point,
            self.input_scale, self.weight_scale, self.output_scale
            )

def get_children(model: torch.nn.Module):
    # get children form model!
    children = list(model.children())
    flatt_children = []
    if children == []:
        # if model has no children; model is last child! :O
        return model
    else:
       # look for children from children... to the last child!
       for child in children:
            try:
                flatt_children.extend(get_children(child))
            except TypeError:
                flatt_children.append(get_children(child))
    return flatt_children

def get_model_size(model: nn.Module, data_width=32):
    """
    calculate the model size in bits
    :param data_width: #bits per element
    """
    num_elements = 0
    for param in model.parameters():
        num_elements += param.numel()
    return num_elements * data_width

def get_activations(args, model):
    trainmodel = TrainModel(args=args, model=model)
    train_dataloader = trainmodel.train_dataloader()
    print(train_dataloader)
    input_activation = {}
    output_activation = {}

    def add_range_recoder_hook(model):
        import functools
        def _record_range(self, x, y, module_name):
            x = x[0]
            input_activation[module_name] = x.detach()
            output_activation[module_name] = y.detach()

        all_hooks = []
        for name, m in model.named_modules():
            if isinstance(m, (nn.Conv2d, nn.Linear, nn.ReLU, nn.ReLU6)):
                all_hooks.append(m.register_forward_hook(
                    functools.partial(_record_range, module_name=name)))
        return all_hooks

    hooks = add_range_recoder_hook(model)
    sample_data = iter(train_dataloader).__next__()
    left, right, face, gaze = sample_data
    print(left.shape, right.shape, face.shape)
    left = left.expand(left.shape[0], 3, left.shape[2], left.shape[3])
    right = right.expand(right.shape[0], 3, right.shape[2], right.shape[3])
    face = face.expand(face.shape[0], 3, face.shape[2], face.shape[3])

    model(left.cuda(), right.cuda(), face.cuda())

    # # remove hooks
    for h in hooks:
        h.remove()
    
    return input_activation, output_activation
if __name__ == "__main__": 
    model = MyModelv7(arch="proxyless-w0.3-r176_imagenet").cuda()
    print(model)

    # print(model.face_channel[0].blocks[2].mobile_inverted_conv.depth_conv.conv)
    # print(model.face_channel[0].blocks[17].mobile_inverted_conv)
    # print(model.face_channel[2])
    quantized_model = copy.deepcopy(model)
    # model_size = get_model_size(model)
    km = KMeansQuantizer(model=quantized_model, bitwidth=8)
    # quantized_model_size = get_model_size(quantized_model, 8)
    # for m in model.named_parameters():
    #     print(m[0])

    # conv = []
    # for name, m in model.named_modules():
    #     if isinstance(m, (nn.Conv2d, nn.Linear, nn.ReLU, nn.ReLU6)):
    #         print(name)

    # print(conv)
    # print("fusing model...")
    # l = LinearQuantizer(model, 8)
    # l.fuse_model()
    # print(l.model_fused)
    # l.quantize_model()
    # l.quantize_model(name_list=conv)
    # children = get_children(model)

    pl.seed_everything(47)
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=False, type=str, default="./configs/config.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        yaml_args = yaml.load(f, Loader=yaml.FullLoader)
    yaml_args.update(vars(args))
    args = argparse.Namespace(**yaml_args)

    print("getting activations...")
    # input_activation, output_activation = get_activations(args, l.model_fused.cuda())
    # print(input_activation.keys())
    # print(output_activation.keys())
    print("quantizing model...")
    # l.quantize_model(input_activation, output_activation)

    # print(l.quantized_model)

    # model = TrainModel(args, model=l.quantized_model)
    model = TrainModel(args=args, model=quantized_model, kmeans_quantizer=km)
    mylogger = WandbLogger(project=args.project, 
                                log_model=False, 
                                name=args.run_name,
                                id=args.run_name)
    mylogger.log_hyperparams(args)
    mylogger.watch(model, None, 10000, log_graph=False)

    checkpoint_callback = ModelCheckpoint(dirpath=args.ckpt_dir,
                                        filename='{epoch}-{val_loss:.4f}-{val_angle_error:.2f}',
                                        monitor='val_angle_error',
                                        save_last=True,
                                        save_top_k=3,
                                        verbose=False)
    

    print("Starting training...")
    trainer = Trainer(default_root_dir=args.ckpt_dir,
                    devices=1,
                    precision=32,
                    callbacks=[checkpoint_callback],
                    max_epochs=args.epoch,
                    benchmark=True,
                    strategy="ddp_find_unused_parameters_true",
                    logger=mylogger
                    )
    trainer.fit(model)
    trainer.validate(model)
    
    # for k, v in model.state_dict().items():
    #     print(k)
    # print(f"Original size: {model_size}")
    # print(f"Quantized Model Size: {quantized_model_size}")
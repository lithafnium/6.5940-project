import torch.nn as nn
import torch
import copy
from fast_pytorch_kmeans import KMeans
from collections import namedtuple

from models import MyModelv7

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
    def __init__(self, model, bitwidth):
        self.model = model
        self.bitwidth = bitwidth
    
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

    def get_quantization_scale_and_zero_point(fp_tensor, bitwidth):
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
        assert(len(padding) == 4)
        assert(input.dtype == torch.int8)
        assert(weight.dtype == input.dtype)
        assert(bias is None or bias.dtype == torch.int32)
        assert(isinstance(input_zero_point, int))
        assert(isinstance(output_zero_point, int))
        assert(isinstance(input_scale, float))
        assert(isinstance(output_scale, float))
        assert(weight_scale.dtype == torch.float)

        # Step 1: calculate integer-based 2d convolution (8-bit multiplication with 32-bit accumulation)
        input = torch.nn.functional.pad(input, padding, 'constant', input_zero_point)
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

if __name__ == "__main__": 
    model = MyModelv7(arch="proxyless-w0.3-r176_imagenet")

    # print(model.face_channel[0].blocks[2].mobile_inverted_conv.depth_conv.conv)
    # print(model.face_channel[0].blocks[17].mobile_inverted_conv)
    # print(model.face_channel[2])
    # quantized_model = copy.deepcopy(model)
    # model_size = get_model_size(model)
    # km = KMeansQuantizer(model=quantized_model, bitwidth=8)
    # quantized_model_size = get_model_size(quantized_model, 8)
    # for m in model.named_parameters():
    #     print(m[0])

    children = get_children(model)
    # for k, v in model.state_dict().items():
    #     print(k)
    # print(f"Original size: {model_size}")
    # print(f"Quantized Model Size: {quantized_model_size}")
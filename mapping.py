class OperationMapping:
    @staticmethod
    def relu_map(node=None, in_shape=None, out_shape=None):
        return 'nn.ReLU()'

    @staticmethod
    def leakyrelu_map(node=None, in_shape=None, out_shape=None):
        return 'nn.LeakyReLU()'

    @staticmethod
    def sigmoid_map(node, in_shape=None, out_shape=None):
        return 'nn.Sigmoid()'

    @staticmethod
    def maxpool_map(node, in_shape=None, out_shape=None):
        result = f'nn.MaxPool2d('
        parameters = {
            'kernel_size': node.get('kernel_shape'),
            'stride': node.get('strides'),
            'padding': node['pads'][2:] if node.get('pads') else None,
            'dilation': node.get('dilations'),
        }
        is_first = True
        for param, value in parameters.items():
            if value:
                if not is_first:
                    result += ', '
                result += f'{param}={value}'
                is_first = False
        return result + ')'

    @staticmethod
    def flatten_map(node=None, in_shape=None, out_shape=None):
        return 'nn.Flatten()'

    @staticmethod
    def avgpool_map(node, in_shape=None, out_shape=None):
        # TODO: Global avgpool https://github.com/onnx/onnx/blob/main/docs/Operators.md#GlobalAveragePool
        return f"nn.AdaptiveAvgPool2d({(node['output_shape'][2], node['output_shape'][3])})"

    @staticmethod
    def batchnorm_map(node, in_shape=None, out_shape=None):
        result = f'nn.BatchNorm2d(num_features={in_shape}'
        parameters = {
            'eps': node.get('epsilon'),
            # 'momentum': node.get('momentum'),
        }
        for param, value in parameters.items():
            if value:
                result += f', {param}={value}'
        return result + ')'

    @staticmethod
    def tanh_map(node=None, in_shape=None, out_shape=None):
        return 'nn.Tanh()'

    @staticmethod
    def pad_map(node, in_shape=None, out_shape=None):
        # TODO: F.pad https://github.com/onnx/onnx/blob/main/docs/Operators.md#Pad
        return f"#  Unsupportable layer type: {node['op']}"

    @staticmethod
    def reducemean_map(node, in_shape=None, out_shape=None):
        # TODO: numpy.mean https://github.com/onnx/onnx/blob/main/docs/Operators.md#ReduceMean
        return f"#  Unsupportable layer type: {node['op']}"

    @staticmethod
    def add_map(node, in_shape=None, out_shape=None):
        return f"#  Unsupportable layer type: {node['op']}"

    @staticmethod
    def concat_map(node, in_shape=None, out_shape=None):
        return f"#  Unsupportable layer type: {node['op']}"


class LayerMapping:
    @staticmethod
    def linear_map(node, in_feature, out_feature):
        return f'nn.Linear(in_features={in_feature}, out_features={out_feature})'

    @staticmethod
    def conv_map(node, in_feature, out_feature):
        result = f'nn.Conv2d(in_channels={in_feature}, out_channels={out_feature}'
        parameters = {
            'kernel_size': tuple(node['kernel_shape']) if node.get('kernel_shape') else None,
            'stride': tuple(node['strides']) if node.get('strides') else None,
            'padding': tuple(node['pads'][2:]) if node.get('pads') else None,
            'dilation': tuple(node['dilations']) if node.get('dilations') else None,
            'groups': node.get('group'),
        }
        for param, value in parameters.items():
            if value:
                result += f', {param}={value}'
        return result + ')'


class NetworkMapping:
    # Layers:
    __name_to_layer = {
        "Linear": LayerMapping.linear_map,
        "Conv": LayerMapping.conv_map
    }
    # Operations:
    __name_to_operation = {
        "Relu": OperationMapping.relu_map,
        "MaxPool": OperationMapping.maxpool_map,
        "AveragePool": OperationMapping.avgpool_map,
        "GlobalAveragePool": OperationMapping.avgpool_map,
        "Flatten": OperationMapping.flatten_map,
        "LeakyRelu": OperationMapping.leakyrelu_map,
        "Sigmoid": OperationMapping.sigmoid_map,
        "BatchNorm": OperationMapping.batchnorm_map,
        "Pad": OperationMapping.pad_map,
        "ReduceMean": OperationMapping.reducemean_map,
        "Tanh": OperationMapping.tanh_map,
    }

    @staticmethod
    def map_node(node, in_shape=None, out_shape=None):
        if node['op'] in NetworkMapping.__name_to_layer:
            return NetworkMapping.__name_to_layer[node['op']](node, in_shape, out_shape)
        if node['op'] in NetworkMapping.__name_to_operation:
            return NetworkMapping.__name_to_operation[node['op']](node, in_shape, out_shape)
        return f"#  Unsupportable layer type: {node['op']}"
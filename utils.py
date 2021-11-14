import torch
import torch.nn as nn
import torch.optim as optim
# import torch.nn.functional as F
import torchvision.models as models
# from tqdm import tqdm
import numpy as np

from efficientnet_pytorch import EfficientNet
import json
import yaml
import wandb

def get_network(
    options
):
    model = None

    # ResNet18
    if options.network == "ResNet_P":
        model = models.resnet18(pretrained=True)  # pretrained=options.model.pretrained
        model.fc = nn.Linear(512, options.data.num_classes)
    elif options.network == "ResNet_NP":
        model = models.resnet18(pretrained=False)
        model.fc = nn.Linear(512, options.data.num_classes)
    # EfficientNet-b0
    elif options.network == "EfficientNet_P":
        model = EfficientNet.from_pretrained('efficientnet-b0')
        model._fc = nn.Linear(1280, options.data.num_classes)
    elif options.network == "EfficientNet_NP":
        model = EfficientNet.from_name('efficientnet-b0')
        model._fc = nn.Linear(1280, options.data.num_classes)
    # EfficientNet-b4
    elif options.network == "EfficientNet-b4_P":
        model = EfficientNet.from_pretrained('efficientnet-b4')
        model._fc = nn.Linear(1792, options.data.num_classes)
    elif options.network == "EfficientNet-b4_NP":
        model = EfficientNet.from_name('efficientnet-b4')
        model._fc = nn.Linear(1792, options.data.num_classes)
    # MobileNet-v2
    elif options.network == "MobileNet-V2_P":
        model = models.mobilenet_v2(pretrained=True)
        model.classifier[1] = nn.Linear(1280, options.data.num_classes)
    elif options.network == "MobileNet-V2_NP":
        model = models.mobilenet_v2(pretrained=False)
        model.classifier[1] = nn.Linear(1280, options.data.num_classes)
    # DenseNet121
    elif options.network == "DenseNet_P":
        model = models.densenet121(pretrained=True)
        model.classifier = nn.Linear(1024, options.data.num_classes)
    elif options.network == "DenseNet_NP":
        model = models.densenet121(pretrained=False)
        model.classifier = nn.Linear(1024, options.data.num_classes)
    # VGGNet16
    elif options.network == "VGGNet_P":
        model = models.vgg16(pretrained=True)
        model.classifier[6] = nn.Linear(4096, options.data.num_classes)
    elif options.network == "VGGNet_NP":
        model = models.vgg16(pretrained=False)
        model.classifier[6] = nn.Linear(4096, options.data.num_classes)

    else:
        raise NotImplementedError

    return model.to(options.device)

def get_optimizer(
    params,
    options
):
    if options.optimizer.type == "Adam":
        optimizer = optim.Adam(params, lr=float(options.optimizer.lr), weight_decay=float(options.optimizer.weight_decay))
    else:
        raise NotImplementedError

    return optimizer

def guarantee_numpy(data):
    data_type = type(data)
    if data_type == torch.Tensor:
        device = data.device.type
        if device == 'cpu':
            data = data.numpy()
        else:
            data = data.data.cpu().numpy()
        return data
    elif data_type == np.ndarray or data_type == list:
        return data
    else:
        raise ValueError("Check your data type.")

def write_json(path, data):
    with open(path, "w") as json_file:
        json.dump(data, json_file)

class AttrDict(dict):
    def __init__(self, *config, **kwconfig):
        super(AttrDict, self).__init__(*config, **kwconfig)
        self.__dict__ = self
        for key in self:
            if type(self[key]) == dict:
                self[key] = AttrDict(self[key])

    def __getattr__(self, item):
        return None

    def get_values(self, keys):
        return {key: self.get(key) for key in keys}

def read_yaml(path):
    with open(path) as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
    return data

def initialize_wandb(config_path, options, note):
    config = read_yaml(config_path)
    if config['use_wandb']:
        wandb.login(key=config['key'])
        run = wandb.init(
            project=config['project'],
            config=options,
            # notes=config['notes']
            notes=note
        )

        return run
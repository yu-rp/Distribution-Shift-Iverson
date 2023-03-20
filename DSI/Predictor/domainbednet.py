import random, numpy, os, torch, logging

from .domainbed_part import algorithms

from ..utils import *

logger = logging.getLogger(module_structure(__file__))

class DomainbedNet:
    def __init__(self, *args, **kwargs):
        self.checkpointpath = kwargs["checkpointpath"]
        logger.info(f"domain bed net checkpoint {self.checkpointpath} is loading")
        self.checkpoint = torch.load(self.checkpointpath,map_location="cpu")
        logger.info(f"domain bed net checkpoint {self.checkpointpath} loaded")

    def build_model(self, *args, **kwargs):
        algorithm_class = algorithms.get_algorithm_class(self.checkpoint["args"]['algorithm'])
        hparams = self.checkpoint["model_hparams"]
        input_shape = self.checkpoint['model_input_shape']
        num_classes =  self.checkpoint['model_num_classes']
        num_domains =  self.checkpoint['model_num_domains']
        algorithm = algorithm_class(input_shape, num_classes, num_domains, hparams)
        algorithm = algorithm.eval()
        self.model = algorithm

    def modify_output(self, *args, **kwargs):
        logger.info(f"domain bed net has no [modify_output] function implementation")

    def load_model(self, *args, **kwargs):
        if self.checkpoint["args"]['algorithm'] == "GroupDRO":
            self.model.q.data = self.checkpoint['model_dict']["q"].data
        self.model.load_state_dict(self.checkpoint['model_dict'])
        self.model.eval()

    def freeze_model(self):
        for param in self.model.parameters():
            param.requires_grad = False

    def train(self, *args, **kwargs):
        logger.info(f"domain bed net has no [training] function implementation")

    def save_model(self, *args, **kwargs):
        logger.info(f"domain bed net has no[ save model] function implementation")

    def __call__(self,x):
        device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
        x = x.to(device)
        self.model.to(device)
        return self.model.predict(x)
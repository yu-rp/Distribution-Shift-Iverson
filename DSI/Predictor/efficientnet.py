import random, numpy, os, torch, logging
import torch.nn.functional as functional
import torch.optim as optim
import torchvision.utils
import torch.cuda as cuda
import torch.distributed as distributed
import torch.multiprocessing as multiprocessing
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.optim import ZeroRedundancyOptimizer as ZeRO
from torch.utils.data.distributed import DistributedSampler as Sampler
import argparse
from torch.utils.data import DataLoader

from ..Data.utils import *
from ..Data import datasets

from ..metrics import *
from ..utils import *
from .utils import *

logger = logging.getLogger(module_structure(__file__))

class EfficientNet:
    def __init__(self, version = "b0", **kwargs):
        assert version in ["b0","b4","widese_b0","widese_b4"] # here w is widese
        self.model_name = f'nvidia_efficientnet_{version}'

    def build_model(self):
        model = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', self.model_name , pretrained=False)
        model = model.eval()
        self.model = model

    def load_model(self, path = None):
        if path is None:
            self.model = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', self.model_name, pretrained=True)
        else:
            self.model.load_state_dict(torch.load(path,map_location="cpu"))
        self.model.eval()
        

    def save_model(self, path):
        if isinstance(self.model, DDP):
            torch.save(self.model.module.state_dict(), path)
        else:
            torch.save(self.model.state_dict(), path)

    def modify_output(self, num_categories):
        in_features = self.model.classifier.fc.in_features
        self.model.classifier.fc = torch.nn.Linear(in_features, num_categories)

    def freeze_model(self):
        for param in self.model.parameters():
            param.requires_grad = False
        for param in self.model.classifier.parameters():
            param.requires_grad = True

    def model_gpu_memory_report(self):
        mem_params = sum([param.nelement()*param.element_size() for param in self.model.parameters()])
        mem_bufs = sum([buf.nelement()*buf.element_size() for buf in self.model.buffers()])
        mem = mem_params + mem_bufs # in bytes
        return mem

    def __call__(self,x):
        device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
        x = x.to(device)
        self.model.to(device)
        return self.model(x)

    def train(self, kwargs_dataset, kwargs_process):

        args = argparse.Namespace(**kwargs_process)

        # device
        device = torch.device(f"cuda:{args.local_rank}" if torch.cuda.is_available() else "cpu")

        # random seed
        random.seed(args.seed)
        numpy.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(kwargs_process["seed"])
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # network
        model = self.model
        model = model.to(device)

        logger.info(f"Raw model GPU memory consumption: {self.model_gpu_memory_report()*1e-6:10.4f}MB")
 
        # optimizer
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, betas=(.5, 0.999))
        logger.info(f"Process {os.getpid()} on {device} has constructed optimizer.")

        # dataset
        dataset = str_get(kwargs_dataset["name"],datasets,**kwargs_dataset)

        dataset_trval,dataset_te = leave_one_out(dataset, kwargs_dataset["test_envs"], kwargs_dataset["train_envs"])

        if len(kwargs_dataset["extra_training_sets"]) > 0:
            dataset_tr_ext = datasets.PT_Datasets(kwargs_dataset["extra_training_sets"])
            dataset_trval = torch.utils.data.ConcatDataset([dataset_trval] + dataset_tr_ext.datasets)

        dataset_tr, dataset_val = random_train_test_split(dataset_trval,args.train_ratio)

        logger.info(f"Process {os.getpid()} on {device} has constructed dataset")
        sampler_tr = Sampler(
            dataset_tr, distributed.get_world_size(), distributed.get_rank(), shuffle = True, 
                seed = kwargs_process["seed"], drop_last = True)
        dataloader_tr = DataLoader(
            dataset_tr, sampler=sampler_tr,
            batch_size=args.batch_size, 
            num_workers=args.num_workers)

        sampler_val = Sampler(
            dataset_val, distributed.get_world_size(), distributed.get_rank(), shuffle = True, 
                seed = kwargs_process["seed"], drop_last = True)
        dataloader_val = DataLoader(
            dataset_val, sampler=sampler_val,
            batch_size=args.batch_size, 
            num_workers=args.num_workers)

        sampler_te = Sampler(
            dataset_te, distributed.get_world_size(), distributed.get_rank(), shuffle = True, 
                seed = kwargs_process["seed"], drop_last = True)
        dataloader_te = DataLoader(
            dataset_te, sampler=sampler_te,
            batch_size=args.batch_size, 
            num_workers=args.num_workers)

        logger.info(f"Process {os.getpid()} on {device} has constructed dataloader")

        # Lists to keep track of progress
        records_name = [
            "epoch",
            "iter",
            "tr_acc",
            "val_acc",
            "te_acc",
            "tr_loss",
            "val_loss",
            "te_loss",
        ]
        records = {
            "epoch":[],
            "iter":[],
            "tr_acc":[],
            "val_acc":[],
            "te_acc":[],
            "tr_loss":[],
            "val_loss":[],
            "te_loss":[],
        }

        # train loop
        logger.info(f"Process {os.getpid()} on {device} has reached training loop")
        for epoch in range(args.epochs):

            sampler_tr.set_epoch(epoch)
            sampler_val.set_epoch(epoch)
            sampler_te.set_epoch(epoch)

            if epoch > 0 and epoch % args.lr_decay_step ==0:
                decay_lr(optimizer, args.decay_rate)

            iters = len(dataloader_tr)*(epoch)
            
            for i, (x,y) in enumerate(dataloader_tr, iters):

                model.train()

                x = x.to(device)
                y = y.to(device)

                model.zero_grad()
                # Forward pass real batch through D
                logit = model(x).view(-1, args.num_categories)
                # Calculate loss on all-real batch
                loss = functional.cross_entropy(logit, y)
                # Calculate gradients for D in backward pass
                loss.backward()
                optimizer.step()

                # Output training stats
                if i % args.output_steps == 0:
                    model.eval()
                    with torch.no_grad():

                        records["epoch"].append(epoch)
                        records["iter"].append(i)
                        records["tr_acc"].append(Accuracy(logit,y))
                        records["tr_loss"].append(loss.item())

                        acc_te = 0
                        loss_te = []
                        count_te = 0
                        
                        for i, (x_te,y_te) in enumerate(dataloader_te):
                            x_te = x_te.to(device)
                            y_te = y_te.to(device)
                            logit_te = model(x_te).view(-1, args.num_categories)
                            loss_te.append(functional.cross_entropy(logit_te, y_te).item())
                            acc_te = acc_te + Correct_Prediction(logit_te,y_te)
                            count_te = count_te + x_te.shape[0]

                        records["te_acc"].append((acc_te / count_te))
                        records["te_loss"].append(numpy.mean(loss_te))

                        acc_val = 0
                        loss_val = []
                        count_val = 0

                        for i, (x_val,y_val) in enumerate(dataloader_val):
                            x_val = x_val.to(device)
                            y_val = y_val.to(device)
                            logit_val = model(x_val).view(-1, args.num_categories)
                            loss_val.append(functional.cross_entropy(logit_val, y_val).item())
                            acc_val = acc_val + Correct_Prediction(logit_val,y_val)
                            count_val = count_val + x_val.shape[0]

                        records["val_acc"].append((acc_val / count_val))
                        records["val_loss"].append(numpy.mean(loss_val))

                    logger.info(pprint([(k,records[k][-1]) for k in records_name],))

        self.model = model
        logger.info(cuda.memory_summary(device))

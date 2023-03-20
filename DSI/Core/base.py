import torch, numpy, random, os, logging, json
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import utils
import torch.cuda as cuda
import torch.distributed as distributed
import torch.multiprocessing as multiprocessing
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.optim import ZeroRedundancyOptimizer as ZeRO
from torch.utils.data.distributed import DistributedSampler as Sampler
# from torch.utils.tensorboard import SummaryWriter

from itertools import combinations

from ..Predictor import nets
from ..Data import datasets
from ..Data import utils as Datautils
from ..DiffusionModel import diffusion_models

from ..Predictor.ensemble import *
from ..plot import *
from ..metrics import *
from ..utils import *
from .confidence import *

from improved_diffusion.pp import Hardness

"""
Code here is used to control the main flow of the attribution correction (completion, correction, and projection).
"""

logger = logging.getLogger(module_structure(__file__))

class BaseAttModer:
    def __init__(
        self, 

        world_kwargs = None,

        predictor_kwargs = None,
        diffuser_kwargs = None,

        predictor_training_kwargs = None, #{"data":,"process"}
        diffuser_training_kwargs = None, #{"data":,"process"}

        sampling_kwargs = None, #{"data":,"process"}

        load_predict_kwargs = None,
        ):
        """
        classifier/downstream task handler
        diffussion model
        """

        self.world_kwargs = world_kwargs
        self.predictor_kwargs = predictor_kwargs
        self.diffuser_kwargs = diffuser_kwargs 
        self.predictor_training_kwargs = predictor_training_kwargs 
        self.diffuser_training_kwargs = diffuser_training_kwargs 
        self.sampling_kwargs = sampling_kwargs 
        self.load_predict_kwargs = load_predict_kwargs

        self.world_kwargs["device"] = f"cuda:{self.world_kwargs['local_rank']}" if torch.cuda.is_available() else "cpu"

        if self.world_kwargs["action"] in ["ai","sn"]:
            if diffuser_kwargs["pp"] == "hardness":
                self.diffuser_kwargs["hardness"] = self.get_hardness()
            else:
                self.diffuser_kwargs["hardness"] = None
            self.diffuser = str_get(diffuser_kwargs["name"], diffusion_models,**diffuser_kwargs, **world_kwargs)
            self.create_DDP("diffusor")
        elif self.world_kwargs["action"] in ["tp","lp","lpe","lpm"]:
            self.predictor = str_get(predictor_kwargs["name"],nets,**predictor_kwargs, **world_kwargs)
        elif self.world_kwargs["action"] in ["f","debug","debug2"]:
            self.diffuser = str_get(diffuser_kwargs["name"], diffusion_models,**diffuser_kwargs, **world_kwargs)
            self.create_DDP("diffusor")
            self.predictor = str_get(predictor_kwargs["name"],nets,**predictor_kwargs, **world_kwargs)

        self.log_parameters()

    def PACSValidationCheck(self):
        assert self.sampling_kwargs["data"]["name"] == "PACS"
        assert self.predictor_training_kwargs["data"]["name"] == "PACS"

        PACS_dmoains = {"a":0,"c":1,"p":2,"s":3}
        diffusor_training_domain = PACS_dmoains[self.diffuser_kwargs["model_type"][-1]]
        predictor_training_domains = self.predictor_training_kwargs["data"]["train_envs"]
        predictor_testing_domains = self.predictor_training_kwargs["data"]["test_envs"]
        sampling_domain = self.sampling_kwargs["data"]["domain_index"]
        assert self.diffuser_kwargs["model_type"] in predictor_training_domains
        assert predictor_testing_domains not in predictor_training_domains
        assert predictor_testing_domains  == sampling_domain        

    def get_hardness(self):
        dataset = str_get(
            self.diffuser_kwargs["hardness"]["name"],datasets, **self.diffuser_kwargs["hardness"],
        )[self.diffuser_kwargs["hardness"]["select_domain"]]
        dataloader = DataLoader( dataset, shuffle = True, batch_size=1024, num_workers=2)
        y = next(iter(dataloader))[0]
        y = y.to(self.world_kwargs["device"])
        return Hardness(y)
    
    def log_parameters(self):
        if self.world_kwargs["local_rank"] == 0:
            for k,v in vars(self).items():
                if k.endswith("_kwargs"):
                    logger.info("*="*20)
                    logger.info(f"{k}:")
                    for name,value in v.items():
                        if isinstance(value, dict):
                            logger.info(f"\t{name}:{json.dumps(value, indent = 2, ensure_ascii = False)}")
                        else:
                            logger.info(f"\t{name}:{value}")
                    logger.info("-~"*20)

    def create_DDP(self, model = "diffusor"):
        if isinstance(model,str):
            model = [model]
        if "diffusor" in model:
            self.diffuser.model = DDP(self.diffuser.model,device_ids=[self.world_kwargs['local_rank']])
        if "predictor" in model:
            self.predictor.model.to(torch.device(f"cuda:{distributed.get_rank()}" if torch.cuda.is_available() else "cpu"))
            self.predictor.model = DDP(self.predictor.model,device_ids=[self.world_kwargs['local_rank']])

    def train_predictor(self): # TODO: 需要改成 多卡 模式

        self.predictor.build_model()
        if self.predictor_training_kwargs["data"]["name"] == "PACS" and 2 in self.predictor_training_kwargs["data"]["test_envs"] :
            self.predictor.modify_output(self.predictor_kwargs["num_classes"])
            logger.info(f"when testing envs includes photo, imagenet pretrained model cannot be used.")
        else:
            self.predictor.load_model()
            self.predictor.modify_output(self.predictor_kwargs["num_classes"])
            self.predictor.freeze_model()

        self.create_DDP("predictor")

        self.predictor.train(self.predictor_training_kwargs["data"],self.predictor_training_kwargs["process"]) 

    def load_model(self, model = "predictor", path = None): 
        self.predictor.build_model()
        self.predictor.modify_output(self.predictor_kwargs["num_classes"])

        self.predictor.load_model(path)

    def prepare_predictor(self): 
        # REUSE
        if self.predictor_kwargs["name"] == "EfficientNet":
            base_path = os.path.join(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
            path = os.path.join(base_path,"Predictor", "checkpoints", f"{self.get_hash('predictor')}.pt") 
            if os.path.exists(path):
                logger.info(f"predictor {self.get_hash('predictor')} to be loaded.")
                self.load_model(model = "predictor", path = path)
                logger.info(f"predictor {self.get_hash('predictor')} loaded.")
            else:
                logger.info(f"predictor {self.get_hash('predictor')} to be trained.")
                self.train_predictor()
                if self.world_kwargs['local_rank'] == 0:
                    self.save_model(model = "predictor", path = path)
                    logger.info(f"predictor {self.get_hash('predictor', True)} saved.")
        elif self.predictor_kwargs["name"] == "DomainbedNet":
            logger.info(f"predictor DomainbedNet to be loaded.")
            self.load_model(model = "predictor")
            logger.info(f"predictor DomainbedNet loaded.")

    def train_diffusor(self):
        self.diffuser.finetune(**self.diffuser_training_kwargs)

    def get_hash(self, model = "predictor", single = False): 
        if model == "predictor":
            if single:
                description = {
                    "predictor_kwargs":self.predictor_kwargs,
                    "predictor_training_kwargs":self.predictor_training_kwargs                
                }
                sorted_description = sort_dict(description)
                return string_hash(sorted_description)
            else:
                if self.world_kwargs['local_rank'] == 0:
                    description = {
                        "predictor_kwargs":self.predictor_kwargs,
                        "predictor_training_kwargs":self.predictor_training_kwargs                
                    }
                    sorted_description = sort_dict(description)
                    predictor_hash = [string_hash(sorted_description)]
                else:
                    predictor_hash = [None]
                
                distributed.barrier()
                distributed.broadcast_object_list(predictor_hash, src=0)
                distributed.barrier()
                return predictor_hash[0]

    def save_model(self, model = "predictor", path = None): 
        if model == "predictor": 
            if path is None:
                # REUSE
                base_path = os.path.join(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
                path = os.path.join(base_path,"Predictor", "checkpoints", f"{self.get_hash('predictor',True)}.pt") 
            self.predictor.save_model(path)
            
    def invernorm(self, x, conduct = False):
        if conduct:
            inverse_mu = - numpy.array(self.sampling_kwargs["data"]["mean"]) / numpy.array(self.sampling_kwargs["data"]["std"])
            inverse_std = 1 / numpy.array(self.sampling_kwargs["data"]["std"])

            x_ = transforms.Normalize(inverse_mu, inverse_std, inplace=False)(x)
            x_ = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], inplace=False)(x_)
            return x_
        else:
            return x

    def sample(self):
        '''
            1 import data
            2 initialize the diffusion model and predictor
            3 process the data
            4 draw prediction
            5 output statistics: processed data: FID, KID classification: Accuracy
        '''
        
        device = self.world_kwargs["device"]

        random.seed(self.sampling_kwargs["process"]["seed"])
        numpy.random.seed(self.sampling_kwargs["process"]["seed"])
        torch.manual_seed(self.sampling_kwargs["process"]["seed"])
        torch.cuda.manual_seed_all(self.sampling_kwargs["process"]["seed"])
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        sample_number = 0

        with torch.no_grad():
            distributed.barrier()
            for i in range(0,self.sampling_kwargs["process"]["sample_amount"],self.sampling_kwargs["process"]["batch_size"]):

                samples,gen_labels = self.diffuser.sample() 

                if self.world_kwargs["local_rank"] == 0:
                    utils.save_image(
                        samples[:256], 
                        os.path.join(self.sampling_kwargs["process"]["img_path"],f"{i}.jpg"),
                        nrow = 16, normalize = True) # ATT normalize True
                    
                    logger.info(f"Displayed labels: {[(e,l) for e,l in enumerate(gen_labels[:256].detach().cpu().tolist())]}")

                torch.save(
                    {"samples":samples,"gen_labels":gen_labels,}, 
                    os.path.join(self.sampling_kwargs["process"]["img_path"],f"{i}_{self.world_kwargs['local_rank']}.pt"))

                sample_number = sample_number + samples.shape[0]

                logger.info(f"Finish sampling iteration {i}, with samples {sample_number}")

    def load_ai_data(self, folder):
        try:
            fs = os.listdir(folder)
            for f in fs:
                if f.endswith(".pt"):
                    break
            int(f.split("_")[1])
            def prefixfunc(f):
                return int(f.split("_")[1])
        except Exception as e:
            def prefixfunc(f):
                return int(f.split(".")[0])
        allfiles = [
            (
                prefixfunc(f), os.path.join(folder, f)
                ) for f in os.listdir(folder) if f.endswith(".pt")
            ]
        for file in sorted(allfiles, key = lambda x:x[0]):
            yield torch.load(file[1])

 
    def add_hist(self, writer, tens, name,step):
        for i in range(tens.shape[1]):
            writer.add_histogram(f"{name}_{i}",tens[:,i],step)

    def concate(self,old,new):
        if old is None:
            return new
        else:
            return torch.cat([old,new],dim =0)

    def load_predict_mix(self):

        print("*="*20)
        print(f"load_predict_kwargs:")
        for name,value in self.load_predict_kwargs.items():
            if isinstance(value, dict):
                print(f"\t{name}:{json.dumps(value, indent = 2, ensure_ascii = False)}")
            else:
                print(f"\t{name}:{value}")
        print("-~"*20)

        random.seed(self.load_predict_kwargs["process"]["seed"])
        numpy.random.seed(self.load_predict_kwargs["process"]["seed"])
        torch.manual_seed(self.load_predict_kwargs["process"]["seed"])
        torch.cuda.manual_seed_all(self.load_predict_kwargs["process"]["seed"])
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        device = self.world_kwargs["device"]

        # writer = SummaryWriter(log_dir = os.path.join(self.sampling_kwargs["process"]["img_path"],"tfboard"))

        clear_dataset = str_get(self.load_predict_kwargs["data"]["name"],datasets,**self.load_predict_kwargs["data"])[self.load_predict_kwargs["data"]["domain_index"]]
        clear_dataset, outstr = Datautils.use_shuffle(clear_dataset, self.load_predict_kwargs["data"]["name"], self.load_predict_kwargs["data"]["domain_index"])
        clear_dataset = datasets.wrapDataset(clear_dataset,self.sampling_kwargs["data"]["name"])
        logger.info(outstr)
        clear_sampler = Sampler(
            clear_dataset, distributed.get_world_size(), distributed.get_rank(), shuffle = False, 
                seed = self.load_predict_kwargs["process"]["seed"], drop_last = True)
        clear_dataloader = DataLoader(
            clear_dataset, sampler=clear_sampler,
            batch_size=self.load_predict_kwargs["process"]["batch_size"], 
            num_workers=self.load_predict_kwargs["process"]["num_workers"])

        domain_T_data_dict = {}
        domains = ["clear"]
        for T,domain_path_dict in self.load_predict_kwargs["data"]["checkpoints_folders"].items():
            domain_T_data_dict[T] = {}
            for domain,path in domain_path_dict.items():
                domain_T_data_dict[T][domain] = self.load_ai_data(path)
                domains.append(domain)

        domains = set(domains)
        sorted_Ts = sorted([int(T) for T in domain_T_data_dict.keys()],reverse = False)

        ensemble_dict = {"y":None}

        for n in range(1,len(domains)+1):
            for k in combinations(domains, n):
                k = tuple(k)
                ensemble_dict[k] = {"all":0,"clear_true":0,"clear_false":0,"predict":None}

        sample_number = 0

        with torch.no_grad():
            distributed.barrier()
            clear_sampler.set_epoch(0)
            for i, data in enumerate(clear_dataloader, 0):
                # print("is training",self.predictor.model.training)
                # get the inputs; data is a list of [inputs, labels]
                clear, labels = data

                if self.sampling_kwargs["data"]["name"] == "FixedColoredMNIST":
                    clear = clear[:,:2,:,:]

                ensemble_dict["y"]  = self.concate(ensemble_dict["y"],labels) 

                clear = clear.to(device)

                labels = labels.to(device)

                clear_ = self.invernorm(clear)
                logit_clear = self.predictor(clear_)
                cpl_clear = Correct_Prediction_list(logit_clear, labels)

                sample_number = sample_number + clear.shape[0]

                logits_batch = dict( [(str(t),{"clear":logit_clear}) for t in sorted_Ts] )

                for T, domains_data in domain_T_data_dict.items():
                    for domain, data_iter in domains_data.items():
                        saved_pack = next(data_iter)
                        if isinstance(saved_pack, (tuple,list,dict)):
                            augmented, labels_d = saved_pack["augmented"], saved_pack["gen_labels"]
                            labels_d = labels_d.to(device)
                            assert torch.all(labels_d == labels)
                        else:
                            augmented = saved_pack.float()
                        augmented = transforms.Resize(self.load_predict_kwargs["data"]["image_size"])(augmented)
                        if self.sampling_kwargs["data"]["name"] == "FixedColoredMNIST":
                            augmented = augmented[:,:2,:,:]
                        augmented = augmented.to(device)
                        if augmented.min().item() >=0:
                            augmented = augmented * 2 * self.load_predict_kwargs["data"]["augmented_scale"] - 1 * self.load_predict_kwargs["data"]["augmented_scale"]
                        else:
                            augmented = augmented * self.load_predict_kwargs["data"]["augmented_scale"] 
                            print("not use mean")
                        augmented_ = self.invernorm(augmented) 
                        logit_aug = self.predictor(augmented_)
                        logits_batch[T][domain] = logit_aug

                for ensemble in ensemble_dict.keys():
                    if not isinstance(ensemble, tuple):
                        continue
                    final_logit = logit_clear
                    final_mask = maximalclassprobability(logit_clear, self.load_predict_kwargs["process"]["confidence_threshold"])
                    for T in sorted_Ts:
                        T = str(T)
                        T_ensembled_logit = ensemblek(logits_batch[T], ensemble)
                        T_mask = maximalclassprobability(T_ensembled_logit, self.load_predict_kwargs["process"]["confidence_threshold"])
                        if final_logit is None:
                            final_logit = T_ensembled_logit
                            final_mask = T_mask
                        else:
                            increment_mask = (~final_mask) & T_mask 
                            final_mask = final_mask | T_mask
                            final_logit[increment_mask] = T_ensembled_logit[increment_mask]
                        if torch.all(final_mask):
                            break
                    cpl_ensemble_logit = Correct_Prediction_list(final_logit, labels)

                    ensemble_dict[ensemble]["all"]  = ensemble_dict[ensemble]["all"] + cpl_ensemble_logit.sum().item()
                    ensemble_dict[ensemble]["clear_true"]  = ensemble_dict[ensemble]["clear_true"] + cpl_ensemble_logit[cpl_clear].sum().item()
                    ensemble_dict[ensemble]["clear_false"]  = ensemble_dict[ensemble]["clear_false"] + cpl_ensemble_logit[~cpl_clear].sum().item()
                    ensemble_dict[ensemble]["predict"]  = self.concate(ensemble_dict[ensemble]["predict"],cpl_ensemble_logit.detach().cpu()) 

                logger.info(f"At iteration {i},sample number={sample_number}, clear true is {ensemble_dict[('clear',)]['clear_true']}")
                for ensemble in ensemble_dict.keys():
                    if not isinstance(ensemble, tuple):
                        continue
                    vdict = ensemble_dict[ensemble]
                    logger.info(f"At iteration {i}, ensemble {ensemble}, acc {vdict['all'] / sample_number}, " \
                        f"acc true clear {vdict['clear_true'] / ensemble_dict[('clear',)]['clear_true']}, " \
                            f"acc false clear {vdict['clear_false'] / (sample_number - ensemble_dict[('clear',)]['clear_true'])}")

                if sample_number > self.load_predict_kwargs["process"]["sample_amount"]:
                    logger.info(cuda.memory_summary(device))
                    print(self.load_predict_kwargs["data"]["name"],self.load_predict_kwargs["data"]["domain_index"],self.predictor.checkpoint["args"]['algorithm'])
                    for ensemble in ensemble_dict.keys():
                        if not isinstance(ensemble, tuple):
                            continue
                        vdict = ensemble_dict[ensemble]
                        print(f"At iteration {i}, ensemble {ensemble}, acc {vdict['all'] / sample_number}, " \
                            f"acc true clear {vdict['clear_true'] / ensemble_dict[('clear',)]['clear_true']}, " \
                                f"acc false clear {vdict['clear_false'] / (sample_number - ensemble_dict[('clear',)]['clear_true'])}")
                    info_dict = {
                        "results":ensemble_dict,
                        "params":{"load_predict_kwargs":self.load_predict_kwargs,"predictor":self.predictor.checkpoint["args"]['algorithm'],"seed":self.predictor.checkpoint["args"]['seed']},
                    }
                    torch.save(info_dict, os.path.join(self.sampling_kwargs["process"]["img_path"],f"result.pt"))
                    break
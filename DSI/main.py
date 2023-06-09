import sys,os,logging,argparse,json

from .Core import base, Screen
from .utils import *

def get_args():
    parser = argparse.ArgumentParser()

    parser_world = parser.add_argument_group('world')
    parser_world.add_argument('--local_rank', type=int, default=-1, help='Local rank passed by torch.distributed.launch')
    parser_world.add_argument('--exp_group', "-G", type=str, default="", help='The group name of the experiment')
    parser_world.add_argument('--action', "-A", type=str, default="f", help='The function to be carried out')

    parser_general = parser.add_argument_group('general')
    parser_general.add_argument('--image_size', "-is", type=int, default=64, help='image size for all exps')

    parser_predictor = parser.add_argument_group('predictor')
    parser_predictor.add_argument('--predictor_name', "-Pn", type=str, default="DomainbedNet", help='name of the predictor: "EfficientNet" or "DomainbedNet"')
    parser_predictor.add_argument('--DomainbedNet_checkpoint_path', "-dncp", type=str, default="", help='the checkpoint path of the domain bed net')
    parser_predictor.add_argument('--predictor_structure', "-P", type=str, default="b0", help='structure of the predictor')

    parser_predictor_training = parser.add_argument_group('predictor_training')
    parser_predictor_training.add_argument('--predictor_training_dataset', "-pD", type=str, default="PACS", help='the dataset used for finetune predictor')
    parser_predictor_training.add_argument('--predictor_extra_training_dataset', "-pDe", type=str, nargs = "*", default=[], help='supplementary training dataset')
    parser_predictor_training.add_argument('--predictor_test_env', "-E", type=int, nargs='+', default=[1], help='index of the test domain')
    parser_predictor_training.add_argument('--predictor_train_env', "-rE", type=int, nargs='+', default=[2], help='index of the train domain')
    parser_predictor_training.add_argument('--predictor_lr', "-plr", type=float, default=1e-3, help='learning rate of predictor training')
    parser_predictor_training.add_argument('--predictor_epoch', "-pE", type=int, default=21, help='epoch of predictor training')

    parser_diffusor = parser.add_argument_group('diffusor')
    parser_diffusor.add_argument('--diffusion_model',"-dm",type = str, default = "imagenet_64", help = 'the diffusion model name')
    parser_diffusor.add_argument('--spacing',"-S",type = str, default = "100,100,40", help = 'the steps used in the different sampling stages')
    parser_diffusor.add_argument('--diffusion_inner_batch_size',"-dib",type = int, default = 16, help = 'the inner batchsize of the diffusor. ')
    parser_diffusor.add_argument('--use_ddim',"-UD", action='store_true', default = False, help = 'Whether use ddim for sampling')
    parser_diffusor.add_argument('--post_processing_method',"-pp",type = str, default = "", help = 'use what kind of post processing method, empty string means no post processing')
    parser_diffusor.add_argument('--lowpass_mode',"-lpm",type = str, default = "smooth", help = 'the lowpass method to use')
    parser_diffusor.add_argument('--lowpass_scale',"-lps",type = int, default = 8, help = 'the lowpass scale to use')
    parser_diffusor.add_argument('--combine_alpha',"-caph",type = float, default = 0.5, help = 'the combination coefficient')
    parser_diffusor.add_argument('--frange',"-frng",type = float, default = (0.1,0.9), nargs = 2, help = 'the range of frequence generated by diffusion model')
    parser_diffusor.add_argument('--hardness_dataset',"-hdD",type = str, default = "PACS", help = 'the dataset used for hardness')
    parser_diffusor.add_argument('--hardness_domain',"-hdd",type = int, default = 0, help = 'the domain of the dataset used for hardness')
    parser_diffusor.add_argument('--post_processing_time',"-ppt",type = int, nargs=2, default = [10000,10000], help = 'The period to use post processing')

    parser_diffusor_training = parser.add_argument_group('diffusor_training')
    parser_diffusor_training.add_argument('--diffusor_train_env', "-drE", type=int, nargs='+', default=[2], help='index of the train domain for diffusor finetune')
    
    parser_sampling = parser.add_argument_group('sampling')
    parser_sampling.add_argument('--reverse_start_time', "-T", type=int, default=8, help='The starting time index of the reverse sampling')
    parser_sampling.add_argument('--sampling_dataset', "-sD", type=str, default="PACS", help='The clear dataset to be augmented')
    parser_sampling.add_argument('--sampling_domain', "-sd", type=int, default=0, help='The domain to be augmented')
    parser_sampling.add_argument('--sampling_batch_size', "-sbs", type=int, default=32, help='')
    parser_sampling.add_argument('--sampling_amount', "-sa", type=int, default=50000, help='')
    parser_sampling.add_argument('--sampling_seed', "-ss", type=int, default=0, help='')
    parser_sampling.add_argument('--sampling_augmentation_times', "-sat", type=int, default=1, help='')
    
    parser_load_predict = parser.add_argument_group('load_predict')
    parser_load_predict.add_argument('--load_predict_domains', "-lpd", type=int, default=[], nargs='+', help='the domain labels used in load and predict')
    parser_load_predict.add_argument('--load_predict_data_folders', "-lpdf", type=str, default=[], nargs='+', help='the data floders used in load and predict')
    parser_load_predict.add_argument('--load_predict_checkpoints_dict', "-lpcd", type=str, default='', help='the json string of the data folders')
    parser_load_predict.add_argument('--load_predict_threshold', "-lpt", type=float, default=0, help='threshold used in load predict process')
    parser_load_predict.add_argument('--load_predict_scale', "-lpsc", type=float, default=1.5, help='rescale the input to align with the original figure')

    parser_screen = parser.add_argument_group('screen')
    parser_load_predict.add_argument('--screen_checkpoints_dict', "-scd", type=str, default="",   help='classifier ckpt path')
    parser_load_predict.add_argument('--screen_data_path', "-sdp", type=str, default=[], nargs="+",  help='data path')
    parser_load_predict.add_argument('--screen_skip', "-sskp", type=int, default=0, help='skip the first sskp batches')
    parser_load_predict.add_argument('--screen_scale_clear_ensemble', "-ssce", type=float, default=1.0, help='scale the clear logit when ensemble, set to 0 to remove it')

    return parser.parse_args()

def generate_path(args):
    # REUSE
    base_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), "exp")
    exp_group = args.exp_group
    group_path = os.path.join(base_path,exp_group)
    os.makedirs(group_path, exist_ok = True)

    args_dict = vars(args)
    args_dict["local_rank"] = 0
    sorted_description = sort_dict(args_dict)

    exp_path = os.path.join(group_path, f"{args.action}-{args.sampling_dataset}-{args.sampling_domain}-{string_hash(sorted_description)}")
    os.makedirs(exp_path, exist_ok = True)
    
    trash_path = os.path.join(exp_path,"trash.log")
    log_path = os.path.join(exp_path,"log.log")
    img_path = os.path.join(exp_path, "img")
    os.makedirs(img_path, exist_ok = True)
    
    print(trash_path, log_path, img_path)
    
    return trash_path, log_path, img_path


if __name__ == "__main__":
    print("starting the main process")
    args = get_args()
    print("got args")

    local_rank = args.local_rank

    assert args.action in [
        "lpm", # "load augmented data and draw prediction" # 
        ]

    trash_path, log_path, img_path = generate_path(args)
    print("path generated")

    device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"

    LOG_FORMAT = "%(asctime)s | %(name)s " + f"| {device} " + "| %(message)s" 
    logging.basicConfig(filename=log_path, level=logging.INFO, format=LOG_FORMAT)
    
    logger = logging.getLogger(module_structure(__file__))
    # logger = logging.getLogger()

    logger.info(f"trash path: {trash_path}")
    logger.info(f"log path: {log_path}")
    logger.info(f"img path: {img_path}")

    establish_communication(device = local_rank)
    print("communication established")
    # os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_device
    
    ba = base.BaseAttModer(

            world_kwargs = {
                "local_rank":local_rank,
                "action":args.action,
            },

            predictor_kwargs = {
                "name":args.predictor_name,
                "checkpointpath":args.DomainbedNet_checkpoint_path,
                "version":args.predictor_structure, 
                "num_classes":7,
            },

            diffuser_kwargs = {
                "name":"ImprovedDiffusionTransform",
                "model_type":args.diffusion_model,
                "spacing":args.spacing, 
                "num_samples":args.sampling_batch_size, # ATT equal to the batch size in the sampling kwargs
                "inner_batch_size":args.diffusion_inner_batch_size,
                "use_ddim":args.use_ddim,
                "pp": None if args.post_processing_method == "" else args.post_processing_method,
                "scale":args.lowpass_scale,
                "mode":args.lowpass_mode,
                "alpha":args.combine_alpha,
                "frange":args.frange,
                "hardness":{
                    "name":args.hardness_dataset,
                    "root":"~/Data/", 
                    "test_envs":[0], 
                    "image_size" : args.image_size,
                    "data_augmentation":False, 
                    "mean": (0.8263, 0.7787, 0.7222),# "mean": (0.5, 0.5, 0.5),
                    "std": (0.2383, 0.2741, 0.3325),# "std": (0.5, 0.5, 0.5),
                    "select_domain":args.hardness_domain,
                },
                "ppt":args.post_processing_time,
            },

            predictor_training_kwargs = {
                "data":{
                    "name":args.predictor_training_dataset,
                    "root":"~/Data/", 
                    "extra_training_sets":args.predictor_extra_training_dataset,
                    "test_envs":args.predictor_test_env, 
                    "train_envs":args.predictor_train_env, 
                    "image_size" : args.image_size,
                    "data_augmentation":False,# ATT: True
                },
                "process":{
                    "local_rank":local_rank,
                    "seed":0,
                    "learning_rate":args.predictor_lr,
                    "train_ratio":0.9,
                    "batch_size":64,
                    "num_workers":2,
                    "epochs":args.predictor_epoch, 
                    "num_categories":7,
                    "output_steps":50,
                    "decay_rate":0.3,
                    "lr_decay_step":5,
                }
            },
            diffuser_training_kwargs = {
                "data":{
                    "name":"PACS",
                    "root":"~/Data/", 
                    "test_envs":[], 
                    "train_envs":args.diffusor_train_env, 
                    "image_size" : args.image_size,
                    "data_augmentation":False, 
                    "mean": (0.5, 0.5, 0.5),
                    "std": (0.5, 0.5, 0.5),
                    "new_classes": [207,385,None,402,None,498,None],
                },"process":{
                    "num_workers":2,
                }},

            sampling_kwargs = {
                "data":{
                    "name":args.sampling_dataset,
                    "root":"~/Data/", 
                    "test_envs":[1], 
                    "image_size" : args.image_size,
                    "data_augmentation":False, 
                    "domain_index":args.sampling_domain, # which domain to be augmented
                },
                "process":{
                    "time_index":args.reverse_start_time,
                    "num_workers":2,
                    "batch_size":args.sampling_batch_size, 
                    "sample_amount":args.sampling_amount,
                    "seed":args.sampling_seed,
                    "img_path":img_path,
                    "aug_iter":args.sampling_augmentation_times,
                }
            },

            load_predict_kwargs = {
                "data":{
                    "name":args.sampling_dataset,
                    "root":"~/Data/", 
                    "test_envs":[1], 
                    "image_size" : args.image_size,
                    "data_augmentation":False, 
                    "domain_index":args.sampling_domain, # which domain to be augmented
                    "checkpoints_folders":json.load(open(args.load_predict_checkpoints_dict,"r")) if args.load_predict_checkpoints_dict else dict(zip(args.load_predict_domains, args.load_predict_data_folders)),
                    "augmented_scale":args.load_predict_scale,
                },
                "process":{
                    "num_workers":2,
                    "batch_size":args.sampling_batch_size, 
                    "sample_amount":args.sampling_amount,
                    "seed":args.sampling_seed,
                    "confidence_threshold":args.load_predict_threshold,
                    "clear_scale":args.screen_scale_clear_ensemble,
                }
            },

    )
    print("basemodel established")

    if args.action  == "lpm":
        print("dive into load and prediction")
        ba.prepare_predictor()
        ba.load_predict_mix()
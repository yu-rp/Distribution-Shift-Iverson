import torch

from ..metrics import *

def ensembleN(data_dict, y, model):
    log = []
    for k,v in data_dict.keys():
        x = v["x"]
        logits = model(x)
        v["logits"] = logits
        v["acc"] = Accuracy(logits, y)
        log.append(logits)
    log = torch.stack(log,dim = 0)
    logit = log.mean(dim = 0)
    overallacc= Accuracy(logit, y)
    return data_dict, overallacc

def ensemble2(logits1, logits2, y, acc = False):
    metric = Accuracy if acc else Correct_Prediction
    log = []
    # metric1 = metric(logits1, y)
    log.append(logits1)
    # metric2 = metric(logits2, y)
    log.append(logits2)
    log = torch.stack(log,dim = 0)
    logit = log.mean(dim = 0)
    overallmetric= metric(logit, y)
    return overallmetric

def ensemble(logits1, logits2):
    metric = Correct_Prediction_list
    log = []
    # metric1 = metric(logits1, y)
    log.append(logits1)
    # metric2 = metric(logits2, y)
    log.append(logits2)
    log = torch.stack(log,dim = 0)
    logit = log.mean(dim = 0)
    return logit

def normalize_logit(logit):
    mu = logit.mean(dim=1, keepdim=True).expand(*logit.shape)
    std = logit.std(dim=1, keepdim=True).expand(*logit.shape)
    return (logit - mu) / std

def ensemblek(logit_dict, ks, normalize = False):
    log = []
    for k in ks:
        if normalize:
            log.append(normalize_logit(logit_dict[k]))
        else:
            log.append(logit_dict[k])
    # import pdb
    # pdb.set_trace()
    log = torch.stack(log,dim = 0)
    logit = log.mean(dim = 0)
    return logit

def ensemble_load_pred(logit_dict, labels, ensemble_dict):
    clear = logit_dict["clear"]
    cpl_clear = Correct_Prediction_list(clear, labels)

    for ensemble in ensemble_dict.keys():
        templogit = ensemblek(logit_dict, ensemble)
        cpl_templogit = Correct_Prediction_list(templogit, labels)
        ensemble_dict[ensemble]["all"]  = ensemble_dict[ensemble]["all"] + cpl_templogit.sum().item()
        ensemble_dict[ensemble]["clear_true"]  = ensemble_dict[ensemble]["clear_true"] + cpl_templogit[cpl_clear].sum().item()
        ensemble_dict[ensemble]["clear_false"]  = ensemble_dict[ensemble]["clear_false"] + cpl_templogit[~cpl_clear].sum().item()
    
    return ensemble_dict


def ensemble_load_pred_sfotmax(logit_dict, labels, ensemble_dict):
    clear = logit_dict["clear"]
    cpl_clear = Correct_Prediction_list(clear, labels)

    for ensemble in ensemble_dict.keys():
        templogit = ensemblek(logit_dict, ensemble)
        cpl_templogit = Correct_Prediction_list(templogit, labels)
        ensemble_dict[ensemble]["all"]  = ensemble_dict[ensemble]["all"] + cpl_templogit.sum().item()
        ensemble_dict[ensemble]["clear_true"]  = ensemble_dict[ensemble]["clear_true"] + cpl_templogit[cpl_clear].sum().item()
        ensemble_dict[ensemble]["clear_false"]  = ensemble_dict[ensemble]["clear_false"] + cpl_templogit[~cpl_clear].sum().item()
    
    return ensemble_dict
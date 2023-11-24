import os
import sys

sys.path.append("/mnt/efs/Libs")
sys.path.append("/mnt/efs")
sys.path.append("networks")
import numpy as np
import torch


def get_method_here(model_name, weights_path):
    basePath = "/mnt/efs/"
    if False:
        pass
    elif model_name == "Grag2021_progan":
        model_name = "Grag2021_progan"
        model_path = os.path.join(
            basePath + weights_path, model_name + "/model_epoch_best.pth"
        )
        print("ITS ME PROGAN", model_path)
        arch = "res50stride1"
        norm_type = "resnet"
        patch_size = None
    elif model_name == "Grag2021_latent":
        model_name = "Grag2021_latent"
        model_path = os.path.join(
            basePath + weights_path, model_name + "/model_epoch_best.pth"
        )
        arch = "res50stride1"
        norm_type = "resnet"
        patch_size = None
    else:
        print(model_name)
        from get_method import get_method

        model_name, model_path, arch, norm_type, patch_size = get_method(model_name)

    return model_name, model_path, arch, norm_type, patch_size


def rule_minmax(x):
    x = x.reshape(x.shape[0], 1, -1)
    sm = torch.mean(x, -1)
    su = torch.max(x, -1)[0]
    sd = torch.min(x, -1)[0]
    assert sm.shape == su.shape
    return torch.where(sm <= 0, sd, su)


def rule_trim(x, th=np.log(0.8), tl=np.log(0.2)):
    x = torch.nn.functional.logsigmoid(x)
    x = x.view(x.shape[0], 1, -1)
    a = torch.median(x, -1)[0]
    su = torch.sum(torch.where(x >= th, x, torch.zeros_like(x)), -1) / torch.sum(
        x >= th, -1
    )
    sd = torch.sum(torch.where(x <= tl, x, torch.zeros_like(x)), -1) / torch.sum(
        x <= tl, -1
    )
    x = torch.mean(x, -1)
    x = torch.where(a >= th, su, x)
    x = torch.where(a <= tl, sd, x)
    return x


dict_rule = {
    "avg": lambda x: torch.mean(x, (-2, -1)),
    "max": lambda x: torch.quantile(x.reshape(x.shape[0], x.shape[1], -1), 1.0, dim=-1),
    "perc97": lambda x: torch.quantile(
        x.reshape(x.shape[0], x.shape[1], -1), 0.97, dim=-1
    ),
    "minmax": rule_minmax,
    "trim": rule_trim,
    "lss": lambda x: torch.logsumexp(torch.nn.functional.logsigmoid(x), (-2, -1)),
}


def avpool(x, s):
    import scipy.ndimage as ndi

    h = s // 2
    return ndi.uniform_filter(x, (1, s, s), mode="constant")[:, h : 1 - h, h : 1 - h]


def def_size_avg(arch):
    if arch == "res50":
        return 8
    elif arch == "res50stride1":
        return 8
    else:
        assert False


def def_model(arch, model_path, localize=False):
    import torch

    if arch == "res50":
        from networks.resnet import resnet50

        model = resnet50(num_classes=1)
    elif arch == "resnet18":
        from torchvision.models import resnet18

        model = resnet18(pretrained=True)
        num_ftrs = model.fc.in_features
        model.fc = torch.nn.Linear(num_ftrs, 1)
    elif arch == "res50stride1":
        import networks.resnet_mod as resnet_mod

        model = resnet_mod.resnet50(num_classes=1, gap_size=1, stride0=1)
    else:
        print(arch)
        assert False

    assert localize is False

    if model_path == "None":
        Warning("model_path is None! No weights loading in eval.py")
    else:
        dat = torch.load(model_path, map_location="cpu")
        if "model" in dat:
            if (
                ("module._conv_stem.weight" in dat["model"])
                or ("module.fc.fc1.weight" in dat["model"])
                or ("module.fc.weight" in dat["model"])
            ):
                model.load_state_dict(
                    {key[7:]: dat["model"][key] for key in dat["model"]}
                )
            else:
                model.load_state_dict(dat["model"])
        elif "state_dict" in dat:
            model.load_state_dict(dat["state_dict"])
        elif "net" in dat:
            model.load_state_dict(dat["net"])
        elif "main.0.weight" in dat:
            model.load_state_dict(dat)
        elif "_fc.weight" in dat:
            model.load_state_dict(dat)
        elif "conv1.weight" in dat:
            model.load_state_dict(dat)
        else:
            print(list(dat.keys()))
            assert False
    return model

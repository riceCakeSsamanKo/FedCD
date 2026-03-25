import copy
import math
import torch
import torch.nn.functional as F
from torch import nn


class Flatten(nn.Module):
    def forward(self, x):
        return torch.flatten(x, 1)


class VGG8FeatureEncoder(nn.Module):
    def __init__(self, channels, in_channels=3):
        super().__init__()
        cfg = [channels[0], "M", channels[1], "M", channels[2], channels[3], "M"]
        layers = []
        current_in = in_channels
        for v in cfg:
            if v == "M":
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                layers.append(nn.Conv2d(current_in, v, kernel_size=3, padding=1))
                layers.append(nn.BatchNorm2d(v))
                layers.append(nn.ReLU(inplace=True))
                current_in = v
        self.features = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = Flatten()
        self.out_dim = channels[-1]

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        return self.flatten(x)


class VGGClassifierHead(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_classes):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        return self.classifier(x)


class TinyPMBranch(nn.Module):
    def __init__(self, in_channels, feature_dim):
        super().__init__()
        hidden = max(16, feature_dim // 4)
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            Flatten(),
            nn.Linear(32, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, feature_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class FedCD2Net(nn.Module):
    def __init__(self, gm_base, gm_head, pm_base, pm_head, gm_feature_dim, pm_feature_dim, fnc_dim):
        super().__init__()
        self.gm_base = gm_base
        self.gm_head = gm_head
        self.pm_base = pm_base
        self.pm_head = pm_head
        self.gm_proj = nn.Linear(gm_feature_dim, fnc_dim)
        self.pm_proj = nn.Linear(pm_feature_dim, fnc_dim)

    def gm_forward(self, x):
        feat = self.gm_base(x)
        if feat.ndim > 2:
            feat = torch.flatten(feat, 1)
        logits = self.gm_head(feat)
        return feat, logits

    def pm_forward(self, x):
        feat = self.pm_base(x)
        if feat.ndim > 2:
            feat = torch.flatten(feat, 1)
        logits = self.pm_head(feat)
        return feat, logits

    def forward_all(self, x):
        gm_feat, gm_logits = self.gm_forward(x)
        pm_feat, pm_logits = self.pm_forward(x)
        fused_logits = gm_logits + pm_logits
        gm_proj = F.normalize(self.gm_proj(gm_feat), dim=1)
        pm_proj = F.normalize(self.pm_proj(pm_feat), dim=1)
        return {
            "gm_features": gm_feat,
            "pm_features": pm_feat,
            "gm_logits": gm_logits,
            "pm_logits": pm_logits,
            "fused_logits": fused_logits,
            "gm_proj": gm_proj,
            "pm_proj": pm_proj,
        }

    def forward(self, x):
        return self.forward_all(x)["fused_logits"]


def _find_last_linear(module):
    last_linear = None
    for child in module.modules():
        if isinstance(child, nn.Linear):
            last_linear = child
    return last_linear


def infer_input_channels(dataset):
    if any(name in dataset for name in ["MNIST", "Fashion", "EMNIST", "Omniglot"]):
        return 1
    return 3


def _extract_base_and_head(model):
    model_copy = copy.deepcopy(model)

    if hasattr(model_copy, "features") and hasattr(model_copy, "avgpool") and hasattr(model_copy, "classifier"):
        base = nn.Sequential(model_copy.features, model_copy.avgpool, Flatten())
        head = copy.deepcopy(model_copy.classifier)
        head_linear = _find_last_linear(head)
        if head_linear is None:
            raise ValueError("FedCD2 requires a Linear classifier head.")
        return base, head, head_linear.in_features, True

    if hasattr(model_copy, "base") and hasattr(model_copy, "head"):
        base = copy.deepcopy(model_copy.base)
        head = copy.deepcopy(model_copy.head)
        head_linear = _find_last_linear(head)
        if head_linear is None:
            raise ValueError("FedCD2 requires a head with a Linear layer.")
        return base, head, head_linear.in_features, False

    if hasattr(model_copy, "fc") and isinstance(model_copy.fc, nn.Module):
        head = copy.deepcopy(model_copy.fc)
        feature_dim = head.in_features if isinstance(head, nn.Linear) else _find_last_linear(head).in_features
        model_copy.fc = nn.Identity()
        base = model_copy
        return base, head, feature_dim, False

    raise ValueError("FedCD2 currently supports VGG-style, BaseHeadSplit-style, or fc-style models.")


def _slice_tensor_to_shape(source, target_shape):
    out = torch.zeros(target_shape, dtype=source.dtype)
    slices = tuple(slice(0, min(s, t)) for s, t in zip(source.shape, target_shape))
    out[slices] = source[slices].clone()
    return out


def build_fedcd2_model(base_model, dataset, num_classes, pm_feature_dim=128, fnc_dim=128, pm_vgg_width_ratio=0.25):
    gm_base, gm_head, gm_feature_dim, is_vgg_style = _extract_base_and_head(base_model)
    in_channels = infer_input_channels(dataset)

    if is_vgg_style:
        gm_channels = [64, 128, 256, 256]
        pm_channels = [
            max(8, int(math.ceil(c * pm_vgg_width_ratio / 8.0) * 8))
            for c in gm_channels
        ]
        pm_base = VGG8FeatureEncoder(pm_channels, in_channels=in_channels)
        pm_hidden_dim = max(64, pm_base.out_dim)
        pm_head = VGGClassifierHead(pm_base.out_dim, pm_hidden_dim, num_classes)
        pm_feature_dim = pm_base.out_dim
    else:
        pm_base = TinyPMBranch(in_channels=in_channels, feature_dim=pm_feature_dim)
        pm_head = VGGClassifierHead(pm_feature_dim, max(64, pm_feature_dim), num_classes)

    return FedCD2Net(
        gm_base=gm_base,
        gm_head=gm_head,
        pm_base=pm_base,
        pm_head=pm_head,
        gm_feature_dim=gm_feature_dim,
        pm_feature_dim=pm_feature_dim,
        fnc_dim=fnc_dim,
    )


def extract_gm_state(model):
    return {
        "base": {k: v.detach().cpu().clone() for k, v in model.gm_base.state_dict().items()},
        "head": {k: v.detach().cpu().clone() for k, v in model.gm_head.state_dict().items()},
    }


def extract_pm_state(model):
    return {
        "base": {k: v.detach().cpu().clone() for k, v in model.pm_base.state_dict().items()},
        "head": {k: v.detach().cpu().clone() for k, v in model.pm_head.state_dict().items()},
    }


def average_nested_states(weighted_states):
    reference = weighted_states[0][1]
    averaged = {"base": {}, "head": {}}
    for section in ["base", "head"]:
        for key in reference[section].keys():
            agg = None
            for weight, state in weighted_states:
                tensor = state[section][key].float()
                agg = tensor * weight if agg is None else agg + tensor * weight
            averaged[section][key] = agg.type_as(reference[section][key])
    return averaged


def load_nested_state(module, state):
    module.gm_base.load_state_dict(state["base"], strict=True)
    module.gm_head.load_state_dict(state["head"], strict=True)


def load_pm_nested_state(module, state):
    module.pm_base.load_state_dict(state["base"], strict=True)
    module.pm_head.load_state_dict(state["head"], strict=True)


def slice_pm_from_gm(gm_state, pm_reference_state):
    extracted = {"base": {}, "head": {}}
    for section in ["base", "head"]:
        gm_items = list(gm_state[section].items())
        for idx, (ref_key, ref_tensor) in enumerate(pm_reference_state[section].items()):
            source_tensor = gm_state[section].get(ref_key)
            if source_tensor is None:
                if idx < len(gm_items):
                    source_tensor = gm_items[idx][1]
                else:
                    source_tensor = torch.zeros_like(ref_tensor)
            extracted[section][ref_key] = _slice_tensor_to_shape(source_tensor, ref_tensor.shape)
    return extracted

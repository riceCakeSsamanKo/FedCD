import math
from collections import OrderedDict

import numpy as np
import torch


def is_masked_parameter(name, param):
    return name.endswith("weight") and param.ndim > 1


def masked_parameter_names(model):
    return [name for name, param in model.named_parameters() if is_masked_parameter(name, param)]


def dense_parameter_bits(model, fp16=False):
    bits = 16 if fp16 else 32
    return sum(param.numel() * bits for param in model.parameters())


def bits_to_mb(bits):
    return float(bits) / 8.0 / (1024.0 * 1024.0)


def _erk_target_counts(named_params, sparsity):
    total = sum(param.numel() for _, param in named_params)
    target_nonzero = int(round((1.0 - sparsity) * total))
    if target_nonzero <= 0:
        return {name: 0 for name, _ in named_params}
    if target_nonzero >= total:
        return {name: param.numel() for name, param in named_params}

    dense = set()
    epsilon = 0.0
    while True:
        divisor = 0.0
        rhs = target_nonzero
        raw_probs = {}
        for name, param in named_params:
            n = param.numel()
            if name in dense:
                rhs -= n
                continue
            shape = tuple(param.shape)
            raw_prob = sum(shape) / float(np.prod(shape))
            raw_probs[name] = raw_prob
            divisor += raw_prob * n

        if divisor <= 0 or rhs <= 0:
            break
        epsilon = rhs / divisor
        newly_dense = {name for name, prob in raw_probs.items() if epsilon * prob > 1.0}
        if not newly_dense:
            break
        dense.update(newly_dense)

    param_map = dict(named_params)
    counts = {}
    allocated = 0
    for name, param in named_params:
        n = param.numel()
        if name in dense:
            count = n
        else:
            shape = tuple(param.shape)
            raw_prob = sum(shape) / float(np.prod(shape))
            density = min(1.0, max(0.0, epsilon * raw_prob if divisor > 0 else 1.0 - sparsity))
            count = int(round(density * n))
        counts[name] = min(n, max(0, count))
        allocated += counts[name]

    diff = target_nonzero - allocated
    if diff != 0:
        adjustable = [name for name, param in named_params if 0 < counts[name] < param.numel()]
        if not adjustable:
            adjustable = [name for name, _ in named_params]
        idx = 0
        while diff != 0 and adjustable:
            name = adjustable[idx % len(adjustable)]
            n = param_map[name].numel()
            if diff > 0 and counts[name] < n:
                counts[name] += 1
                diff -= 1
            elif diff < 0 and counts[name] > 0:
                counts[name] -= 1
                diff += 1
            idx += 1
            if idx > len(adjustable) * 8:
                break
    return counts


def target_active_counts(model, sparsity, distribution="erk"):
    named_params = [(name, param.detach().cpu()) for name, param in model.named_parameters() if is_masked_parameter(name, param)]
    if not named_params:
        return {}
    sparsity = min(1.0, max(0.0, float(sparsity)))
    if distribution == "uniform":
        return {name: int(round((1.0 - sparsity) * param.numel())) for name, param in named_params}
    if distribution in ("er", "erk"):
        return _erk_target_counts(named_params, sparsity)
    raise ValueError(f"Unsupported FedDST sparsity distribution: {distribution}")


def make_initial_masks(model, sparsity, distribution="erk"):
    counts = target_active_counts(model, sparsity, distribution)
    masks = OrderedDict()
    with torch.no_grad():
        for name, param in model.named_parameters():
            if not is_masked_parameter(name, param):
                continue
            n = param.numel()
            keep = min(n, max(0, int(counts.get(name, n))))
            mask = torch.zeros(n, dtype=torch.bool, device="cpu")
            if keep >= n:
                mask[:] = True
            elif keep > 0:
                scores = param.detach().abs().flatten().cpu()
                _, idx = torch.topk(scores, keep, largest=True)
                mask[idx] = True
            masks[name] = mask.view_as(param).clone()
    return masks


def clone_masks(masks):
    return OrderedDict((name, mask.detach().cpu().clone().bool()) for name, mask in masks.items())


def apply_masks(model, masks):
    if not masks:
        return
    with torch.no_grad():
        for name, param in model.named_parameters():
            mask = masks.get(name)
            if mask is not None:
                param.mul_(mask.to(device=param.device, dtype=param.dtype))


def mask_gradients(model, masks):
    if not masks:
        return
    for name, param in model.named_parameters():
        mask = masks.get(name)
        if mask is not None and param.grad is not None:
            param.grad.mul_(mask.to(device=param.grad.device, dtype=param.grad.dtype))


def sparse_payload_bits(model, masks, include_mask_bits=False, fp16=False):
    bits = 16 if fp16 else 32
    total = 0
    for name, param in model.named_parameters():
        mask = masks.get(name) if masks else None
        if mask is None:
            total += param.numel() * bits
        else:
            total += int(mask.sum().item()) * bits
            if include_mask_bits:
                total += mask.numel()
    return total


def masks_equal(left, right):
    if left is None or right is None:
        return False
    if set(left.keys()) != set(right.keys()):
        return False
    return all(torch.equal(left[name].cpu(), right[name].cpu()) for name in left)


def readjust_masks(model, masks, ratio, target_sparsity, distribution="erk"):
    if ratio <= 0 or not masks:
        return clone_masks(masks)
    new_masks = clone_masks(masks)
    targets = target_active_counts(model, target_sparsity, distribution)

    for name, param in model.named_parameters():
        if name not in new_masks:
            continue
        data = param.detach().cpu().flatten()
        grad = param.grad.detach().cpu().flatten() if param.grad is not None else torch.zeros_like(data)
        mask = new_masks[name].flatten().clone()
        target_active = int(targets.get(name, int(mask.sum().item())))
        target_active = min(mask.numel(), max(0, target_active))

        active_idx = torch.nonzero(mask, as_tuple=False).flatten()
        inactive_idx = torch.nonzero(~mask, as_tuple=False).flatten()

        if active_idx.numel() > target_active:
            n_prune = active_idx.numel() - target_active
        else:
            n_prune = int(round(active_idx.numel() * ratio))
        n_prune = min(n_prune, active_idx.numel(), inactive_idx.numel())

        if n_prune > 0:
            prune_scores = data[active_idx].abs()
            _, prune_local = torch.topk(prune_scores, n_prune, largest=False)
            mask[active_idx[prune_local]] = False

        active_idx = torch.nonzero(mask, as_tuple=False).flatten()
        inactive_idx = torch.nonzero(~mask, as_tuple=False).flatten()
        n_grow = max(0, target_active - active_idx.numel())
        n_grow = min(n_grow, inactive_idx.numel())
        if n_grow > 0:
            grow_scores = grad[inactive_idx].abs()
            if torch.count_nonzero(grow_scores).item() == 0:
                grow_scores = data[inactive_idx].abs()
            _, grow_local = torch.topk(grow_scores, n_grow, largest=True)
            grow_idx = inactive_idx[grow_local]
            mask[grow_idx] = True
            with torch.no_grad():
                flat = param.data.flatten()
                flat[grow_idx.to(param.device)] = 0.0

        new_masks[name] = mask.view_as(new_masks[name])
    apply_masks(model, new_masks)
    return new_masks

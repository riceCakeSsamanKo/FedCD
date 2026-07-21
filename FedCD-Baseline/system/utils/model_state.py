import copy

import torch


def copy_module_state(source, target):
    '''Copy parameters and persistent buffers between compatible modules.'''
    target.load_state_dict(source.state_dict(), strict=True)


def copy_module_buffers(source, target):
    '''Copy only persistent buffers while leaving parameters untouched.'''
    source_buffers = dict(source.named_buffers())
    target_buffers = dict(target.named_buffers())
    if source_buffers.keys() != target_buffers.keys():
        raise ValueError('Source and target modules have different buffer layouts.')
    with torch.no_grad():
        for name, source_buffer in source_buffers.items():
            target_buffers[name].copy_(source_buffer.to(target_buffers[name].device))


def average_module_states(modules, weights, target=None):
    '''Return a weighted average of complete module states, including buffers.'''
    if not modules:
        raise ValueError('At least one module is required for aggregation.')
    if len(modules) != len(weights):
        raise ValueError('The number of modules and aggregation weights must match.')

    total_weight = float(sum(weights))
    if total_weight <= 0:
        raise ValueError('Aggregation weights must have a positive sum.')
    normalized_weights = [float(weight) / total_weight for weight in weights]
    states = [module.state_dict() for module in modules]
    reference_keys = list(states[0].keys())
    for state in states[1:]:
        if list(state.keys()) != reference_keys:
            raise ValueError('All modules must have identical state_dict layouts.')

    averaged_state = {}
    for key in reference_keys:
        reference = states[0][key]
        if torch.is_floating_point(reference) or torch.is_complex(reference):
            value = torch.zeros_like(reference)
            for weight, state in zip(normalized_weights, states):
                value.add_(state[key].to(value.device), alpha=weight)
            averaged_state[key] = value
        else:
            value = torch.zeros_like(reference, dtype=torch.float64)
            for weight, state in zip(normalized_weights, states):
                value.add_(state[key].to(device=value.device, dtype=torch.float64), alpha=weight)
            averaged_state[key] = value.round().to(dtype=reference.dtype)

    averaged = copy.deepcopy(modules[0]) if target is None else target
    averaged.load_state_dict(averaged_state, strict=True)
    return averaged


def blend_module_states(previous, current, current_weight, target=None):
    '''Blend two complete module states with the given current-model weight.'''
    return average_module_states(
        [previous, current],
        [1.0 - float(current_weight), float(current_weight)],
        target=target,
    )

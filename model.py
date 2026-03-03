import torch
from torch.nn import functional as F


def merge_data(data):
    merged_data = []
    start_positions = []
    current_position = 0

    for sequence in data:
        start_positions.append(current_position)
        merged_data.extend(sequence)
        current_position += len(sequence)

    return merged_data, start_positions


def _stack_and_pad(tensors, padding_side):
    tensors = list(tensors)

    if not tensors:
        raise ValueError('At least one sequence tensor is required.')

    if padding_side not in ('left', 'right'):
        raise ValueError('Padding side must be left or right.')

    for tensor in tensors:
        if not isinstance(tensor, torch.Tensor):
            raise TypeError('Every sequence must be a torch.Tensor.')
        if tensor.ndim != 2:
            raise ValueError('Sequence tensors must have shape [length, embedding_size].')

    first_tensor = tensors[0]
    embedding_size = first_tensor.shape[1]
    device = first_tensor.device
    dtype = first_tensor.dtype

    for tensor in tensors:
        if tensor.shape[1] != embedding_size:
            raise ValueError('All sequences must use the same embedding size.')
        if tensor.device != device:
            raise ValueError('All sequence tensors must be on the same device.')
        if tensor.dtype != dtype:
            raise ValueError('All sequence tensors must use the same dtype.')

    maximum_length = max(tensor.shape[0] for tensor in tensors)
    padded_tensors = []
    attention_masks = []

    for tensor in tensors:
        sequence_length = tensor.shape[0]
        padding_length = maximum_length - sequence_length

        if padding_side == 'left':
            padding = (0, 0, padding_length, 0)
        else:
            padding = (0, 0, 0, padding_length)

        padded_tensor = F.pad(tensor, padding)
        valid_positions = torch.ones(sequence_length, dtype=torch.long, device=device)
        padding_positions = torch.zeros(padding_length, dtype=torch.long, device=device)

        if padding_side == 'left':
            attention_mask = torch.cat([padding_positions, valid_positions])
        else:
            attention_mask = torch.cat([valid_positions, padding_positions])

        padded_tensors.append(padded_tensor)
        attention_masks.append(attention_mask)

    batch_embeddings = torch.stack(padded_tensors)
    batch_attention_mask = torch.stack(attention_masks)

    return batch_embeddings, batch_attention_mask


def stack_and_pad_left(tensors):
    return _stack_and_pad(tensors, padding_side='left')


def stack_and_pad_right(tensors):
    return _stack_and_pad(tensors, padding_side='right')

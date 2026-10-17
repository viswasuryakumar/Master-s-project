import math

import torch
from tqdm import tqdm


def print_number_of_trainable_model_parameters(model):
    parameters = list(model.parameters())
    trainable = [parameter for parameter in parameters if parameter.requires_grad]
    print(
        f'All parameters: {sum(parameter.numel() for parameter in parameters)}; '
        f'trainable parameters: {sum(parameter.numel() for parameter in trainable)}'
    )
    return trainable


def _optimizer_step(parameters, optimizer, token_count):
    for parameter in parameters:
        if parameter.grad is not None:
            parameter.grad.div_(token_count)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)


def trainModel(
    model, dataloader, gradient_accumulation_steps=1, n_epochs=1, lr=5e-5,
    device=None,
):
    if gradient_accumulation_steps < 1 or n_epochs < 1:
        raise ValueError('Epochs and accumulation steps must be positive.')
    batches_per_epoch = len(dataloader)
    if batches_per_epoch == 0:
        raise ValueError('Training requires at least one batch.')

    parameters = print_number_of_trainable_model_parameters(model)
    device = torch.device(device if device is not None else model.device)
    optimizer = torch.optim.AdamW(parameters, lr=lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.7)
    updates_per_epoch = math.ceil(batches_per_epoch / gradient_accumulation_steps)
    scheduler_interval = max(1, n_epochs * updates_per_epoch // 10)
    normal = set(model.Llama_tokenizer(
        'The sequence is normal.', add_special_tokens=False,
    )['input_ids'])
    anomalous = set(model.Llama_tokenizer(
        'The sequence is anomalous.', add_special_tokens=False,
    )['input_ids'])
    classification_ids = torch.tensor(
        sorted(normal.symmetric_difference(anomalous)), device=device, dtype=torch.long,
    )
    history = []
    total_updates = 0
    model.train()
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(n_epochs):
        epoch_loss = 0.0
        epoch_tokens = 0
        class_correct = 0
        class_tokens = 0
        pending_batches = 0
        pending_tokens = 0
        epoch_updates = 0
        progress = tqdm(dataloader, desc=f'Epoch {epoch + 1}/{n_epochs}')

        for batch in progress:
            inputs = batch['inputs'].to(device)
            logits, targets = model.train_helper(inputs, batch['seq_positions'], batch['labels'])
            loss_sum = torch.nn.functional.cross_entropy(logits, targets, reduction='sum')
            loss_sum.backward()
            token_count = targets.numel()
            epoch_loss += loss_sum.detach().item()
            epoch_tokens += token_count
            pending_tokens += token_count
            pending_batches += 1

            with torch.no_grad():
                class_mask = torch.isin(targets, classification_ids)
                class_correct += (logits.argmax(dim=-1)[class_mask] == targets[class_mask]).sum().item()
                class_tokens += class_mask.sum().item()

            if pending_batches == gradient_accumulation_steps:
                _optimizer_step(parameters, optimizer, pending_tokens)
                total_updates += 1
                epoch_updates += 1
                if total_updates % scheduler_interval == 0:
                    scheduler.step()
                pending_batches = 0
                pending_tokens = 0
            progress.set_postfix(loss=epoch_loss / epoch_tokens, lr=optimizer.param_groups[0]['lr'])

        if pending_batches:
            _optimizer_step(parameters, optimizer, pending_tokens)
            total_updates += 1
            epoch_updates += 1
            if total_updates % scheduler_interval == 0:
                scheduler.step()
        result = {
            'epoch': epoch + 1,
            'loss': epoch_loss / epoch_tokens,
            'class_token_accuracy': class_correct / class_tokens if class_tokens else None,
            'target_tokens': epoch_tokens,
            'classification_tokens': class_tokens,
            'optimizer_updates': epoch_updates,
            'learning_rate': optimizer.param_groups[0]['lr'],
        }
        history.append(result)
        print(result)

    return history

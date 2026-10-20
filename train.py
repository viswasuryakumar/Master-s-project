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


def main():
    import argparse
    import json
    from pathlib import Path

    import numpy as np
    from torch.utils.data import DataLoader
    from customDataset import BalancedSampler, CustomCollator, CustomDataset
    from model import LogAnomalyModel

    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description='Train log anomaly adapters in four stages.')
    parser.add_argument('--dataset', choices=['BGL', 'HDFS', 'Liberty', 'Thunderbird'], default='Liberty')
    parser.add_argument('--data', type=Path)
    parser.add_argument('--bert-path', type=Path, default=root / 'models/bert-base-uncased')
    parser.add_argument('--llama-path', type=Path, default=root / 'models/Meta-Llama-3-8B')
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda', 'mps'], default='auto')
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--accumulation-steps', type=int, default=4)
    parser.add_argument('--epochs', type=int, nargs=4, default=[1, 1, 1, 2])
    parser.add_argument('--learning-rates', type=float, nargs=4, default=[5e-4, 5e-4, 5e-5, 5e-5])
    parser.add_argument('--minority-ratio', type=float, default=0.3)
    parser.add_argument('--warmup-samples', type=int, default=1000)
    parser.add_argument('--max-samples', type=int, default=0)
    parser.add_argument('--max-content-len', type=int, default=100)
    parser.add_argument('--max-seq-len', type=int, default=128)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if min(args.batch_size, args.accumulation_steps, args.warmup_samples) < 1 or args.max_samples < 0:
        parser.error('Batch settings must be positive; max samples must be nonnegative.')
    if min(args.epochs) < 0 or not any(args.epochs) or not 0 < args.minority_ratio < 1:
        parser.error('Choose at least one training epoch and a minority ratio between zero and one.')

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device_name = args.device
    if device_name == 'auto':
        device_name = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    device = torch.device(device_name)
    data_path = args.data or root / 'data' / args.dataset / 'train.csv'
    output_dir = args.output_dir or root / 'outputs' / 'training' / args.dataset
    dataset = CustomDataset(data_path, drop_duplicates=False)
    if set(dataset.get_label()) != {0, 1}:
        parser.error('Balanced training requires both normal and anomalous sequences.')
    model = LogAnomalyModel(
        str(args.bert_path), str(args.llama_path),
        ft_path=str(args.checkpoint) if args.checkpoint else None,
        is_train_mode=True, device=device,
        max_content_len=args.max_content_len, max_seq_len=args.max_seq_len,
    )
    collator = CustomCollator(
        model.Bert_tokenizer, max_seq_len=args.max_seq_len,
        max_content_len=args.max_content_len,
    )

    def make_loader(limit):
        sampler = BalancedSampler(dataset, target_ratio=args.minority_ratio, min_samples=0)
        if limit:
            sampler = BalancedSampler(
                dataset, target_ratio=args.minority_ratio,
                max_samples=min(limit, len(sampler)), min_samples=0,
            )
        return DataLoader(
            dataset, batch_size=args.batch_size, sampler=sampler,
            collate_fn=collator, num_workers=0, drop_last=False,
        )

    stages = [
        ('llama', model.set_train_only_Llama),
        ('projector', model.set_train_only_projector),
        ('projector_bert', model.set_train_projectorAndBert),
        ('all', model.set_finetuning_all),
    ]
    completed = []
    for index, ((name, select), epochs, learning_rate) in enumerate(zip(stages, args.epochs, args.learning_rates)):
        if epochs == 0:
            continue
        select()
        limit = args.max_samples
        if index == 0:
            limit = min(args.warmup_samples, limit) if limit else args.warmup_samples
        print(f'Training stage: {name}; device: {device}; epochs: {epochs}')
        history = trainModel(
            model, make_loader(limit), args.accumulation_steps, epochs,
            learning_rate, device=device,
        )
        completed.append({'stage': name, 'history': history})
    model.save_ft_model(output_dir)
    metadata = {**vars(args), 'device': str(device), 'data': str(data_path.resolve()), 'stages': completed}
    (output_dir / 'training.json').write_text(json.dumps(metadata, indent=2, default=str) + '\n')
    print(f'Saved adapters, projection weights and training history to {output_dir}')


if __name__ == '__main__':
    main()

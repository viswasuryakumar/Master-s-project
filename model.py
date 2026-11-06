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



def build_quantization_config(compute_dtype=torch.bfloat16):
    from transformers import BitsAndBytesConfig

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=False,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_compute_dtype=compute_dtype,
    )


class LogAnomalyModel(torch.nn.Module):
    def __init__(
        self, Bert_path, Llama_path, ft_path=None, is_train_mode=True,
        device=torch.device('cuda:0'), max_content_len=128, max_seq_len=128,
        max_new_tokens=6,
    ):
        super().__init__()

        if max_content_len <= 0 or max_seq_len <= 0:
            raise ValueError('Content and sequence length limits must be positive.')

        if isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, int):
            raise TypeError('Generation length must be an integer.')
        if max_new_tokens <= 0:
            raise ValueError('Generation length must be positive.')

        self.Bert_path = Bert_path
        self.Llama_path = Llama_path
        self.ft_path = ft_path
        self.is_train_mode = is_train_mode
        self.max_content_len = max_content_len
        self.max_seq_len = max_seq_len
        self.max_new_tokens = max_new_tokens
        self.device = torch.device(device)

        self.load_tokenizers()
        self.load_models()
        self.initialize_projection()
        self.load_adapters()

    def load_tokenizers(self):
        from transformers import AutoTokenizer, BertTokenizerFast

        self.Llama_tokenizer = AutoTokenizer.from_pretrained(
            self.Llama_path, padding_side='right',
        )
        if self.Llama_tokenizer.eos_token is None:
            raise ValueError('The Llama tokenizer must define an end-of-sequence token.')
        self.Llama_tokenizer.pad_token = self.Llama_tokenizer.eos_token

        self.Bert_tokenizer = BertTokenizerFast.from_pretrained(
            self.Bert_path, do_lower_case=True,
        )

    def load_models(self):
        from transformers import AutoModelForCausalLM, BertModel

        self.model_dtype = (
            torch.float32 if self.device.type == 'cpu' else torch.float16
        )
        load_options = {
            'low_cpu_mem_usage': True,
            'device_map': (
                self.device if self.device.type == 'cpu'
                else {'': str(self.device)}
            ),
            'torch_dtype': self.model_dtype,
        }
        if self.device.type in ('cuda', 'mps'):
            compute_dtype = (
                torch.float16 if self.device.type == 'mps' else torch.bfloat16
            )
            load_options['quantization_config'] = build_quantization_config(
                compute_dtype=compute_dtype,
            )

        self.Llama_model = AutoModelForCausalLM.from_pretrained(
            self.Llama_path, **load_options,
        )
        self.Bert_model = BertModel.from_pretrained(
            self.Bert_path, **load_options,
        )

    def initialize_projection(self):
        self.projector = torch.nn.Linear(
            self.Bert_model.config.hidden_size,
            self.Llama_model.config.hidden_size,
            device=self.device,
            dtype=torch.float32,
        )
        self.instruc_tokens = self.Llama_tokenizer(
            [
                'Below is a sequence of system log messages:',
                '. Is this sequence normal or anomalous? \\n',
            ],
            return_tensors='pt', padding=True,
        ).to(self.device)

    def load_adapters(self):
        from pathlib import Path
        from peft import LoraConfig, PeftModel, TaskType, get_peft_model

        if self.ft_path is None:
            bert_config = LoraConfig(
                task_type=TaskType.FEATURE_EXTRACTION,
                r=4, lora_alpha=32, lora_dropout=0.01,
            )
            llama_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=8, lora_alpha=16, lora_dropout=0.1,
                target_modules=['q_proj', 'v_proj'], bias='none',
            )
            self.Bert_model = get_peft_model(self.Bert_model, bert_config)
            self.Llama_model = get_peft_model(self.Llama_model, llama_config)
        else:
            checkpoint = Path(self.ft_path)
            self.Bert_model = PeftModel.from_pretrained(
                self.Bert_model, str(checkpoint / 'Bert_ft'),
                is_trainable=self.is_train_mode,
                torch_dtype=self.model_dtype,
            )
            self.Llama_model = PeftModel.from_pretrained(
                self.Llama_model, str(checkpoint / 'Llama_ft'),
                is_trainable=self.is_train_mode,
                torch_dtype=self.model_dtype,
            )
            projector_state = torch.load(
                checkpoint / 'projector.pt',
                map_location=self.device, weights_only=True,
            )
            self.projector.load_state_dict(projector_state)

    def save_ft_model(self, path):
        from pathlib import Path

        destination = Path(path)
        destination.mkdir(parents=True, exist_ok=True)
        self.Llama_model.save_pretrained(
            destination / 'Llama_ft', safe_serialization=True,
        )
        self.Bert_model.save_pretrained(
            destination / 'Bert_ft', safe_serialization=True,
        )
        torch.save(self.projector.state_dict(), destination / 'projector.pt')

    def _set_trainable_components(self, projector, bert, llama):
        for parameter in self.projector.parameters():
            parameter.requires_grad = projector
        for name, parameter in self.Bert_model.named_parameters():
            parameter.requires_grad = bert and 'lora_' in name
        for name, parameter in self.Llama_model.named_parameters():
            parameter.requires_grad = llama and 'lora_' in name

    def set_train_only_projector(self):
        self._set_trainable_components(projector=True, bert=False, llama=False)

    def set_train_only_Llama(self):
        self._set_trainable_components(projector=False, bert=False, llama=True)

    def set_train_projectorAndBert(self):
        self._set_trainable_components(projector=True, bert=True, llama=False)

    def set_finetuning_all(self):
        self._set_trainable_components(projector=True, bert=True, llama=True)

    def train_helper(self, inputs, seq_positions, labels):
        labels = [str(label) for label in labels]
        if not labels or any(label not in ('normal', 'anomalous') for label in labels):
            raise ValueError('Training labels must be normal or anomalous.')
        if len(seq_positions) + 1 != len(labels):
            raise ValueError('Sequence boundaries and labels must describe the same batch.')

        embedding_layer = self.Llama_model.get_input_embeddings()
        encoded = self.Bert_model(**inputs).pooler_output
        projected = self.projector(encoded.to(dtype=self.projector.weight.dtype))
        sequences = torch.tensor_split(
            projected.to(dtype=embedding_layer.weight.dtype), seq_positions,
        )
        instruction_embeddings = embedding_layer(self.instruc_tokens['input_ids'])
        prefix_mask = self.instruc_tokens['attention_mask'][0].bool()
        question_mask = self.instruc_tokens['attention_mask'][1].bool()
        prefix = instruction_embeddings[0][prefix_mask]
        question = instruction_embeddings[1][question_mask]
        question_ids = self.instruc_tokens['input_ids'][1][question_mask]
        bos_token_id = self.Llama_tokenizer.bos_token_id
        if bos_token_id is not None and question_ids[0].item() == bos_token_id:
            question = question[1:]

        answer_tokens = self.Llama_tokenizer(
            [f'The sequence is {label}.' for label in labels],
            padding=True, return_tensors='pt', add_special_tokens=False,
        ).to(self.device)
        eos = torch.tensor(
            [self.Llama_tokenizer.eos_token_id], dtype=torch.long, device=self.device,
        )
        embeddings = []
        targets = []
        prompt_lengths = []
        for index, sequence in enumerate(sequences):
            answer_mask = answer_tokens['attention_mask'][index].bool()
            answer_ids = answer_tokens['input_ids'][index][answer_mask]
            prompt = torch.cat([prefix, sequence, question])
            embeddings.append(torch.cat([prompt, embedding_layer(answer_ids)]))
            targets.append(torch.cat([answer_ids, eos]))
            prompt_lengths.append(prompt.shape[0])

        inputs_embeds, attention_mask = stack_and_pad_left(embeddings)
        logits = self.Llama_model(
            inputs_embeds=inputs_embeds, attention_mask=attention_mask,
        ).logits
        selected_logits = []
        for index, prompt_length in enumerate(prompt_lengths):
            padding_length = inputs_embeds.shape[1] - embeddings[index].shape[0]
            start = padding_length + prompt_length - 1
            end = padding_length + embeddings[index].shape[0]
            selected_logits.append(logits[index, start:end])
        return torch.cat(selected_logits), torch.cat(targets)

    def forward(self, inputs, seq_positions):
        from transformers import DynamicCache

        embedding_layer = self.Llama_model.get_input_embeddings()
        encoded = self.Bert_model(**inputs).pooler_output
        projected = self.projector(encoded.to(dtype=self.projector.weight.dtype))
        sequences = torch.tensor_split(
            projected.to(dtype=embedding_layer.weight.dtype), seq_positions,
        )
        instruction_embeddings = embedding_layer(self.instruc_tokens['input_ids'])
        prefix_mask = self.instruc_tokens['attention_mask'][0].bool()
        question_mask = self.instruc_tokens['attention_mask'][1].bool()
        prefix = instruction_embeddings[0][prefix_mask]
        question = instruction_embeddings[1][question_mask]
        question_ids = self.instruc_tokens['input_ids'][1][question_mask]
        bos_token_id = self.Llama_tokenizer.bos_token_id
        if bos_token_id is not None and question_ids[0].item() == bos_token_id:
            question = question[1:]

        answer_prefix_ids = self.Llama_tokenizer(
            'The sequence is', add_special_tokens=False, return_tensors='pt',
        )['input_ids'][0].to(self.device)
        answer_prefix = embedding_layer(answer_prefix_ids)
        prompts = [
            torch.cat([prefix, sequence, question, answer_prefix])
            for sequence in sequences
        ]
        inputs_embeds, attention_mask = stack_and_pad_left(prompts)
        batch_size = len(sequences)
        unfinished = torch.ones(batch_size, dtype=torch.bool, device=self.device)
        eos_tokens = torch.as_tensor(
            self.Llama_tokenizer.eos_token_id, device=self.device,
        ).reshape(-1)
        pad_token_id = self.Llama_tokenizer.pad_token_id
        cache = DynamicCache()
        answer = []

        for _ in range(self.max_new_tokens):
            position_ids = attention_mask.long().cumsum(dim=-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 0)
            if cache.get_seq_length() == 0:
                current_embeddings = inputs_embeds
            else:
                current_embeddings = embedding_layer(next_tokens).unsqueeze(1)
                position_ids = position_ids[:, -1:]

            outputs = self.Llama_model(
                inputs_embeds=current_embeddings,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=cache,
                use_cache=True,
            )
            cache = outputs.past_key_values
            next_tokens = outputs.logits[:, -1, :].argmax(dim=-1)
            next_tokens = torch.where(unfinished, next_tokens, pad_token_id)
            answer.append(next_tokens)
            attention_mask = torch.cat(
                [attention_mask, unfinished.long().unsqueeze(1)], dim=1,
            )
            unfinished = unfinished & (
                next_tokens.unsqueeze(1) != eos_tokens.unsqueeze(0)
            ).all(dim=1)
            if not unfinished.any():
                break

        return torch.stack(answer, dim=1)

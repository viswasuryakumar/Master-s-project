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



def build_quantization_config():
    from transformers import BitsAndBytesConfig

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=False,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_compute_dtype=torch.bfloat16,
    )


class LogAnomalyModel(torch.nn.Module):
    def __init__(
        self, Bert_path, Llama_path, ft_path=None, is_train_mode=True,
        device=torch.device('cuda:0'), max_content_len=128, max_seq_len=128,
    ):
        super().__init__()

        if max_content_len <= 0 or max_seq_len <= 0:
            raise ValueError('Content and sequence length limits must be positive.')

        self.Bert_path = Bert_path
        self.Llama_path = Llama_path
        self.ft_path = ft_path
        self.is_train_mode = is_train_mode
        self.max_content_len = max_content_len
        self.max_seq_len = max_seq_len
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
            'device_map': self.device,
            'torch_dtype': self.model_dtype,
        }
        if self.device.type == 'cuda':
            load_options['quantization_config'] = build_quantization_config()

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
                '. Is this sequence normal or anomalous? \n',
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

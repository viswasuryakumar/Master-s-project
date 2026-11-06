"""Capture small, real intermediate values for a presentation, without changing outputs."""
import json
from pathlib import Path


class PipelineTrace:
    def __init__(self, model, batch):
        self.model = model
        self.data = {"scope": "First batch only; numerical excerpts are not explanations or confidence scores."}
        inputs = batch['inputs']
        self.data['bert_input'] = {
            'shape': list(inputs['input_ids'].shape),
            'input_ids_first_message': inputs['input_ids'][0].tolist(),
            'attention_mask_first_message': inputs['attention_mask'][0].tolist(),
            'tokens_first_message': model.Bert_tokenizer.convert_ids_to_tokens(inputs['input_ids'][0].tolist()),
            'sequence_split_positions': batch['seq_positions'].tolist(),
        }
        self.handles = [
            model.Bert_model.register_forward_hook(self.bert),
            model.projector.register_forward_hook(self.projector),
            model.Llama_model.register_forward_pre_hook(self.llama_input, with_kwargs=True),
        ]

    @staticmethod
    def summary(tensor):
        return {'shape': list(tensor.shape), 'dtype': str(tensor.dtype),
                'first_eight_values': tensor.detach().reshape(-1)[:8].float().cpu().tolist()}

    def bert(self, module, args, output):
        self.data['bert_output'] = self.summary(output.pooler_output)

    def projector(self, module, args, output):
        self.data['projector_output'] = self.summary(output)

    def llama_input(self, module, args, kwargs):
        if 'llama_input' not in self.data:
            self.data['llama_input'] = self.summary(kwargs['inputs_embeds'])
            self.data['llama_input']['attention_mask_shape'] = list(kwargs['attention_mask'].shape)
            self.data['llama_input']['composition'] = 'instruction embeddings + projected message vectors + question embeddings + answer-prefix embeddings'

    def finish(self, output_ids, responses):
        self.data['generated_token_ids'] = output_ids.detach().cpu().tolist()
        self.data['responses'] = responses

    def close(self):
        for handle in self.handles:
            handle.remove()

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.data, indent=2) + '\n')

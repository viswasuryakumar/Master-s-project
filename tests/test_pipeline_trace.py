"""The presentation observer must not change model values or retain hooks."""
import unittest
from types import SimpleNamespace

import torch
from pipeline_trace import PipelineTrace


class TraceTests(unittest.TestCase):
    def test_observer_preserves_outputs_and_detaches(self):
        class Encoder(torch.nn.Module):
            def forward(self, input_ids):
                return SimpleNamespace(pooler_output=input_ids.float())

        class Decoder(torch.nn.Module):
            def forward(self, inputs_embeds, attention_mask):
                return inputs_embeds.sum(-1)

        model = SimpleNamespace(Bert_model=Encoder(), projector=torch.nn.Linear(3, 4),
                                Llama_model=Decoder(), Bert_tokenizer=SimpleNamespace(
                                    convert_ids_to_tokens=lambda ids: [str(i) for i in ids]))
        batch = {'inputs': {'input_ids': torch.tensor([[1, 2, 3], [4, 5, 6]]),
                            'attention_mask': torch.ones(2, 3, dtype=torch.long)},
                 'seq_positions': torch.tensor([], dtype=torch.long)}

        def forward():
            encoded = model.Bert_model(batch['inputs']['input_ids']).pooler_output
            projected = model.projector(encoded).unsqueeze(0)
            return model.Llama_model(inputs_embeds=projected, attention_mask=torch.ones(1, 2))

        before = forward()
        trace = PipelineTrace(model, batch)
        try:
            observed = forward()
        finally:
            trace.close()
        self.assertTrue(torch.equal(before, observed))
        self.assertEqual(trace.data['bert_output']['shape'], [2, 3])
        self.assertEqual(trace.data['llama_input']['shape'], [1, 2, 4])
        self.assertFalse(model.Bert_model._forward_hooks)
        self.assertFalse(model.projector._forward_hooks)
        self.assertFalse(model.Llama_model._forward_pre_hooks)


if __name__ == '__main__':
    unittest.main()

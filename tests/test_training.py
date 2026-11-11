import contextlib
import copy
import io
import math

import pytest

torch = pytest.importorskip('torch')
from train import trainModel


class BatchInputs(dict):
    def to(self, device):
        return BatchInputs({name: value.to(device) for name, value in self.items()})


class TinyTrainingModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(2, 3)
        self.device = torch.device('cpu')
        self.distinct_class_tokens = True

    def Llama_tokenizer(self, text, **options):
        if not self.distinct_class_tokens:
            return {'input_ids': [0]}
        return {'input_ids': [0, 1 if ' normal.' in text else 2]}

    def train_helper(self, inputs, positions, labels):
        return self.layer(inputs['features']), torch.tensor(labels, dtype=torch.long)


@pytest.fixture
def model():
    torch.manual_seed(17)
    return TinyTrainingModel()


@pytest.fixture
def samples():
    return [[1.0, 2.0], [2.0, -1.0], [0.5, 3.0], [-2.0, 1.0], [1.0, -1.0]], [1, 2, 1, 2, 1]


def batch(features, labels):
    return {
        'inputs': BatchInputs(features=torch.tensor(features, dtype=torch.float32)),
        'seq_positions': [],
        'labels': labels,
    }


def train(model, batches, **settings):
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        return trainModel(model, batches, lr=0.01, **settings)


def assert_same_parameters(left, right):
    for first, second in zip(left.parameters(), right.parameters()):
        torch.testing.assert_close(first, second, rtol=1e-5, atol=1e-7)


def test_unequal_microbatches_match_one_full_batch(model, samples):
    features, labels = samples
    full = copy.deepcopy(model)
    accumulated = copy.deepcopy(model)
    expected = train(full, [batch(features, labels)])
    actual = train(accumulated, [
        batch(features[:1], labels[:1]),
        batch(features[1:3], labels[1:3]),
        batch(features[3:], labels[3:]),
    ], gradient_accumulation_steps=4)
    assert_same_parameters(full, accumulated)
    assert actual[0]['loss'] == pytest.approx(expected[0]['loss'])
    assert actual[0]['class_token_accuracy'] == expected[0]['class_token_accuracy']
    assert actual[0]['target_tokens'] == 5
    assert actual[0]['optimizer_updates'] == 1


@pytest.mark.parametrize('epochs', [1, 2])
def test_partial_epoch_windows_match_explicit_optimizer_updates(model, samples, epochs):
    features, labels = samples
    expected = copy.deepcopy(model)
    actual = copy.deepcopy(model)
    # Use full batches as the reference for accumulated microbatch updates.
    optimizer = torch.optim.AdamW(expected.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.7)
    for _ in range(epochs):
        for start in range(0, len(features), 2):
            inputs = torch.tensor(features[start:start + 2], dtype=torch.float32)
            targets = torch.tensor(labels[start:start + 2], dtype=torch.long)
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.cross_entropy(expected.layer(inputs), targets)
            loss.backward()
            optimizer.step()
            scheduler.step()
    history = train(actual, [batch([item], [label]) for item, label in zip(features, labels)],
                    gradient_accumulation_steps=2, n_epochs=epochs)
    assert_same_parameters(expected, actual)
    assert len(history) == epochs
    assert [row['optimizer_updates'] for row in history] == [3] * epochs
    assert history[-1]['learning_rate'] == pytest.approx(0.01 * 0.7 ** (3 * epochs))


def test_frozen_parameters_are_preserved(model, samples):
    features, labels = samples
    model.layer.bias.requires_grad_(False)
    frozen = model.layer.bias.detach().clone()
    original_weight = model.layer.weight.detach().clone()
    train(model, [batch(features, labels)])
    assert torch.equal(model.layer.bias, frozen)
    assert model.layer.bias.grad is None
    assert not torch.equal(model.layer.weight, original_weight)


def test_training_enables_train_mode_and_clears_gradients(model, samples):
    features, labels = samples
    model.eval()
    train(model, [batch(features, labels)])
    assert model.training
    assert all(parameter.grad is None for parameter in model.parameters())


def test_class_accuracy_is_undefined_without_distinct_label_tokens(model, samples):
    features, labels = samples
    model.distinct_class_tokens = False
    history = train(model, [batch(features, labels)])
    assert history[0]['class_token_accuracy'] is None
    assert history[0]['classification_tokens'] == 0
    assert math.isfinite(history[0]['loss'])


def test_loss_is_weighted_by_target_tokens(model, samples):
    features, labels = samples
    with torch.no_grad():
        expected = torch.nn.functional.cross_entropy(
            model.layer(torch.tensor(features)), torch.tensor(labels),
        ).item()
    history = train(model, [batch(features[:1], labels[:1]), batch(features[1:], labels[1:])],
                    gradient_accumulation_steps=2)
    assert history[0]['loss'] == pytest.approx(expected)


@pytest.mark.parametrize('settings', [{'n_epochs': 0}, {'gradient_accumulation_steps': 0}])
def test_zero_training_steps_fail_before_parameters_change(model, samples, settings):
    features, labels = samples
    original = copy.deepcopy(model)
    with pytest.raises(ValueError):
        train(model, [batch(features, labels)], **settings)
    assert_same_parameters(model, original)


def test_empty_training_data_preserves_parameters(model):
    original = copy.deepcopy(model)
    with pytest.raises(ValueError):
        train(model, [])
    assert_same_parameters(model, original)


def test_no_trainable_parameters_reports_an_error(model, samples):
    features, labels = samples
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    with pytest.raises(ValueError):
        train(model, [batch(features, labels)])

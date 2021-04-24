import numpy as np
from flax import linen as nn
from flax import optim
from jax import random, numpy as jnp
import functools
from typing import Tuple, Any
import jax

symbols = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ' ', '+']


class EncoderLSTM(nn.Module):
    @functools.partial(
        nn.transforms.scan,
        variable_broadcast='params',
        split_rngs={'params': False})
    @nn.compact
    def __call__(self, carry, x):
        return nn.LSTMCell()(carry, x)

    @staticmethod
    def initialize_carry(hidden_size):
        return nn.LSTMCell.initialize_carry(jax.random.PRNGKey(0), (), hidden_size)


class DecoderLSTM(nn.Module):

    @functools.partial(
        nn.transforms.scan,
        variable_broadcast='params',
        split_rngs={'params': False})
    @nn.compact
    def __call__(self, carry, x):
        rng, lstm_state, last_prediction = carry
        carry_rng, categorical_rng = jax.random.split(rng, 2)
        x = last_prediction
        lstm_state, y = nn.LSTMCell()(lstm_state, x)
        logits = nn.Dense(features=len(symbols))(y)
        max_number = jax.random.categorical(categorical_rng, logits)
        prediction = jnp.array(
            max_number == jnp.arange(len(symbols)), dtype=jnp.float32)
        return (carry_rng, lstm_state, prediction), (logits, prediction)


class Decoder(nn.Module):
    init_state: Tuple[Any]

    @nn.compact
    def __call__(self, inputs):
        lstm = DecoderLSTM()
        first_token = jax.lax.slice_in_dim(inputs, 0, 1)[0]
        init_carry = (self.make_rng('lstm'), self.init_state, first_token)
        _, (logits, predictions) = lstm(init_carry, inputs)
        return logits, predictions


class Seq2seq(nn.Module):
    hidden_size: int

    @nn.compact
    def __call__(self, encoder_inputs, decoder_inputs):
        # Encoder
        encoder = EncoderLSTM()
        init_carry = encoder.initialize_carry(self.hidden_size)
        init_decoder_state, _ = encoder(init_carry, encoder_inputs)
        # Decoder
        decoder_inputs = jax.lax.slice_in_dim(decoder_inputs, 0, -1)
        decoder = Decoder(
            init_state=init_decoder_state)
        logits, predictions = decoder(decoder_inputs)

        return logits, predictions


def model():
    return Seq2seq(hidden_size=150)


IN_SHAPES = [{'data': '(n, _)', 'label': '(m + 2, _)'}]
OUT_ELEM = f'(m + 1, {len(symbols)})'
OUT_SHAPE = (OUT_ELEM, OUT_ELEM)


def apply(batch, in_masks, out_masks, params, key):
    @functools.partial(jax.mask, in_shapes=IN_SHAPES, out_shape=OUT_SHAPE)
    def apply_fn(x):
        logits, prediction = model().apply(
            {'params': params},
            x['data'],
            x['label'],
            rngs={'lstm': key})

        return logits, prediction

    return jax.vmap(apply_fn)([batch], dict(n=in_masks, m=out_masks))


def get_initial_params(key, max_length):
    encoder_shape = jnp.ones((max_input_length(max_length), len(symbols)), jnp.float32)
    decoder_shape = jnp.ones((max_output_length(max_length), len(symbols)), jnp.float32)
    initial_params = model().init({"params": key, "lstm": key},
                                  encoder_shape, decoder_shape)['params']
    return initial_params


def create_optimizer(params):
    optimizer_def = optim.Adam(learning_rate=0.003)
    optimizer = optimizer_def.create(params)
    return optimizer


def mask(seq_batch, lengths):
    return seq_batch * (
            lengths[:, np.newaxis] > np.arange(seq_batch.shape[1])
    )


def cross_entropy_loss(logits, labels, length):
    x = jnp.sum(nn.log_softmax(logits) * labels, axis=-1)
    masked_x = -jnp.mean(mask(x, length))
    return masked_x


def compute_metrics(logits, labels, length):
    loss = cross_entropy_loss(logits, labels, length)
    t_accuracy = jnp.argmax(logits, -1) == jnp.argmax(labels, -1)
    seq_accuracy = (
            jnp.sum(mask(t_accuracy, length), axis=-1) == length
    )
    accuracy = jnp.mean(seq_accuracy)
    metrics = {
        "loss": loss,
        "accuracy": accuracy
    }
    return metrics


@jax.jit
def train_step(optimizer, key, batch, in_mask, out_mask):
    labels = batch["label"][:, 1:]

    def loss_fn(params):
        logits, _ = apply(batch, in_mask, out_mask, params, key)
        loss = cross_entropy_loss(logits, labels, out_mask)
        return loss, logits

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (_, logits), grad = grad_fn(optimizer.target)
    optimizer = optimizer.apply_gradient(grad)
    metrics = compute_metrics(logits, labels, out_mask - 1)
    return optimizer, metrics


def train_epoch(optimizer, batch_size, batch, epoch, in_mask, out_mask):
    data_length = len(batch["data"])
    step_per_epoch = data_length // batch_size
    batch_metrics = []
    for _ in range(step_per_epoch):
        key = random.PRNGKey(0)
        optimizer, metrics = train_step(optimizer, key, batch, in_mask, out_mask - 1)
        batch_metrics.append(metrics)

    epoch_metrics = {
        k: np.mean([metrics[k] for metrics in batch_metrics])
        for k in batch_metrics[0]
    }

    print("train epoch: {}, loss: {}, accuracy: {}".format(
        epoch, epoch_metrics["loss"], epoch_metrics["accuracy"] * 100
    ))
    return optimizer, epoch_metrics


def train_and_evaluate(learning_config: dict, data_generation_config: dict):
    key = random.PRNGKey(0)
    params = get_initial_params(key, data_generation_config["max_number"])
    optimizer = create_optimizer(params)

    for epoch in range(1, learning_config["num_epochs"] + 1):
        batch = genarate(
            data_generation_config["symbols"],
            data_generation_config["max_number"],
            data_generation_config["max_sample"],
            data_generation_config["max_length"]
        )
        in_mask, out_mask = return_length()
        optimizer, train_metrics = train_epoch(
            optimizer,
            learning_config["batch_size"],
            batch,
            epoch,
            in_mask,
            out_mask
        )
    print("Training successfully finished")

    return optimizer


class Data_generation():
    RANDOM_DATA = None
    DATA_LENGTH = None
    LABEL_LENGTH = None

    def __init__(self, symbols, max_number, max_sample, max_random_data_length):
        self.symbols = symbols
        self.max_number = max_number
        self.max_sample = max_sample
        self.max_random_data_length = max_random_data_length

    def random_generate(self):
        random_labels = []
        random_data = []
        data_length = []
        label_length = []
        for _ in range(max_sample):
            num1 = np.random.randint(1, max_number)
            num2 = np.random.randint(1, max_number)
            sum = str(num1 + num2)
            label_length.append(len(sum))
            if len(sum) < len(str(max_number)):
                sum += ''.join(' ' for _ in range(len(str(max_number)) - len(sum)))
            data_gen = str(num1) + "+" + str(num2)
            data_length.append(len(data_gen))
            if len(data_gen) < max_input_length(max_number):
                data_gen += ''.join(' ' for _ in range(max_input_length(max_number) - len(data_gen)))
            random_data.append(data_gen)
            random_labels.append(sum)
        Data_generation.RANDOM_DATA = random_data
        Data_generation.DATA_LENGTH = np.array(data_length)
        Data_generation.LABEL_LENGTH = np.array(label_length)
        return random_data, random_labels

    def encode(self, x, y):
        encoded_data = []
        encoded_labels = []
        for operations in x:
            int_data = [symbols.index(value) for value in operations]
            encoded_data.append(int_data)
        for number in y:
            int_label = [symbols.index(value) for value in number]
            encoded_labels.append(int_label)

        return encoded_data, encoded_labels

    def one_hot(self, x, y):
        one_hot_label = []
        one_hot_data = []
        for value in x:
            temp = []
            for i, j in enumerate(value):
                zeros = np.zeros(len(self.symbols))
                zeros[j] = 1
                temp.append(zeros)
            one_hot_data.append(temp)

        for value in y:
            temp = []
            for i, j in enumerate(value):
                zeros = np.zeros(len(self.symbols))
                zeros[j] = 1.
                temp.append(zeros)
            one_hot_label.append(temp)

        x = np.array(one_hot_data)
        y = np.array(one_hot_label)
        batch = {
            "data": x,
            "label": y
        }
        return batch

    def process(self):
        x, y = self.random_generate()
        x, y = self.encode(x, y)
        batch = self.one_hot(x, y)
        return batch


def genarate(symbols, max_number, max_sample, max_random_data_length):
    return Data_generation(symbols, max_number, max_sample
                           , max_random_data_length).process()


def max_input_length(max_number):
    return len(2 * str(max_number)) + 1


def max_output_length(max_number):
    return len(str(max_number))


def return_format():
    return Data_generation.RANDOM_DATA


def return_length():
    return Data_generation.DATA_LENGTH, Data_generation.LABEL_LENGTH


def test_preprocess(x, symbols):
    int_data = dict((i, j) for i, j in enumerate(symbols))
    temp = []
    for value in x:
        string = int_data[np.argmax(value)]
        temp.append(string)
    return ''.join(temp)


max_number = 100
max_sample = 400
epochs = 200
batch_size = 100
max_length = max_input_length(max_number)

learning_config = {
    "num_epochs": epochs,
    "batch_size": batch_size
}

data_generation_config = {
    "symbols": symbols,
    "max_number": max_number,
    "max_sample": max_sample,
    "max_length": max_length
}

train_and_evaluate(learning_config=learning_config, data_generation_config=data_generation_config)

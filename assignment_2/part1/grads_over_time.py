"""
!!!!!
This one does not really work and I am not sure what is asked for in this question.
, in the report I reported the gradients over W_hh, which seems to make more sense.
"""

import argparse
import csv

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from dataset import PalindromeDataset
from vanilla_rnn import VanillaRNN
from lstm import LSTM


def main_grads(config):
    assert config.model_type in ('RNN', 'LSTM')

    # Initialize the device which to run the model on
    device = torch.device(config.device)
    for seq in range(config.input_length-1, config.input_length):
        # Initialize the model that we are going to use
        if config.model_type == "RNN":
            model_def = VanillaRNN
        else:
            model_def = LSTM
        model = model_def(seq, config.input_dim,
                          config.num_hidden, config.num_classes,
                          config.device).to(device)  # fixme

        # Initialize the dataset and data loader (note the +1)
        dataset = PalindromeDataset(seq + 1)
        data_loader = DataLoader(dataset, config.batch_size, num_workers=1)
        batch_inputs, batch_targets = next(iter(data_loader))
        batch_inputs.requires_grad_(True)
        # Setup the loss and optimizer
        criterion = nn.CrossEntropyLoss()  # fixme
        optimizer = optim.RMSprop(model.parameters(), config.learning_rate)  # fixme

        hidden_state, model_outputs = model.forward(batch_inputs)
        # retain grad before doing actual backward pass
        loss = criterion(model_outputs, batch_targets)

        # model.zero_grad()

        # optimizer.step()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.max_norm)
        grads_file = "results/{}_grad_hidden_state_seq.txt".format(config.model_type)
        with open(grads_file, 'w+') as fd:
            writer = csv.writer(fd)
            for i, x in enumerate(model.hidden_states):
                print(x.grad.abs().mean().item())
                writer.writerow([i, x.grad.abs().mean().item()])
            # for x in model.hidden_weights:
            #     # print("")
            #     print(x.grad.abs().mean().item())
            #     writer.writerow([x.grad.abs().mean().item()])


if __name__ == "__main__":
    # Parse training configuration
    parser = argparse.ArgumentParser()

    # Model params
    parser.add_argument('--model_type', type=str, default="RNN", help="Model type, should be 'RNN' or 'LSTM'")
    parser.add_argument('--input_length', type=int, default=100, help='Length of an input sequence')
    parser.add_argument('--input_dim', type=int, default=1, help='Dimensionality of input sequence')
    parser.add_argument('--num_classes', type=int, default=10, help='Dimensionality of output sequence')
    parser.add_argument('--num_hidden', type=int, default=128, help='Number of hidden units in the model')
    parser.add_argument('--batch_size', type=int, default=128, help='Number of examples to process in a batch')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--train_steps', type=int, default=10000, help='Number of training steps')
    parser.add_argument('--max_norm', type=float, default=10.0)
    parser.add_argument('--device', type=str, default="cpu", help="Training device 'cpu' or 'cuda:0'")

    config = parser.parse_args()

    # Train the model
    main_grads(config)

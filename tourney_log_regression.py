import numpy as np

import torch
from torch.autograd import Variable
from torch import optim


def build_model(input_dim, output_dim):
    # We don't need the softmax layer here since CrossEntropyLoss already
    # uses it internally.
    model = torch.nn.Sequential()
    model.add_module("linear",
                     torch.nn.Linear(input_dim, output_dim, bias=False))
    return model


def predict(model, x):
    x = Variable(torch.from_numpy(x).float(), requires_grad=False)
    output = model.forward(x)
    return output.data.numpy().argmax(axis=1)


def update(loss, optimizer, pred_y, true_y):
    optimizer.zero_grad()
    output = loss.forward(pred_y,
                          Variable(torch.from_numpy(np.array(true_y).reshape(1)).int(), requires_grad=False))
    output.backward()
    optimizer.step()
    return output.data[0]


def pred_compute_ncaa_score(predictions):
    """
    TODO: Implement this function in pytorch.
    """
    return NotImplementedError

def train(N, d, X, y, tournament_sizes):
    """
    N: int
        Number of data instances
    d: int
        dimensionality of features
    X: np-array
        Data matrix
    y: np-array
        Ground truth
    tournament_sizes: np-array
        Array representing sizes of tournaments indexed in the same order as y.
    """
    model = build_model(d, 1)
    loss = torch.nn.L1Loss(size_average=True)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    
    for i in range(100):
        curr_ind = 0
        for j in range(len(y)):
            tourney_size = tournament_sizes[j]
            print tourney_size
            curr_batch = X[curr_ind:curr_ind+tourney_size]
            print curr_batch.shape
            predictions = predict(model, curr_batch)
            pred_score = pred_compute_ncaa_score(predictions)
            update(loss, optimizer, pred_score, y[j])
            curr_ind += tourney_size
    
    final_scores = []
    curr_ind = 0
    for j in range(len(y)):
        tourney_size = tournament_sizes[j]
        print tourney_size
        curr_batch = X[curr_ind:curr_ind+tourney_size]
        predictions = predict(model, curr_batch)
        pred_score = pred_compute_ncaa_score(predictions)
        final_scores.extend(pred_score)
        curr_ind += tourney_size
    return final_scores


def test_training():
    N = 1000
    d = 100
    M = 10
    X = np.random.rand(N, d)
    y = np.random.randint(0, 2, M)
    tournament_sizes = np.zeros(M)
    tournament_sizes.fill(int(N/M))
    
    train(N, d, X, y, tournament_sizes.astype(int))

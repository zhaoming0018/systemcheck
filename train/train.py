# coding: utf-8

import torch
import torch.nn as nn
import numpy as np
import os

import pprint
import sys
import time
import argparse

sys.path.append('../')
sys.path.append('../data/')

from resnet import resnet
from preact_resnet import *
from load_cifar import load_cifar
from load_olivetti import load_olivetti
from attack import *


CAP = 'cap'  # Capacity abuse attack
COR = 'cor'  # Correlation value encoding attack
SGN = 'sgn'  # Sign encoding attack
LSB = 'lsb'  # LSB encoding attack
NO = 'no'  # No attack

MODEL_DIR = './models/'
if not os.path.exists(MODEL_DIR):
    os.mkdir(MODEL_DIR)


def reshape_data(X_train, y_train, X_test):
    # reshape train and subtract mean
    pixel_mean = np.mean(X_train, axis=0)
    X_train -= pixel_mean
    X_test -= pixel_mean
    X_train_flip = X_train[:, :, :, ::-1]
    X_train = np.concatenate((X_train, X_train_flip), axis=0)
    y_train = np.concatenate((y_train, y_train), axis=0)
    return X_train, y_train, X_test


def iterate_minibatches(inputs, targets, batchsize, shuffle=False, augment=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        if augment:
            padded = np.pad(inputs[excerpt], ((0, 0), (0, 0), (4, 4), (4, 4)), mode='constant')
            random_cropped = np.zeros(inputs[excerpt].shape, dtype=np.float32)
            crops = np.random.random_integers(0, high=8, size=(batchsize, 2))
            for r in range(batchsize):
                random_cropped[r, :, :, :] = padded[r, :, crops[r, 0]:(crops[r, 0] + 32),
                                             crops[r, 1]:(crops[r, 1] + 32)]
            inp_exc = random_cropped
        else:
            inp_exc = inputs[excerpt]

        yield inp_exc, targets[excerpt]


def load_data(name='cifar10'):
    if name == 'cifar10':
        X_train, y_train, X_test, y_test = load_cifar(10)
        X_train = np.dstack((X_train[:, :1024], X_train[:, 1024:2048], X_train[:, 2048:]))
        X_train = X_train.reshape((-1, 32, 32, 3)).transpose(0, 3, 1, 2)
        X_test = np.dstack((X_test[:, :1024], X_test[:, 1024:2048], X_test[:, 2048:]))
        X_test = X_test.reshape((-1, 32, 32, 3)).transpose(0, 3, 1, 2)
        return X_train, y_train, X_test, y_test
    if name == 'olivetti':
        X_train, y_train, X_test, y_test = load_olivetti()
        return X_train, y_train, X_test, y_test


def main(num_epochs=500, lr=0.1, attack=CAP, res_n=5, corr_ratio=0.0, mal_p=0.1, model_name='resnet'):
    pprint.pprint(locals(), stream=sys.stderr)
    sys.stderr.write("Loading data...\n")
    X_train, y_train, X_test, y_test = load_data(name='cifar10')

    mal_n = int(mal_p * len(X_train) * 2)
    n_out = len(np.unique(y_train))

    if attack in {SGN, COR}:
        raw_data = X_train if X_train.dtype == np.uint8 else X_train * 255
        if raw_data.shape[-1] != 3:
            raw_data = raw_data.transpose(0, 2, 3, 1)
        raw_data = rbg_to_grayscale(raw_data).astype(np.uint8)
        sys.stderr.write('Raw data shape {}\n'.format(raw_data.shape))
        hidden_data_dim = np.prod(raw_data.shape[1:])
    elif attack == CAP:
        hidden_data_dim = int(np.prod(X_train.shape[2:]))
        mal_n //= hidden_data_dim
        if mal_n == 0:
            mal_n = 1
        X_mal, y_mal, mal_n = mal_data_synthesis(X_train, num_targets=mal_n)
        sys.stderr.write('Number of encoded image: {}\n'.format(mal_n))
        sys.stderr.write('Number of synthesized data: {}\n'.format(len(X_mal)))


    X_train, y_train, X_test = reshape_data(X_train, y_train, X_test)

    if attack == CAP:
        X_train_mal = np.vstack([X_train, X_mal])
        y_train_mal = np.concatenate([y_train, y_mal])

    n = len(X_train)
    sys.stderr.write("Number of training data, output: {}, {}...\n".format(n, n_out))

    sys.stderr.write("Building model and compiling functions...\n")
    if model_name == 'resnet':
        network = resnet()
    else:
        network = PreActResNet18()
    params = [p for p in network.parameters() if p.requires_grad]
    total_params = sum(p.numel() for p in params if p.ndimension()>1)
    sys.stderr.write("Number of parameters in model: %d\n" % total_params)

    if attack == COR:
        n_hidden_data = total_params // int(hidden_data_dim)
        sys.stderr.write("Number of data correlated: %d\n" % n_hidden_data)
        corr_targets = raw_data[:n_hidden_data].flatten()
        offset = set_params_init(params, corr_targets)
    elif attack == SGN:
        n_hidden_data = total_params // int(hidden_data_dim) // 8
        sys.stderr.write("Number of data sign-encoded: %d\n" % n_hidden_data)
        corr_targets = get_binary_secret(raw_data[:n_hidden_data])
        offset = set_params_init(params, corr_targets)

    sys.stderr.write("Starting training...\n")
    optimizer = torch.optim.SGD(network.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    criterion = nn.CrossEntropyLoss()
    network = network.to(device)

    for epoch in range(num_epochs):

        train_indices = np.arange(n)
        np.random.shuffle(train_indices)
        X_train = X_train[train_indices, :, :, :]
      
        y_train = y_train[train_indices]

        train_err = 0
        train_batches = 0
        start_time = time.time()
        train_r = 0

        network.train()
        for batch in iterate_minibatches(X_train, y_train, 128, shuffle=True, augment=True):
            inputs, targets = batch
            inputs = torch.from_numpy(inputs).to(device)
            targets = torch.from_numpy(targets).long().to(device)
            outputs = network(inputs)
            err = criterion(outputs, targets)
            if attack == COR:
                corr_loss, r = corr_term(params, corr_targets, size=offset)
                corr_loss *= corr_ratio
                err += corr_loss
            elif attack == SGN:
                corr_loss, r = sign_term(params, corr_targets, size=offset)
                corr_loss *= corr_ratio
                err += corr_loss
                
            optimizer.zero_grad()
            err.backward()
            optimizer.step()
            if attack in {SGN, COR}:
                train_r += r
            train_err += err
            train_batches += 1

        if attack == CAP:
            # And a full pass over the malicious data
            for batch in iterate_minibatches(X_train_mal, y_train_mal, 128, shuffle=True, augment=False):
                inputs, targets = batch
                inputs = torch.from_numpy(inputs).to(device)
                targets = torch.from_numpy(targets).long().to(device)
                outputs = network(inputs)
                err = criterion(outputs, targets)
                
                optimizer.zero_grad()
                err.backward()
                optimizer.step()
                train_err += err.item()
                train_batches += 1

        network.eval()
        if attack == CAP:
            mal_err = 0
            mal_acc = 0
            mal_batches = 0
            with torch.no_grad():
                for batch in iterate_minibatches(X_mal, y_mal, 512, shuffle=False):
                    inputs, targets = batch
                    inputs = torch.from_numpy(inputs).to(device)
                    targets = torch.from_numpy(targets).long().to(device)
                    outputs = network(inputs)

                    err = criterion(outputs, targets)
                    acc = outputs.max(1)[1].eq(targets).sum()
                    mal_err += err.item()
                    mal_acc += acc.item()
                    mal_batches += 1

        val_err = 0
        val_acc = 0
        val_batches = 0
        with torch.no_grad():
            for batch in iterate_minibatches(X_test, y_test, 500, shuffle=False):
                inputs, targets = batch
                inputs = torch.from_numpy(inputs).to(device)
                targets = torch.from_numpy(targets).long().to(device)
                outputs = network(inputs)
                err = criterion(outputs, targets)
                acc = outputs.max(1)[1].eq(targets).sum()

                val_err += err.item()
                val_acc += acc.item()
                val_batches += 1

        if (epoch + 1) == 41 or (epoch + 1) == 61:
            for p in optimizer.param_groups:
                p['lr'] *= 0.1

        # Then we sys.stderr.write the results for this epoch:
        sys.stderr.write("Epoch {} of {} took {:.3f}s\n".format(epoch + 1, num_epochs, time.time() - start_time))
        sys.stderr.write("  training loss:\t\t{:.6f}\n".format(train_err / train_batches))
        if attack == CAP:
            sys.stderr.write("  malicious loss:\t\t{:.6f}\n".format(mal_err / mal_batches))
            sys.stderr.write("  malicious accuracy:\t\t{:.2f} %\n".format(
                mal_acc / mal_batches / 512 * 100))
        if attack in {SGN, COR}:
            sys.stderr.write("  training r:\t\t{:.6f}\n".format(train_r / train_batches))

        sys.stderr.write("  validation loss:\t\t{:.6f}\n".format(val_err / val_batches))
        sys.stderr.write("  validation accuracy:\t\t{:.2f} %\n".format(val_acc / val_batches / 500 * 100))

    test_err = 0
    test_acc = 0
    test_batches = 0
    network.eval()
    with torch.no_grad():
        for batch in iterate_minibatches(X_test, y_test, 500, shuffle=False):
            inputs, targets = batch
            inputs = torch.from_numpy(inputs).to(device)
            targets = torch.from_numpy(targets).long().to(device)
            outputs = network(inputs)
            err = criterion(outputs, targets)
            acc = outputs.max(1)[1].eq(targets).sum()
            test_err += err.item()
            test_acc += acc.item()
            test_batches += 1

    sys.stderr.write("Final results:\n")
    sys.stderr.write("  test loss:\t\t\t{:.6f}\n".format(test_err / test_batches))
    sys.stderr.write("  test accuracy:\t\t{:.2f} %\n".format(test_acc / test_batches / 500 * 100))

    # save final model
    model_path = MODEL_DIR + 'cifar_{}_{}_'.format(attack, model_name)
    if attack == CAP:
        model_path += '{}_'.format(mal_p)
    if attack in {COR, SGN}:
        model_path += '{}_'.format(corr_ratio)
    torch.save(network.state_dict(), model_path + 'model.ckpt')

    print(test_acc / test_batches / 500)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=0.1)   
    parser.add_argument('--epoch', type=int, default=100)   
    parser.add_argument('--model', type=int, default=5)  
    parser.add_argument('--attack', type=str, default=CAP)
    parser.add_argument('--corr', type=float, default=0.)
    parser.add_argument('--mal_p', type=float, default=0.1)
    parser.add_argument('--model_name', type=str, default='resnet')
    args = parser.parse_args()
    main(num_epochs=args.epoch, lr=args.lr, corr_ratio=args.corr, mal_p=args.mal_p, attack=args.attack,
         res_n=args.model)

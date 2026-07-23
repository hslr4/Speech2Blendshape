import argparse
from data_loader import get_dataloaders
from model import Speech2BlendshapeNetwork
from test import test_loop
import torch
import csv
import os
import numpy as np

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="data/beat")
    parser.add_argument("--train_subjects", type=str, default="2 4 6 8")#scott, lawrence, carla, catherine
    parser.add_argument("--validation_subjects", type=str, default="2 4 6 8")
    parser.add_argument("--test_subjects", type=str, default="2 4 6 8")
    parser.add_argument("--embed_subjects", type=str2bool, default=True, help='whether or not to create one-hot embeddings for `train_subjects`')
    parser.add_argument("--embed_eval_subj", type=str, default='zeros', help='strategy for subj embedding of unseen subjects: zeros, ones or random')
    parser.add_argument("--embed_emotions", type=str2bool, default=True, help='whether or not to create one-hot embeddings for emotions')
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_dim", type=int, default=51, help='number of blendshapes')
    parser.add_argument("--gru_latent_dim", type=int, default=512, help='GRU decoder hidden size')
    parser.add_argument("--gru_layers", type=int, default=2, help='GRU decoder hidden size')
    parser.add_argument("--gru_dropout", type=float, default=0.3, help='GRU decoder dropout')
    parser.add_argument("--layer_dropout", type=float, default=0.1, help='dropout to apply before the last layer')
    parser.add_argument("--subject_dropout", type=float, default=0.3, help='dropout to apply to the one-hot subject embedding')
    parser.add_argument("--model_norm", type=str, default="none", help='normalization to apply either batch or layer or none')
    parser.add_argument("--output_fps", type=int, default=30,
                        help='fps of the visual data, BEAT was captured in 60 fps and downsampled to 30 by beat_split.py')
    parser.add_argument("--learning_rate", type=float, default=0.0001, help='learning rate')
    parser.add_argument("--epochs", type=int, default=5, help='number of epochs')
    parser.add_argument("--forzen_encoder_layers", type=str, default="0 1 2 3 4 5 6 7 8 9 10 11", help='layers of speech encoder (hubert) to freeze (0-11)')
    parser.add_argument("--loss_fn", type=str, default="mse", help='loss function: use mse, l1 or huber')
    parser.add_argument("--huber_delta", type=float, default=1.0, help='delta for huber loss function')
    parser.add_argument("--optimizer", type=str, default="adam", help='optimizer: adam, adamw or sgd')
    parser.add_argument("--momentum", type=float, default=0, help='optimizer momentum, only used when optimizer==sgd')
    parser.add_argument("--weight_decay", type=float, default=0, help='optimizer weight decay')
    parser.add_argument("--save_path", type=str, default="models", help='path to save trained models and metrics')
    parser.add_argument("--scale_loss", type=str2bool, default=False, help='whether or not to scale loss by label mean')
    parser.add_argument("--embedding_integration", type=str, default='mul', help='how to combine subject and emotion embeddings with processed audio: mul, add or cat')
    args = parser.parse_args()
    args.forzen_encoder_layers = [int(l) for l in args.forzen_encoder_layers.split()]
    return args


def train_loop(dataloader, model, loss_fn, optimizer, batch_size, label_means):
    size = len(dataloader.dataset)
    train_loss = 0
    num_batches = len(dataloader)
    model.train()
    for batch, (X, y, subj, emo) in enumerate(dataloader):
        pred = model(X, subj, emo)
        if args.scale_loss:
            loss = loss_fn(pred, y)
            mean = label_means.repeat([y.shape[1]], axis=1)
            mean = mean.repeat([y.shape[0]], axis=0)
            mean = torch.from_numpy(mean).to(torch.device('cuda'))
            loss = loss * (1+2*torch.abs(y-mean))
            loss = torch.mean(loss)
        else:
            loss = loss_fn(pred, y)
        train_loss += loss.item()

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 15 == 0:
            loss, current = loss.item(), batch * batch_size + len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    train_loss /= num_batches
    return train_loss


if __name__ == "__main__":
    args = parse_args()
    assert torch.cuda.is_available(), 'cuda should be available for training at least'
    data_loaders, n_emos = get_dataloaders(
        args.data_path, 
        args.train_subjects, 
        args.validation_subjects, 
        args.test_subjects, 
        args.batch_size, 
        args.embed_eval_subj,
        args.device,
    )

    reduction = 'mean'
    train_label_means = []
    if args.scale_loss:
        reduction = 'none'
        for fn in data_loaders['train'].dataset.filenames:
            labels = np.load(os.path.join(args.data_path, 'vertices_npy', fn + '.npy'))
            train_label_means.append(np.mean(labels, axis=0))
        train_label_means = np.array(train_label_means)
        train_label_means = np.mean(train_label_means, axis=0)
        train_label_means = np.array([[train_label_means]])
    
    match args.loss_fn.lower():
        case 'mse':
            loss_fn = torch.nn.MSELoss(reduction=reduction)
        case 'huber':
            loss_fn = torch.nn.HuberLoss(delta=args.huber_delta)
        case 'l1':
            loss_fn = torch.nn.L1Loss(reduction=reduction)
        case _:
            print(f'unknown loss function provided {args.loss_fn}. Use mse, huber or l1')
            exit()
    
    model = Speech2BlendshapeNetwork(
        args.output_dim,
        args.gru_latent_dim, 
        args.gru_layers, 
        args.output_fps, 
        args.gru_dropout,
        len(args.train_subjects.split()) if args.embed_subjects else 0, 
        n_emos if args.embed_emotions else 0,
        args.forzen_encoder_layers,
        args.model_norm, 
        args.layer_dropout, 
        args.subject_dropout,
        args.embedding_integration,
        args.train_subjects
    )
    model.to(torch.device('cuda'))
    print(model)

    match args.optimizer.lower():
        case 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        case 'adamw':
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        case 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
        case _:
            print(f'unknown optimizer provided {args.optimizer}. Use adam, adamw or sgd')
    
    metric_rows = [['epoch', 'train_loss', 'test_loss', 'mae', 'mouth_mae', 'mse', 'mouth_mse', 'rae']]
    for t in range(args.epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loss = train_loop(data_loaders['train'], model, loss_fn, optimizer, args.batch_size, label_means=train_label_means)
        test_metrics = test_loop(data_loaders['test'], model)
        metric_rows.append([t, train_loss] + test_metrics[:6])
    model.save_pretrained(args.save_path)
    print(f'Saved model to {args.save_path}')
    with open(os.path.join(os.path.join(args.save_path),'metrics.csv'), 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for row in metric_rows:
            writer.writerow(row)
    print("Done!")
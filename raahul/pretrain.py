import os
import sys
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl
import wandb
from copy import deepcopy
from pathlib import Path
from datetime import datetime
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from torch_geometric.data import DataLoader
from torch_geometric.data.sampler import NeighborSampler as PyGeoNeighborSampler

sys.path.insert(0, '..')  # Add project_config to path

# Own modules
import preprocess
from node_embedder_model import NodeEmbeder
import project_config
from hparams import get_pretrain_hparams
from samplers import NeighborSampler

def parse_args():
    parser = argparse.ArgumentParser(description="Learn node embeddings.")
    parser.add_argument("--edgelist", type=str, help="File with edge list")
    parser.add_argument("--node_map", type=str, help="File with node list")
    parser.add_argument('--save_dir', type=str, help='Directory for saving files')
    parser.add_argument('--nfeat', type=int, default=2048, help='Embedding layer dimension')
    parser.add_argument('--hidden', type=int, default=256)
    parser.add_argument('--output', type=int, default=128)
    parser.add_argument('--n_heads', type=int, default=4)
    parser.add_argument('--wd', type=float, default=0.0)
    parser.add_argument('--dropout', type=float, default=0.3, help='Dropout rate')
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--max_epochs', type=int, default=1000)
    parser.add_argument('--resume', type=str, default="")
    parser.add_argument('--best_ckpt', type=str, help='Best performing checkpoint')
    parser.add_argument('--save_embeddings', action='store_true')
    return parser.parse_args()

def get_dataloaders(hparams, all_data):
    train_dataloader = NeighborSampler('train', all_data.edge_index[:, all_data.train_mask], 
                                       all_data.edge_index[:, all_data.train_mask], 
                                       sizes=hparams['neighbor_sampler_sizes'],
                                       batch_size=hparams['batch_size'],
                                       shuffle=True, num_workers=hparams['num_workers'],
                                       do_filter_edges=hparams['filter_edges'])
    val_dataloader = NeighborSampler('val', all_data.edge_index[:, all_data.train_mask], 
                                     all_data.edge_index[:, all_data.val_mask], 
                                     sizes=hparams['neighbor_sampler_sizes'],
                                     batch_size=hparams['batch_size'],
                                     shuffle=False, num_workers=hparams['num_workers'],
                                     do_filter_edges=hparams['filter_edges'])
    test_dataloader = NeighborSampler('test', all_data.edge_index[:, all_data.train_mask], 
                                      all_data.edge_index[:, all_data.test_mask], 
                                      sizes=hparams['neighbor_sampler_sizes'],
                                      batch_size=hparams['batch_size'],
                                      shuffle=False, num_workers=hparams['num_workers'],
                                      do_filter_edges=hparams['filter_edges'])
    return train_dataloader, val_dataloader, test_dataloader

def train(args, hparams):
    pl.seed_everything(hparams['seed'])
    all_data, edge_attr_dict, nodes = preprocess.preprocess_graph(args)
    run_name = args.resume if args.resume else f"{datetime.now().strftime('%H:%M:%S')}_run"
    wandb_logger = WandbLogger(run_name, project='kg-train', entity='rare_disease_dx',
                               save_dir=hparams['wandb_save_dir'], id=run_name.replace(':', '_'), resume="allow")
    if args.resume:
        model = NodeEmbeder.load_from_checkpoint(
            checkpoint_path=str(Path(args.save_dir) / 'checkpoints' / args.best_ckpt),
            all_data=all_data, edge_attr_dict=edge_attr_dict,
            num_nodes=len(nodes["node_idx"].unique()), combined_training=False)
    else:
        model = NodeEmbeder(all_data, edge_attr_dict, hp_dict=hparams,
                            num_nodes=len(nodes["node_idx"].unique()), combined_training=False)
    checkpoint_callback = ModelCheckpoint(monitor='val/node_total_acc', dirpath=Path(args.save_dir) / 'checkpoints',
                                          filename=f'{run_name}_{{epoch}}', save_top_k=1, mode='max')
    lr_monitor = LearningRateMonitor(logging_interval='step')
    wandb_logger.watch(model, log='all')
    trainer = pl.Trainer(
        gpus=hparams['n_gpus'], logger=wandb_logger, max_epochs=hparams['max_epochs'],
        callbacks=[checkpoint_callback, lr_monitor], gradient_clip_val=hparams['gradclip'],
        profiler=hparams['profiler'], log_every_n_steps=hparams['log_every_n_steps'],
        limit_train_batches=1.0 if not hparams['debug'] else 1,
        limit_val_batches=1.0 if not hparams['debug'] else 1)
    train_dataloader, val_dataloader, test_dataloader = get_dataloaders(hparams, all_data)
    trainer.fit(model, train_dataloader, val_dataloader)
    trainer.test(ckpt_path='best', test_dataloaders=test_dataloader)

def save_embeddings(args, hparams):
    print('Saving Embeddings')
    pl.seed_everything(hparams['seed'])
    all_data, edge_attr_dict, nodes = preprocess.preprocess_graph(args)
    all_data.num_nodes = len(nodes["node_idx"].unique())
    model = NodeEmbeder.load_from_checkpoint(
        checkpoint_path=str(Path(args.save_dir) / 'checkpoints' / args.best_ckpt),
        all_data=all_data, edge_attr_dict=edge_attr_dict,
        num_nodes=len(nodes["node_idx"].unique()), combined_training=False)
    dataloader = DataLoader([all_data], batch_size=1)
    trainer = pl.Trainer(gpus=0, gradient_clip_val=hparams['gradclip'])
    embeddings = trainer.predict(model, dataloaders=dataloader)
    embed_path = Path(args.save_dir) / (str(args.best_ckpt).split('.ckpt')[0] + '.embed')
    torch.save(embeddings[0], str(embed_path))
    print(f"Saved embeddings of shape {embeddings[0].shape} at {embed_path}")

if __name__ == "__main__":
    args = parse_args()
    hparams = get_pretrain_hparams(args, combined=False)
    if args.save_embeddings:
        save_embeddings(args, hparams)
    else:
        train(args, hparams)

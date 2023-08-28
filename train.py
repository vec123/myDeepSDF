import torch
import torch.utils.data as data_utils
import signal
import sys
import os
import math
import json
import pdb

from utils.utils import save_model, save_optimizer, save_latent_vectors, load_specifications

def save_latest(epoch):
        save_model(experiment_directory, "latest.pth", decoder, epoch)
        save_optimizer(experiment_directory, "latest.pth", optimizer_all, epoch)
        save_latent_vectors(experiment_directory, "latest.pth", lat_vecs, epoch)


#Get Specifications:
specs = load_specifications("~/myDeepSDF/specs")

#Get Loss:
loss_l1 = torch.nn.L1Loss(reduction="sum")

#Get optimizer:
optimizer_all = torch.optim.Adam(
        [
            {
                "params": decoder.parameters(),
                "lr": lr_schedules[0].get_learning_rate(0),
            },
            {
                "params": lat_vecs.parameters(),
                "lr": lr_schedules[1].get_learning_rate(0),
            },
        ]
)


#Get decoder and encoder:
decoder = DeepSDF(latent_size, **specs["NetworkSpecs"]).cuda()
print("training with {} GPU(s)".format(torch.cuda.device_count()))
decoder = torch.nn.DataParallel(decoder)

#get training data:
with open(train_split_file, "r") as f:
       train_split = json.load(f)
sdf_dataset = lib.data.SDFSamples(data_source, train_split, num_samp_per_scene)
num_data_loader_threads = get_spec_with_default(specs, "DataLoaderThreads", 1)

#get dataloaders
sdf_loader = data_utils.DataLoader(
        sdf_dataset,
        batch_size=scene_per_batch,
        shuffle=True,
        num_workers=num_data_loader_threads,
        drop_last=False,
    )

sdf_loader_reconstruction = data_utils.DataLoader(
        sdf_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_data_loader_threads,
        drop_last=False,
    )
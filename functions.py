# Copyright 2025, Maxime Burchi.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch

import os
import glob

def find_last_checkpoint(callback_path, return_full_path=False):
    """Return the checkpoint with the highest step count, or None."""

    checkpoints = glob.glob(os.path.join(callback_path, "checkpoints_*.ckpt"))

    max_steps = 0
    last_checkpoint = None
    for checkpoint in checkpoints:
        checkpoint = checkpoint.split("/")[-1]
        checkpoint_steps = int(checkpoint.split("_")[-1].replace(".ckpt", ""))
        if checkpoint_steps > max_steps:
            max_steps = checkpoint_steps
            last_checkpoint = checkpoint

    if last_checkpoint != None and return_full_path:
        last_checkpoint = os.path.join(callback_path, last_checkpoint)

    return last_checkpoint

def load_model(args):
    """Build the configured model on the available device and restore a checkpoint."""

    device = torch.device("cuda:0" if torch.cuda.is_available() and not args.cpu else "cpu")
    if "cuda" in str(device):
        print("device: {}, {}, {}MB".format(device, torch.cuda.get_device_properties(device).name, int(torch.cuda.get_device_properties(device).total_memory // 1e6)))
        args.num_gpus = torch.cuda.device_count()
    else:
        print("device: {}".format(device))
        args.num_gpus = 1

    model = args.config.model.to(device)

    # Callback path defaults to the config module path, plus an optional tag
    args.config.callback_path = getattr(args.config, "callback_path", os.path.join("callbacks", "/".join(args.config_file.replace(".py", "").split("/")[1:])))
    if hasattr(args.config, "callback_tag"):
        args.config.callback_path = os.path.join(args.config.callback_path, args.config.callback_tag)

    if args.load_last:
        last_checkpoint = find_last_checkpoint(args.config.callback_path)
        if last_checkpoint != None:
            args.checkpoint = last_checkpoint

    if args.checkpoint is not None:
        model.load(os.path.join(args.config.callback_path, args.checkpoint))

    model.summary(show_dict=args.show_dict, show_modules=args.show_modules)
    
    return model

def load_datasets(args):
    """Wrap the configured training and evaluation datasets in DataLoaders."""

    def print_dataset(args, dataset, tag):

        print("{} Dataset: {}, {:,} samples - {:,} batches - batch size {}".format(tag, dataset.dataset.__class__.__name__, len(dataset.dataset), len(dataset), dataset.dataset.batch_size))

    # Training Dataset
    if hasattr(args.config, "training_dataset"):

        dataset_train = torch.utils.data.DataLoader(
            dataset=args.config.training_dataset,
            batch_size=args.config.training_dataset.batch_size,
            shuffle=args.config.training_dataset.shuffle,
            sampler=None,
            num_workers=args.config.training_dataset.num_workers,
            collate_fn=args.config.training_dataset.collate_fn,
            pin_memory=False,
            drop_last=True,
            worker_init_fn=getattr(args.config, "worker_init_fn", None),
            persistent_workers=args.config.training_dataset.persistent_workers,
        )

        print_dataset(args, dataset_train, "Training")

    else:

        dataset_train = None

    # Evaluation Dataset
    if hasattr(args.config, "evaluation_dataset"):

        # One or several evaluation datasets
        if isinstance(args.config.evaluation_dataset, list):

            dataset_eval = []
            for dataset in args.config.evaluation_dataset:

                dataset_eval.append(torch.utils.data.DataLoader(
                    dataset=dataset,
                    batch_size=dataset.batch_size,
                    shuffle=dataset.shuffle,
                    sampler=None,
                    num_workers=dataset.num_workers,
                    collate_fn=dataset.collate_fn,
                    pin_memory=False,
                    drop_last=False,
                    worker_init_fn=getattr(args.config, "worker_init_fn", None),
                    persistent_workers=dataset.persistent_workers,
                ))

                print_dataset(args, dataset_eval[-1], "Evaluation")

        else:

            dataset_eval = torch.utils.data.DataLoader(
                dataset=args.config.evaluation_dataset,
                batch_size=args.config.evaluation_dataset.batch_size,
                shuffle=args.config.evaluation_dataset.shuffle,
                sampler=None,
                num_workers=args.config.evaluation_dataset.num_workers,
                collate_fn=args.config.evaluation_dataset.collate_fn,
                pin_memory=False,
                drop_last=False,
                worker_init_fn=getattr(args.config, "worker_init_fn", None),
                persistent_workers=args.config.evaluation_dataset.persistent_workers,
            )

            print_dataset(args, dataset_eval, "Evaluation")
                
    else:
        dataset_eval = None
    
    return dataset_train, dataset_eval
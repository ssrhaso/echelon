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

# PyTorch
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

# Other
from tqdm import tqdm
import os
import time
import glob
import wandb

# Neural Nets
from nnet import modules
from nnet import schedulers
from nnet.optimizers import optim_dict

class Model(modules.Module):

    def __init__(self, name="model"):
        super(Model, self).__init__()

        # Model Attributes
        self.compiled = False
        self.built = False
        self.name = name
        self.grad_scaler_state_dict = None

    def compile(self, losses, loss_weights=None, optimizer="Adam", metrics=None, decoders=None):

        # Optimizer
        if isinstance(optimizer, str):
            self.optimizer = optim_dict[optimizer](params=self.parameters())
        else:
            self.optimizer = optimizer

        # Model Step
        self.model_step = self.optimizer.param_groups[0]["lr_scheduler"].model_step

        # Losses
        if losses == None:
            self.compiled_losses = []
        else:
            self.compiled_losses = losses

        # Loss Weights
        if loss_weights == None:

            self.compiled_loss_weights = schedulers.ConstantScheduler(1.0)

        elif isinstance(loss_weights, float):

            self.compiled_loss_weights = schedulers.ConstantScheduler(loss_weights)

        else:

            # Assert List or Dict
            assert isinstance(loss_weights, dict) or isinstance(loss_weights, list)

            # Convert to Scheduler
            if isinstance(loss_weights, dict):
                for key, value in loss_weights.items():
                    if not isinstance(value, schedulers.Scheduler):
                        loss_weights[key] = schedulers.ConstantScheduler(value)
            else:
                for i, value in enumerate(loss_weights):
                    if not isinstance(value, schedulers.Scheduler):
                        loss_weights[i] = schedulers.ConstantScheduler(value)

            # Assign
            self.compiled_loss_weights = loss_weights

        # Metrics
        if metrics == None:
            self.compiled_metrics = []
        else:
            self.compiled_metrics = metrics
            
        # Decoders
        if decoders == None:
            self.compiled_decoders = []
        else:
            self.compiled_decoders = decoders

        # Set Compiled to True
        self.compiled = True

        # Set Modules Name
        for name, module in self.named_modules():
            if not hasattr(module, "name"):
                module.name = name

    def build(self, outputs):

        # Map to Outputs
        self.losses = self.map_to_outputs(outputs, self.compiled_losses)
        self.loss_weights = self.map_to_outputs(outputs, self.compiled_loss_weights)
        self.decoders = self.map_to_outputs(outputs, self.compiled_decoders)
        self.metrics = self.map_to_outputs(outputs, self.compiled_metrics)

        # Transfer to Device
        self.losses = self.transfer_to_device(self.losses)
        self.decoders = self.transfer_to_device(self.decoders)
        self.metrics = self.transfer_to_device(self.metrics)

        # Set Built to true
        self.built = True

    def map_to_outputs(self, outputs, struct):

        """Convenience method to conform `struct` to `outputs` structure.

        Mappings performed:
            (1) Map a struct to a dict of outputs, using the output names.
            (2) Fill missing struct elements with None.
            (3) Map a single item to all outputs.

        Args:
            outputs: Model outputs predictions dict.
            struct: Arbitrary nested structure (dict, list, item).

        Returns:
            Dict mapping `struct` to `outputs` structure.

        """

        # None
        if struct == None:

            return struct

        # Dictionary
        elif isinstance(struct, dict):

            # Assert struct key in outputs
            for key in struct:
                if not key in outputs:
                    raise Exception("Found unexpected dict key: {}. Valid output names are: {}".format(key, outputs.keys()))

            # Fill missing key with None
            for key in outputs:
                if not key in struct:
                    struct[key] = None

        # List
        elif isinstance(struct, list):

            # Map list items to outputs, Fill missing items with None, Ignore extra items
            struct = {key: struct[i] if i < len(struct) else None for i, key in enumerate(outputs)}

        # Module / Tensor / tuple
        else:

            # Map item to all outputs
            struct = {key: struct for key in outputs}

        return struct

    def forward_model(self, inputs, targets, compute_metrics=True, verbose=0):

        """ forward_model method

        - forward
        - compute losses
        - compute metrics
        
        """

        # Init Batch Dict
        batch_losses = {}
        batch_metrics = {}
        batch_truths = {}
        batch_preds = {}
        total_loss = torch.tensor(0.0, device=self.device)

        # Additional Targets
        self.additional_targets = {}

        # Forward
        outputs = self.forward(inputs)

        # Format Outputs to dict
        if isinstance(outputs, dict):
            pass
        elif isinstance(outputs, list):
            outputs = {"output_" + str(key): value for key, value in enumerate(outputs)}
        else:
            outputs = {"output": outputs}

        # Map Targets to Outputs
        targets = self.map_to_outputs(outputs, targets)

        # Append Additional Targets
        for key in self.additional_targets:
            targets[key] = self.additional_targets[key]

        # Build Model
        if not self.built:
            self.build(outputs)

        # Outputs loop
        for key in outputs:

            # Loss Function
            if self.losses[key] != None:

                # Loss key
                key_loss = "loss_" + key

                # Loss
                batch_losses[key_loss] = self.losses[key](targets[key], outputs[key])

                # Weight Loss
                total_loss += batch_losses[key_loss] * self.loss_weights[key].get_val_step(self.model_step + 1)

            # Metric Functions
            if self.metrics[key] != None and compute_metrics:

                # To list
                if not isinstance(self.metrics[key], list):
                    metrics = [self.metrics[key]]
                else:
                    metrics = self.metrics[key]
                if not isinstance(self.decoders[key], list):
                    decoders = [self.decoders[key] for _ in metrics]
                else:
                    decoders = self.decoders[key]


                for metric, decoder in zip(metrics, decoders):

                    # Metric Key
                    key_metric = metric.name
                    if key_metric in batch_metrics:
                        key_metric += "_" + key

                    # Decoding
                    if decoder != None:
                        batch_truths[key_metric] = decoder(targets[key], from_logits=False) if targets[key] != None else None
                        batch_preds[key_metric] = decoder(outputs[key])
                    else:
                        batch_truths[key_metric] = targets[key]
                        batch_preds[key_metric] = outputs[key]

                    # Prediction Verbose
                    if verbose:
                        print("Groundtruths:\n", batch_truths[key_metric])
                        print("Predictions:\n", batch_preds[key_metric])

                    # Metric
                    batch_metrics[key_metric] = metric(batch_truths[key_metric], batch_preds[key_metric])

        # Module Infos / Losses
        for module in self.modules():

            # Module added losses during forward
            if hasattr(module, "added_losses"):
                for key, value in module.added_losses.items():
                    key_loss = "loss_" + key
                    batch_losses[key_loss] = value["loss"]
                    total_loss += batch_losses[key_loss] * value["weight"]
                module.reset_losses()

            # Module added infos during forward
            if hasattr(module, "infos") and module is not self: # Do not include self to avoid reset infos
                self.infos.update(module.infos)
                module.reset_infos()

            # Module added metrics during forward
            if hasattr(module, "added_metrics"):
                for key, value in module.added_metrics.items():
                    key_metric = key
                    batch_metrics[key_metric] = value
                module.reset_metrics()

        # Append Total loss
        if len(batch_losses) > 1:
            batch_losses = dict({"loss": total_loss}, **batch_losses)
        else:
            batch_losses = {"loss": total_loss}

        return batch_losses, batch_metrics, batch_truths, batch_preds

    def train_step(self, inputs, targets, precision, grad_scaler, accumulated_steps, acc_step, eval_training):

        """ train_step method

        - forward_model (forward + compute losses/metrics)
        - backward
        
        """

        # Automatic Mixed Precision Casting (model forward + loss computing)
        if "cuda" in str(self.device):
            with torch.cuda.amp.autocast(enabled=precision!=torch.float32, dtype=precision):
                batch_losses, batch_metrics, batch_truths, batch_preds = self.forward_model(inputs, targets, compute_metrics=eval_training)
        else:
            batch_losses, batch_metrics, batch_truths, batch_preds = self.forward_model(inputs, targets, compute_metrics=eval_training)

        # Accumulated Steps
        loss = batch_losses["loss"] / accumulated_steps
        acc_step += 1

        # Backward: Accumulate gradients
        grad_scaler.scale(loss).backward()

        # Continue Accumulating
        if acc_step < accumulated_steps:
            return batch_losses, batch_metrics, acc_step

        # Grad Scaler Info
        if grad_scaler.is_enabled():
            self.add_info("grad_scale", grad_scaler.get_scale())

        # Unscale Gradients
        grad_scaler.unscale_(self.optimizer)

        # Optimizer Step and Update Scale
        grad_scaler.step(self.optimizer)
        grad_scaler.update()

        # Zero Gradients
        self.optimizer.zero_grad()
        acc_step = 0

        # Update Model Infos
        if len(self.optimizer.param_groups) > 1:
            for i, param_group in enumerate(self.optimizer.param_groups):

                # learning rate
                self.add_info("lr_{}".format(i), float(param_group['lr']))

                # grad norm
                if "grad_norm" in param_group:
                    self.add_info("grad_norm_{}".format(i), round(float(param_group['grad_norm']), 4))

                # grad infos
                if "grad_min" in param_group:
                    self.add_info("grad_min_{}".format(i), param_group['grad_min'])
                if "grad_max" in param_group:
                    self.add_info("grad_max_{}".format(i), param_group['grad_max'])
                if "grad_mean" in param_group:
                    self.add_info("grad_mean_{}".format(i), param_group['grad_mean'])
                if "grad_std" in param_group:
                    self.add_info("grad_std_{}".format(i), param_group['grad_std'])
        else:

            # learning rate
            self.add_info("lr", float(self.optimizer.param_groups[0]['lr']))

            # grad norm
            if "grad_norm" in self.optimizer.param_groups[0]:
                self.add_info("grad_norm", round(float(self.optimizer.param_groups[0]['grad_norm']), 4))

            # grad infos
            if "grad_min" in self.optimizer.param_groups[0]:
                self.add_info("grad_min", self.optimizer.param_groups[0]['grad_min'])
            if "grad_max" in self.optimizer.param_groups[0]:
                self.add_info("grad_max", self.optimizer.param_groups[0]['grad_max'])
            if "grad_mean" in self.optimizer.param_groups[0]:
                self.add_info("grad_mean", self.optimizer.param_groups[0]['grad_mean'])
            if "grad_std" in self.optimizer.param_groups[0]:
                self.add_info("grad_std", self.optimizer.param_groups[0]['grad_std'])

        # Add Info Model Step
        self.add_info("step", self.model_step.item())

        return batch_losses, batch_metrics, acc_step  

    def eval_step(self, inputs, targets, verbose=0):

        with torch.no_grad():
            batch_losses, batch_metrics, batch_truths, batch_preds = self.forward_model(inputs, targets, verbose=verbose)

        return batch_losses, batch_metrics, batch_truths, batch_preds

    def num_params(self, module=None):

        if module != None:
            if isinstance(module, list):
                return sum([self.num_params(m) for m in module])
            else:
                return sum([p.numel() for p in module.parameters()])
        else:
            return sum([p.numel() for p in self.parameters()])

    def summary(self, show_dict=False, show_modules=False):

        # Model Name
        print("Model name: {}".format(self.name))

        # Number Params
        print("Number Parameters: {:,}".format(self.num_params()))

        # Show Modules Params
        for key, value in self.named_children():
            print("{}: {:,} Parameters".format(key, self.num_params(value)))

        # Options
        if show_dict:
            self.show_dict()
        if show_modules:
            self.show_modules()

        # Modules Buffer
        for key, value in self.modules_buffer.items():
            print("{} Parameters: {:,}".format(key, self.num_params(value)))

    def show_dict(self, module=None):

        # Print
        print("State Dict:")

        # Default Dict
        if module != None:
            state_dict = module.state_dict(keep_vars=True)
        else:
            state_dict = self.state_dict(keep_vars=True)

        # Empty Dict
        if state_dict == {}:
            return

        # Show Dict
        max_len_id = len(str(len(state_dict)))
        max_len_key = max([len(key) for key in state_dict.keys()]) + 5
        for id, (key, value) in enumerate(state_dict.items()):
            print("{} {} type: {:<12} numel: {:<12} shape: {:<20} mean: {:<12.4f} std: {:<12.4f} min: {:<12.4f} max: {:<12.4f} dtype: {:<12} device: {}".format(str(id) + " " * (max_len_id - len(str(id))), key + " " * (max_len_key - len(key)), "param" if isinstance(value, nn.Parameter) else "buffer", value.numel(), str(tuple(value.size())), value.float().mean(), value.float().std(), value.float().min(), value.float().max(), str(value.dtype).replace("torch.", ""), str(value.device)))

    def show_modules(self, module=None):

        # Print
        print("Named Modules:")

        # Named Modules
        if module != None:
            named_modules = dict(module.named_modules())
        else:
            named_modules = dict(self.named_modules())

        # Show Modules
        max_len_id = len(str(len(named_modules)))
        max_len_key = max([len(key) for key in named_modules.keys()]) + 5
        max_len_class = max([len(type(value).__name__) for value in named_modules.values()]) + 5
        for id, (key, value) in enumerate(named_modules.items()):
            print("{} {} class: {} device: {}".format(str(id) + " " * (max_len_id - len(str(id))), key + " " * (max_len_key - len(key)), type(value).__name__ + " " * (max_len_class - len(type(value).__name__)), value.device if hasattr(value, "device") else ""))

    def save(self, path, save_optimizer=True, keep_last_k=None):
        
        # Save Model Checkpoint
        torch.save({
            "model_state_dict": self.state_dict(),
            "optimizer_state_dict": None if not save_optimizer else {key: value.state_dict() for key, value in self.optimizer.items()} if isinstance(self.optimizer, dict) else self.optimizer.state_dict(),
            "model_step": self.model_step,
            "grad_scaler_state_dict": self.grad_scaler.state_dict() if hasattr(self, "grad_scaler") else None
            }, path)

        # Print Model state
        print("Model saved at step {}: {}".format(self.model_step, path))

        # Keep last k checkpoints
        if keep_last_k != None:

            # List checkpoints
            save_dir = os.path.dirname(path)
            checkpoints_list = glob.glob(os.path.join(save_dir, "*.ckpt"))
            checkpoints_list = sorted(checkpoints_list, key=lambda s: int(os.path.splitext(s)[0].split("/")[-1].split("_")[-1]))

            # Remove older_checkpoint
            while len(checkpoints_list) > keep_last_k:

                # Pop older_checkpoint
                older_checkpoint = checkpoints_list.pop(0)

                # Remove older_checkpoint
                os.remove(older_checkpoint)

                # Print
                print("Removed old checkpoint: {}".format(older_checkpoint))

    def load(self, path, load_optimizer=True, verbose=True, strict=True):

        # Print Load state
        if verbose:
            print("Load Model from {}".format(path))

        # Load Model Checkpoint
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        # Load Model State Dict
        self.load_state_dict({key:value for key, value in checkpoint["model_state_dict"].items()}, strict=strict)

        # Load Optimizer State Dict
        if load_optimizer and checkpoint["optimizer_state_dict"] is not None:

            if isinstance(self.optimizer, dict):
                for key, value in self.optimizer.items():
                    value.load_state_dict(checkpoint["optimizer_state_dict"][key])
            else:
                self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

            # Model Step, already loaded from optm
            self.model_step.fill_(checkpoint["model_step"])

        # Load Grad Scaler
        if "grad_scaler_state_dict" in checkpoint:
            self.grad_scaler_state_dict = checkpoint["grad_scaler_state_dict"]

        # Print Model state
        if verbose:
            print("Model loaded at step {}".format(self.model_step))

    def on_train_begin(self):
        pass

    def on_epoch_begin(self, epoch):
        pass

    def on_epoch_end(self, evaluate, save, log_figure, callback_path, epoch, inputs, targets, dataset_eval, eval_steps, verbose_eval, writer, recompute_metrics, keep_last_k):
        self.on_step_end(evaluate, save, log_figure, callback_path, epoch, epoch, inputs, targets, dataset_eval, eval_steps, verbose_eval, writer, recompute_metrics, keep_last_k=keep_last_k, tag="epoch")

        # Print
        print()

    def on_step_end(self, evaluate, save, log_figure, callback_path, epoch, step, inputs, targets, dataset_eval, eval_steps, verbose_eval, writer, recompute_metrics, keep_last_k, tag="step"):

        # Evaluate Model
        if evaluate:
            self._evaluate(dataset_eval, writer, eval_steps, verbose_eval, recompute_metrics, tag="Evaluation-" + tag)
            self.train()

        # Save Checkpoint
        if save and callback_path:
            self.save(os.path.join(callback_path, "checkpoints_epoch_{}_step_{}.ckpt".format(epoch, self.model_step)), keep_last_k=keep_last_k)

        # Log Figure
        if log_figure and callback_path:
            self.eval()
            self.log_figure(step, inputs, targets, writer, tag)
            self.train()

    def log_figure(self, step, inputs, targets, writer, tag): 
        pass

    def display_step(self, losses, metrics, infos, epoch_iterator, step):

        # Description
        description = ""

        # Losses
        for key, value in losses.items():
            description += "{}: {:.4f} - ".format(key, value / step)

        # Metrics
        for key, value in metrics.items():
            description += "{}: {:.4f} - ".format(key, value / step)

        # Infos
        for key, value in infos.items():

            # Display format
            if key.startswith("lr"):
                description += "{}: {:.2e} - ".format(key, value)
            elif isinstance(value, float):
                description += "{}: {:.4f} - ".format(key, value)
            else:
                description += "{}: {} - ".format(key, value)

        # Set description
        epoch_iterator.set_description(description)

    def log_step(self, losses, metrics, infos, writer, step, tag):

        # Losses
        for key, value in losses.items():
            writer.add_scalar(os.path.join(tag, key), value, step)

        # Metrics
        for key, value in metrics.items():
            writer.add_scalar(os.path.join(tag, key), value, step)

        # Infos
        for key, value in infos.items():
            if isinstance(value, float) or isinstance(value, int):
                writer.add_scalar(os.path.join(tag, key), float(value), step)
            elif isinstance(value, torch.Tensor):
                if value.numel() == 1:
                    writer.add_scalar(os.path.join(tag, key), float(value), step)

    def print_step(self, losses, metrics, tag):

        # Losses
        for key, value in losses.items():
            print("{} {}: {:.4f}".format(tag, key, value))

        # val metrics
        for key, value in metrics.items():
            print("{} {}: {:.4f}".format(tag, key, value))

    def fit(
        self, 
        dataset_train, 
        epochs, 
        dataset_eval=None, 
        eval_steps=None, 
        verbose_eval=0, 
        initial_epoch=0, 
        callback_path=None, 
        steps_per_epoch=None, 
        precision=torch.float32, 
        accumulated_steps=1, 
        eval_period_step=None, 
        eval_period_epoch=1,
        saving_period_epoch=1, 
        log_figure_period_step=None, 
        log_figure_period_epoch=1, 
        step_log_period=10, 
        eval_training=True,
        grad_init_scale=65536.0, 
        detect_anomaly=False, 
        recompute_metrics=False,
        wandb_logging=False,
        verbose_progress_bar=1,
        keep_last_k=None
    ):
        
        # Init wandb
        if callback_path is not None and wandb_logging:
            try:
                wandb.init(project='nnet', sync_tensorboard=True, name=callback_path)
            except Exception as e:
                print(str(e))

        # Is Compiled
        if not self.compiled:
            raise Exception("You must compile your model before training/testing.")

        # Mixed Precision Gradient Scaler
        self.grad_scaler = torch.cuda.amp.GradScaler(init_scale=grad_init_scale, enabled=(grad_init_scale != None) and (precision==torch.float16))# and ("cuda" in str(self.device)))
        if self.grad_scaler_state_dict is not None:
            self.grad_scaler.load_state_dict(self.grad_scaler_state_dict)
            self.grad_scaler_state_dict = None
        assert not (precision==torch.float16 and not self.grad_scaler.is_enabled()), "gradient scaling not enabled for float16 precision training!"

        # Anomaly Enabled
        torch.set_anomaly_enabled(detect_anomaly)

        # Init Training
        acc_step = 0

        # Zero Gradients
        self.zero_grad()

        # Callbacks
        if callback_path is not None:

            # Create Callback Dir
            if not os.path.isdir(callback_path):
                os.makedirs(callback_path, exist_ok=True)

            # Create Writer
            writer = SummaryWriter(os.path.join(callback_path, "logs"))

        else:

            writer = None

        # Try Catch
        try:

            # On Train Begin
            self.on_train_begin()

            # Training Loop
            for epoch in range(initial_epoch, epochs):

                # Init Iterator
                print("Epoch {}/{}:".format(epoch + 1, epochs))
                epoch_iterator = tqdm(dataset_train, total=steps_per_epoch * accumulated_steps if steps_per_epoch else None, dynamic_ncols=True, disable=verbose_progress_bar==0)

                # Init Epoch Dict
                epoch_losses = {}
                epoch_metrics = {}

                # Clear Infos
                self.reset_infos()

                # Training Mode
                self.train()

                # Epoch Begin
                self.on_epoch_begin(epoch=epoch + 1)

                # Epoch training loop
                for step, batch in enumerate(epoch_iterator):

                    # Clear display bar before build
                    if not self.built:
                        epoch_iterator.clear()

                    # Unpack Batch
                    inputs, targets = batch["inputs"], batch["targets"]

                    # Transfer Batch elt to model device
                    inputs = self.transfer_to_device(inputs)
                    targets = self.transfer_to_device(targets)

                    # Train Step
                    batch_losses, batch_metrics, acc_step = self.train_step(inputs=inputs, targets=targets, precision=precision, grad_scaler=self.grad_scaler, accumulated_steps=accumulated_steps, acc_step=acc_step, eval_training=eval_training)

                    # Update Epoch Loss and Metric
                    for key, value in batch_losses.items():
                        epoch_losses[key] = epoch_losses[key] + value.detach() if key in epoch_losses else value.detach().type(torch.float64)
                    for key, value in batch_metrics.items():
                        epoch_metrics[key] = epoch_metrics[key] + value.detach() if key in epoch_metrics else value.detach().type(torch.float64)

                    # Continue Accumulating
                    if acc_step > 0:
                        continue

                    # Step Print
                    if verbose_progress_bar >= 2:
                        self.display_step(epoch_losses, epoch_metrics, self.infos, epoch_iterator, step + 1)

                    # Logs Step
                    if writer is not None and self.model_step % step_log_period == 0:
                        self.log_step(losses=batch_losses, metrics=batch_metrics, infos=self.infos, writer=writer, step=self.model_step, tag="Training-step")

                    # On Batch End
                    self.on_step_end(
                        evaluate=self.model_step % eval_period_step == 0 if eval_period_step != None else False,
                        save=False, 
                        log_figure=self.model_step % log_figure_period_step == 0 if log_figure_period_step != None else False, 
                        callback_path=callback_path, 
                        epoch=epoch + 1,
                        step=self.model_step, 
                        inputs=inputs, 
                        targets=targets, 
                        dataset_eval=dataset_eval, 
                        eval_steps=eval_steps, 
                        verbose_eval=verbose_eval,
                        writer=writer,
                        recompute_metrics=recompute_metrics,
                        keep_last_k=keep_last_k
                    )

                    # Step per Epoch
                    if steps_per_epoch is not None:
                        if step + 1 >= steps_per_epoch * accumulated_steps:
                            break

                # Mean loss
                for key, value in epoch_losses.items():
                    epoch_losses[key] = value / (steps_per_epoch * accumulated_steps if steps_per_epoch is not None else len(dataset_train))

                # Mean Metrics
                for key, value in epoch_metrics.items():
                    epoch_metrics[key] = value / (steps_per_epoch * accumulated_steps if steps_per_epoch is not None else len(dataset_train))

                # Logs Epoch
                if writer is not None:
                    self.log_step(losses=epoch_losses, metrics=epoch_metrics, infos={}, writer=writer, step=epoch + 1, tag="Training-epoch")

                # On Epoch End
                self.on_epoch_end(
                    evaluate=(epoch + 1) % eval_period_epoch == 0 if eval_period_epoch != None else False,
                    save=(epoch + 1) % saving_period_epoch == 0 if saving_period_epoch != None else False, 
                    log_figure=(epoch + 1) % log_figure_period_epoch == 0 if log_figure_period_epoch != None else False, 
                    callback_path=callback_path, 
                    epoch=epoch + 1, 
                    inputs=inputs, 
                    targets=targets, 
                    dataset_eval=dataset_eval, 
                    eval_steps=eval_steps, 
                    verbose_eval=verbose_eval,
                    writer=writer,
                    recompute_metrics=recompute_metrics,
                    keep_last_k=keep_last_k
                )

        # Exception Handler
        except Exception as e:

            if writer is not None:
                writer.add_text("Exceptions", "Date: {} \n{}".format(time.ctime(), str(e)), self.model_step)

            raise e

    def _evaluate(self, dataset, writer, eval_steps=None, verbose=0, recompute_metrics=False, tag="Evaluation", verbose_progress_bar=1):
        
        # Evaluation Dataset
        if dataset is not None:

            # Dataset to list
            if not isinstance(dataset, list):
                dataset = [dataset]

            # Eval Datasets loop
            for dataset_i, dataset in enumerate(dataset):

                # Evaluate
                val_losses, val_metrics = self.evaluate(dataset, eval_steps, verbose, recompute_metrics, verbose_progress_bar)

                # Print
                self.print_step(val_losses, val_metrics, "eval")

                # Log
                if writer is not None:
                    self.log_step(losses=val_losses, metrics=val_metrics, infos={}, writer=writer, step=self.model_step, tag=os.path.join(tag, str(dataset_i)))

    def evaluate(self, dataset_eval, eval_steps=None, verbose=0, recompute_metrics=False, verbose_progress_bar=1):

        # Evaluation Mode
        self.eval()

        # Clear Infos
        self.reset_infos()

        # Init Epoch Dict
        epoch_losses = {}
        epoch_metrics = {}
        if recompute_metrics:
            epoch_truths = {}
            epoch_preds = {}

        # tqdm Iterator
        eval_iterator = tqdm(dataset_eval, total=eval_steps, dynamic_ncols=True, disable=verbose_progress_bar==0)

        # Evaluation Loop
        for step, batch in enumerate(eval_iterator):

            # Unpack Batch
            inputs, targets = batch["inputs"], batch["targets"]

            # Transfer Batch elt to model device
            inputs = self.transfer_to_device(inputs)
            targets = self.transfer_to_device(targets)

            # Eval Step
            batch_losses, batch_metrics, batch_truths, batch_preds = self.eval_step(inputs, targets, verbose)

            # Update Epoch Dict
            for key, value in batch_losses.items():
                epoch_losses[key] = epoch_losses[key] + value if key in epoch_losses else value.type(torch.float64)
            for key, value in batch_metrics.items():
                epoch_metrics[key] = epoch_metrics[key] + value if key in epoch_metrics else value.type(torch.float64)
            if recompute_metrics:
                for key, value in batch_truths.items():
                    epoch_truths[key] = epoch_truths[key] + value if key in epoch_truths else value
                for key, value in batch_preds.items():
                    epoch_preds[key] = epoch_preds[key] + value if key in epoch_preds else value

            # Step print
            if verbose_progress_bar >= 2:
                self.display_step(epoch_losses, epoch_metrics, self.infos, eval_iterator, step + 1)

            # Evaluation Steps
            if eval_steps:
                if step + 1 >= eval_steps:
                    break

        # Mean loss
        for key, value in epoch_losses.items():
            epoch_losses[key] = value / (eval_steps if eval_steps is not None else len(dataset_eval))

        # Recompute Metrics
        if recompute_metrics:
            for key in epoch_metrics.keys():
                epoch_metrics[key] = self.metrics["outputs"](epoch_truths[key], epoch_preds[key]) # fix metrics key
        # Mean Metrics
        else:
            for key, value in epoch_metrics.items():
                epoch_metrics[key] = value / (eval_steps if eval_steps is not None else len(dataset_eval))

        return epoch_losses, epoch_metrics
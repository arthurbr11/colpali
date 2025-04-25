import os

import datasets
import torch
from torch.utils.data import DataLoader
from transformers import Trainer
from transformers.trainer_utils import seed_worker
from transformers.utils import is_datasets_available

import wandb

os.environ["WANDB_PROJECT"] = "ColSmolDocling"  # Disable parallelism for tokenizers to avoid warnings


class ContrastiveTrainer(Trainer):
    def __init__(self, loss_func, is_vision_model, shuffle=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_func = loss_func
        self.is_vision_model = is_vision_model  # Unused argument, will be removed in 0.4.0
        self.added_gpu_count = False  # Flag to check if GPU count is already added to wandb config
        self.shuffle = shuffle

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # Add GPU count to wandb config
        if not self.added_gpu_count:
            if wandb.run is not None:
                wandb.config.update({"gpu_count": wandb.run._metadata.gpu_count})
                self.added_gpu_count = True

        query_outputs = model(input_ids=inputs["query_input_ids"], attention_mask=inputs["query_attention_mask"])
        # feed only kwargs with 'doc_' prefix
        doc_outputs = model(**{k[4:]: v for k, v in inputs.items() if k.startswith("doc")})
        if "neg_doc_input_ids" in inputs:
            neg_doc_outputs = model(**{k[8:]: v for k, v in inputs.items() if k.startswith("neg_doc")})
            loss = self.loss_func(query_outputs, doc_outputs, neg_doc_outputs)
            return (loss, (query_outputs, doc_outputs, neg_doc_outputs)) if return_outputs else loss

        loss = self.loss_func(query_outputs, doc_outputs)
        return (loss, (query_outputs, doc_outputs)) if return_outputs else loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=True):
        """This function is used to generate predictions and return the loss for the given inputs."""
        if not prediction_loss_only:
            raise ValueError("prediction_step is only called with prediction_loss_only=True")

        with torch.no_grad():
            # feed only kwargs with 'doc_' prefix
            doc_outputs = model(**{k[4:]: v for k, v in inputs.items() if k.startswith("doc")})
            query_outputs = model(
                input_ids=inputs["query_input_ids"],
                attention_mask=inputs["query_attention_mask"],
            )
            if "neg_doc_input_ids" in inputs:
                neg_doc_outputs = model(**{k[8:]: v for k, v in inputs.items() if k.startswith("neg_doc")})
                loss = self.loss_func(query_outputs, doc_outputs, neg_doc_outputs)
                return loss, None, None

            loss = self.loss_func(query_outputs, doc_outputs)
            return loss, None, None

    def get_train_dataloader(self) -> DataLoader:
        """
        Returns the training [~torch.utils.data.DataLoader].

        Will use no sampler if train_dataset does not implement __len__, a sequential sampler otherwise.

        Subclass and override this method if you want to inject some custom behavior.
        """
        if self.shuffle:
            return super().get_train_dataloader()
        else:
            if self.train_dataset is None:
                raise ValueError("Trainer: training requires a train_dataset.")

            train_dataset = self.train_dataset
            data_collator = self.data_collator
            if is_datasets_available() and isinstance(train_dataset, datasets.Dataset):
                train_dataset = self._remove_unused_columns(train_dataset, description="training")
            else:
                data_collator = self._get_collator_with_removed_columns(data_collator, description="training")

            dataloader_params = {
                "batch_size": self._train_batch_size,
                "collate_fn": data_collator,
                "num_workers": self.args.dataloader_num_workers,
                "pin_memory": self.args.dataloader_pin_memory,
                "persistent_workers": self.args.dataloader_persistent_workers,
            }

            if not isinstance(train_dataset, torch.utils.data.IterableDataset):
                # Use SequentialSampler instead of the default random sampler
                dataloader_params["sampler"] = torch.utils.data.SequentialSampler(train_dataset)
                dataloader_params["drop_last"] = self.args.dataloader_drop_last
                dataloader_params["worker_init_fn"] = seed_worker
                dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor

            return self.accelerator.prepare(DataLoader(train_dataset, **dataloader_params))

from typing import Any

import torch
import wandb
from torch import nn
import numpy as np
from transformers import Trainer

from utils_ner import NerDataset, compute_metrics_direct, align_predictions


class CustomTrainer(Trainer):
    example_sentence_id = 0
    
    def __init__(self, label_map: dict, *args, **kwargs):
        self.label_map = label_map
        super().__init__(*args, **kwargs)
    
    def training_step(self, model: nn.Module, inputs: Any) -> torch.Tensor:
        model.train()
        inputs = self._prepare_inputs(inputs)
       
        labels = inputs["labels"].cpu().detach().numpy()
        with self.compute_loss_context_manager():
            loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
            preds = outputs.logits.cpu().detach().numpy()
            del outputs

        del inputs
        torch.cuda.empty_cache()

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        if self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            self.accelerator.backward(loss)
            
            
        metrics = compute_metrics_direct(preds, labels, self.label_map)
        self.log(metrics)
        return loss.detach()
            
    def predict(self, test_dataset: NerDataset, *args, **kwargs) -> tuple[torch.Tensor, torch.Tensor, dict]:
        predictions, label_ids, metrics = super().predict(test_dataset, *args, **kwargs)
            
        self.log(metrics)
        table = wandb.Table(columns=["Text", "Predicted", "True"])
        words = test_dataset.examples[self.example_sentence_id].words
        preds_list, out_label_list = align_predictions(predictions, label_ids, self.label_map)
        for text, pred, label in zip(words, preds_list[self.example_sentence_id][:len(words)], out_label_list[self.example_sentence_id][:len(words)]):
            table.add_data(text, pred, label)
        self.log({"examples": table})
        return predictions, label_ids, metrics


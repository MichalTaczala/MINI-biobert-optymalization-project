from transformers import Trainer
from custom_src.rms_prop import HybridSGDRMSprop


class CustomTrainer(Trainer):
    def create_optimizer(self):
        """
        This replaces the default optimizer with our custom hybrid optimizer.
        """
        optimizer_cls = HybridSGDRMSprop
        return optimizer_cls(
            self.model.parameters(),
            lr=self.args.learning_rate,
            alpha=0.99,
            eps=1e-8,
            weight_decay=0.01,
            momentum=0.9,
        )

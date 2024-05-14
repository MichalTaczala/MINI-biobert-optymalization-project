from transformers import Trainer
from rms_prop.rms_prop_optimizer import RMSPropOptimizer


class RMSPropTrainer(Trainer):
    def create_optimizer(self):
        """
        This replaces the default optimizer with our custom hybrid optimizer.
        """
        if self.optimizer is None:
            optimizer_cls = RMSPropOptimizer
            self.optimizer = optimizer_cls(
                self.model.parameters(),
                lr=self.args.learning_rate,
                alpha=0.99,
                eps=1e-8,
                weight_decay=0.01,
                momentum=0.9,
            )
        return self.optimizer

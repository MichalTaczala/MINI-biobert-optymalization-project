from custom_trainer import CustomTrainer
from torch.optim import NAdam, SGD


class NADAM(CustomTrainer):
    def create_optimizer(self):
        """
        This replaces the default optimizer with our custom hybrid optimizer.
        """
        if self.optimizer is None:
            optimizer_cls = NAdam
            self.optimizer = optimizer_cls(
                self.model.parameters(),
                lr=self.args.learning_rate,
            )
        return self.optimizer


class Nesterov(CustomTrainer):
    def create_optimizer(self):
        """
        This replaces the default optimizer with our custom hybrid optimizer.
        """
        if self.optimizer is None:
            optimizer_cls = SGD
            self.optimizer = optimizer_cls(
                self.model.parameters(),
                lr=self.args.learning_rate,
                nesterov=True,
                momentum=0.9,
            )
        return self.optimizer

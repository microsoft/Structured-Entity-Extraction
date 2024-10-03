from abc import ABC, abstractmethod


class Trainer(ABC):
    @abstractmethod
    def train(
        self,
        save_path,
        model,
        train_dataloader,
        val_dataloader,
        optimizer,
        scheduler,
        epochs,
        **kwargs
    ):
        # training process
        raise NotImplementedError

    @abstractmethod
    def evaluate(self, model, test_dataloader, tokenizer, **kwargs):
        # testing process
        raise NotImplementedError

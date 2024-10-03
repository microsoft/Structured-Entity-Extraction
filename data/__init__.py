from abc import ABC, abstractmethod


class DataManager(ABC):
    @abstractmethod
    def create_dataset(
        self, file_path: str, use_data: int = None, max_length: int = 512, **kwargs
    ):
        # return a pytorch dataset
        raise NotImplementedError

    @abstractmethod
    def create_data_loader(
        self,
        file_path: str,
        use_data: int,
        max_length: int,
        batch_size: int,
        shuffle: bool,
        **kwargs
    ):
        # return pytorch data loaders for training, validation, test
        raise NotImplementedError

"""Configuration classes for baseline and ablation experiments."""

from dataclasses import dataclass


@dataclass
class BaselineConfig:
    """Baseline configuration."""
    name: str = "baseline"
    
    # Data
    dataset_name:str = 'ag_news'
    num_classes:int = 4

    # text encoding
    text_len:int = 1014
    alpha_len:int = 70

    # augmentation
    p:float = 0.5
    q:float = 0.5

    #model
    use_relu:bool = False

    # Training
    batch_size: int = 128
    num_epochs: int = 10
    learning_rate: float = 0.01
    momentum: float = 0.9
    optimizer_type:str ='sgd'
    decay : bool = False
    saving_epoch : int = 0

    # Misc
    seed: int = 7
    device: str = "cuda"


@dataclass
class Ablation1Config(BaselineConfig):
    name: str = "ablation1_shorten_text_len"
    text_len: int = 507


@dataclass
class Ablation2Config(BaselineConfig):
    name: str = "ablation2_use_adam_optimizer"
    optimizer_type:str ='adam'
    learning_rate: float = 0.001
    decay: bool = True


@dataclass
class Ablation3Config(BaselineConfig):
    name: str = "ablation3_use_Relu_in_FC"
    use_relu: bool = True


def get_config(config_name: str):
    """Get configuration by name."""
    configs = {
        'baseline': BaselineConfig(),
        'ablation1': Ablation1Config(),
        'ablation2': Ablation2Config(),
        'ablation3': Ablation3Config(),
    }
    if config_name not in configs:
        raise ValueError(f"Unknown config: {config_name}")
    return configs[config_name]

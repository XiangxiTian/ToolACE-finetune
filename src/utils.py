from datetime import datetime

import wandb
from config.base_config import Config
from config.qwen3_4b import get_config as get_qwen3_4b_config
from config.qwen3_14b import get_config as get_qwen3_14b_config


def load_config(config_path: str) -> Config:
	"""
		This function loads config object. It exist for the convenience of experimentation. 
  		Since experimentation time was limited, only two configs are supported at the moment.
  
 		Args:
 			config_path: Path to training config file (e.g., config/qwen3_14b.py)
 		
 		Returns:
 			Config object
 	"""
  
	if config_path == "config/qwen3_14b.py":
		return get_qwen3_14b_config()
	elif config_path == "config/qwen3_4b.py":
		return get_qwen3_4b_config()
	else:
		raise ValueError(f"Invalid config path: {config_path}, at the moment only config/qwen3_14b.py and config/qwen3_4b.py are supported. Add support for other configs if needed.")


def initialize_wandb(config: Config) -> tuple[list[str], str | None]:
    """
    Initialize Weights & Biases tracking.
    
    Args:
        config: Training configuration object
        
    Returns:
        A tuple of (report_to, run_name) where:
        - report_to: List of reporting backends (["wandb"] or [])
        - run_name: The wandb run name if enabled, None otherwise
    """
    if not config.wandb.enabled:
        print("Wandb is disabled, skipping initialization...")
        return [], None
    
    if config.wandb.run_name is None:
        model_short_name = config.model_name.split("/")[-1]
        date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"{model_short_name}_{date_str}"
    else:
        run_name = config.wandb.run_name
    
    wandb.init(
        project=config.wandb.project,
        name=run_name,
        entity=config.wandb.entity,
        tags=config.wandb.tags,
        notes=config.wandb.notes,
        config={
            "model_name": config.model_name,
            "lora_r": config.lora_r,
            "lora_alpha": config.lora_alpha,
            "learning_rate": config.learning_rate,
            "batch_size": config.per_device_train_batch_size,
            "num_epochs": config.num_train_epochs,
            "max_steps": config.max_steps,
        }
    )
    
    print(f" Wandb enabled: project={config.wandb.project}, run={run_name}")
    return ["wandb"], run_name


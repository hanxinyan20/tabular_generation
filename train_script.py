
from train import train_model
import os
from pathlib import Path
import os
import hydra
import traceback
from omegaconf import DictConfig, OmegaConf
from train import train_model
@hydra.main(version_base=None, config_path="/mnt/public/hxy/project/diff_scripts/config", config_name="default")
def main(cfg: DictConfig):
    exp = cfg.experiment_name
    results_folder = str(Path(cfg.results_base_dir) / exp)
    os.makedirs(results_folder, exist_ok=True)
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    cfg_dict["results_folder"] = results_folder
    with open(os.path.join(results_folder, 'config.yaml'), 'w') as f:
        f.write(OmegaConf.to_yaml(cfg))
    cfg_dict.pop("experiment_name", None)
    cfg_dict.pop("results_base_dir", None)
    print("Training with config:")
    print(cfg_dict)
    try:
        train_model(**cfg_dict)
    except Exception:
        traceback.print_exc()
        raise
if __name__ == "__main__":
    main()
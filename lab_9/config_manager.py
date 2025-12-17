import os
import yaml
from omegaconf import OmegaConf
from pathlib import Path
from typing import Dict, Any


class NumericalMethodsConfig:
    def __init__(self, config_dir: str = "config", env: str = None):
        """
        Initialize configuration manager.

        :param config_dir: Configuration directory path
        :param env: Environment (dev/prod)
        """
        self.config_dir = Path(config_dir)
        self.env = env or os.getenv("NUM_METHODS_ENV", "dev")
        self._config = self._load_config()

    def _load_config(self):
        """Load configuration from YAML files."""
        try:
            base_config = self.config_dir / "base.yaml"
            env_config = self.config_dir / f"{self.env}.yaml"
            
            base = OmegaConf.load(base_config)
            env = OmegaConf.load(env_config)
            
            return OmegaConf.merge(base, env)
        except Exception as e:
            raise RuntimeError(f"Failed to load configuration: {e}")


    @property
    def config(self):
        """Get configuration."""
        return self._config


    @property
    def debug(self):
        """Get debug flag."""
        return self._config.get('debug', False)
import numpy as np
import os
from pathlib import Path
from typing import Dict
from omegaconf import OmegaConf


class NumericalMethodsConfig:
    """
    Configuration manager for numerical methods.
    """
    def __init__(self, config_dir: str = "config", env: str = None):
        """
        Initialize configuration manager.

        :params:
            config_dir: str - Path to configuration directory
            env: str - Environment (dev/prod)
        """
        self.config_dir = Path(config_dir)
        self.env = env or os.getenv("NUM_METHODS_ENV", "dev")
        self._config = self._load_config()


    @property
    def config(self):
        """
        Get configuration.
        """
        return self._config


    @property
    def debug(self):
        """
        Get debug flag.
        """
        return self._config.get('debug', False)


    def _load_config(self):
        """
        Load configuration.
        """
        try:
            configs = []
            
            base_config = self.config_dir / "base.yaml"
            if base_config.exists():
                configs.append(OmegaConf.load(base_config))
            
            methods_config = self.config_dir / "methods.yaml"
            if methods_config.exists():
                configs.append(OmegaConf.load(methods_config))
            
            env_config = self.config_dir / f"{self.env}.yaml"
            if env_config.exists():
                configs.append(OmegaConf.load(env_config))
            
            merged = OmegaConf.merge(*configs)
            OmegaConf.set_readonly(merged, True)
            return merged
            
        except Exception as e:
            raise RuntimeError(f"Configuration loading error: {e}")


    def should_show_output(self, output_type: str) -> bool:
        """
        Check if output should be displayed.

        :params:
            output_type: str - Output type (table/plot)
            
        :returns:
            bool - True if output should be shown
        """
        match output_type:
            case "table":
                return self._config.output.get('show_tables', True)
            case "plot":
                return self._config.output.get('show_plots', True)
            case _:
                return False


    def get_method_params(self, method_name: str) -> Dict:
        """
        Get parameters for a method.

        :params:
            method_name: str - Method name
            
        :returns:
            dict - Method parameters
        """
        match method_name:
            case "qr":
                return {
                    "method": self._config.qr.get('method', 'gram_schmidt'),
                    "tolerance": self._config.qr.get('tolerance', 1e-10)
                }
            case "seidel":
                return {
                    "max_iterations": self._config.linear_systems.seidel.get(
                        'max_iterations', 100
                    ),
                    "tolerance": self._config.linear_systems.seidel.get(
                        'tolerance', 1e-3
                    )
                }
            case "bisection":
                return {
                    "max_iterations": self._config.nonlinear_equations.bisection.get(
                        'max_iterations', 100
                    ),
                    "tolerance": self._config.nonlinear_equations.bisection.get(
                        'tolerance', 1e-3
                    )
                }
            case "combined":
                return {
                    "max_iterations": self._config.nonlinear_equations.combined_method.get(
                        'max_iterations', 100
                    ),
                    "tolerance": self._config.nonlinear_equations.combined_method.get(
                        'tolerance', 1e-5
                    )
                }
            case "newton_system":
                return {
                    "max_iterations": self._config.nonlinear_systems.newton.get(
                        'max_iterations', 100
                    ),
                    "tolerance": self._config.nonlinear_systems.newton.get(
                        'tolerance', 1e-4
                    ),
                    "initial_guess": self._config.nonlinear_systems.newton.get(
                        'initial_guess', [0.5, 0.8]
                    )
                }
            case _:
                return {}


_config = None


def get_config(env: str = None) -> NumericalMethodsConfig:
    """
    Get global configuration.

    :params:
        env: str - Environment (dev/prod)
        
    :returns:
        NumericalMethodsConfig - Configuration instance
    """
    global _config
    if _config is None:
        _config = NumericalMethodsConfig(env=env)
    return _config

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
    
    
    def should_show_output(self, output_type: str) -> bool:
        """
        Check if output type should be shown.
        
        :params:
            output_type: str - Type of output (table/plot)
            
        :returns:
            bool - True if output should be shown, False otherwise
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
        Get parameters for specific method.
        
        :params:
            method_name: str - Name of the method
            
        :returns:
            Dict - Method parameters
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

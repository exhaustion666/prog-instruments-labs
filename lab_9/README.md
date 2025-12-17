A Python implementation of numerical methods with layered configuration using OmegaConf.

## Features

- QR decomposition (Gram-Schmidt method)
- Linear system solving (QR and Seidel methods)
- Nonlinear equation solving (bisection and combined methods)
- Nonlinear system solving (Newton's method)
- Layered configuration with environment support

## Installation

pip install -r requirements.txt

## Usage

# Run all methods in development environment
python numerical_methods.py --env dev

# Run specific method
python numerical_methods.py --env dev --method qr
python numerical_methods.py --env dev --method seidel
python numerical_methods.py --env dev --method nonlinear
python numerical_methods.py --env dev --method system

# Run in production environment
python numerical_methods.py --env prod --method all
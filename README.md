# PRNG Tester

PRNG Tester is a Python project designed to test and analyze the predictability of different pseudorandom number generators (PRNGs) using a neural network model. This project uses various PRNG algorithms to generate datasets, trains a neural network on this data, and evaluates the performance of the model.

## Features

- Supports multiple PRNG algorithms.
- Generates datasets based on PRNG outputs.
- Trains a neural network model using the generated data.
- Evaluates the performance of the trained model.
- Saves and loads datasets and trained models.

## Installation

### Prerequisites

- Python 3.7+
- `numpy`
- `scikit-learn`
- `keras`
- `rich`
- `matplotlib`
- `Pillow`

### Clone the Repository

```bash
git clone https://github.com/your-username/prng-tester.git
cd prng-tester
```

### Install Dependencies

```bash
pip install numpy scikit-learn keras rich
```

## Usage

### Creating and Training a PRNG Model

Define PRNG Tester Class: Extend the PRNG_tester class to define a specific PRNG or use one of the 2 Bundled ones.
```python
class YourPRNG(PRNG_tester):
    def __init__(self, seed: int, n_datapoints: int, length_of_data: int, console: Console):
        PRNG_tester.__init__(self, seed, n_datapoints, length_of_data, console, "YourPRNG")
        # Initialize your PRNG here

    def prng(self):
        # Implement your PRNG logic here
        return random_number
```

### Train and Save Model
```python
from rich.console import Console

console = Console()
your_prng = YourPRNG(seed=42, n_datapoints=1000000, length_of_data=1000, console=console)
your_prng.create_data()
your_prng.train()
```
### Test Model
```python
your_prng.test()
```

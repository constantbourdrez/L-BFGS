This project explores the use of advanced optimization techniques on the mushrooms dataset from the LibSVM repository. The goal is to analyze the performance of several gradient-based optimization methods and their ability to efficiently converge while minimizing computational costs.

## Overview

The project demonstrates the implementation of the following optimization techniques:
- **Gradient Descent (GD)**
- **Stochastic Gradient Descent (SGD)**
- **Adagrad**
- **Proximal versions of GD and SGD**
- **BFGS (Broyden–Fletcher–Goldfarb–Shanno)**
- **L-BFGS (Limited-memory BFGS)**

Each optimization method is applied to the mushrooms dataset, and the results are compared in terms of both convergence rates and computational efficiency.

## Project Structure

- `opt_methods.py`: Script to find the optimization algorithms.
- `problem.py`: Contains the implementations of the problem class.
- `utils.py`: Function for computing the Lipschitz constant.
- `comparison_methods.ipynb`: Notebook where the evaluation of the algorithms is made.
- `requirements.txt`: List of required Python libraries for running the project.

You can install the dependencies with:

```bash
pip install -r requirements.txt

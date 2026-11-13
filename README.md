# Neural Networks from Scratch

**Implementations of neural networks from first principles**, following the book [*Neural Networks and Deep Learning*](http://neuralnetworksanddeeplearning.com/) by Michael Nielsen.

The objective of this project is to explore the mathematical foundations and computational aspects of neural networks, implementing the underlying algorithms using Python and NumPy.

---

## Repository Structure

```text
NeuralNetworksFromScratch/
├── .gitignore
├── README.md
├── NotesOnFeedforwardNeuralNetworks.pdf
├── data/
├── experiments/
│   ├── 01_learning_curves.py
│   ├── 02_evolution_of_parameters.py
│   └── 03_the_architecture.py
├── main.py
├── mnist_loader.py
├── network.py
├── pyproject.toml
└── uv.lock
```

## Implementations

The repository includes implementations of:

- **Feedforward Neural Networks**
- **Stochastic Gradient Descent (SGD)**
- **Backpropagation**
- **Mini-batch training**
- **MNIST handwritten digit classification**

All algorithms are implemented using Python and NumPy, without relying on high-level deep learning frameworks.

## Experiments

The `experiments/` directory contains numerical experiments investigating different aspects of neural network training.

| Experiment | Description |
|:---|:---|
| **Learning Curves** | Analysis of training and test performance. |
| **Evolution of Parameters** | Study of the evolution of weights and biases during training. |
| **Network Architecture** | Investigation of the influence of network width and architecture on classification performance. |

## Theoretical Notes

Alongside the implementations, I am developing a set of lecture notes covering the mathematical foundations of feedforward neural networks.

The notes include mathematical derivations, theoretical discussions, and illustrative examples.

> **Status: Work in Progress**
>
> These lecture notes are currently under development. The material is being progressively expanded, revised, and supplemented with mathematical derivations, illustrations, and practical examples.

📄 **[Read the Lecture Notes](./NotesOnFeedforwardNeuralNetworks.pdf)**

## Installation and Usage

The project uses Python, NumPy and [uv](https://docs.astral.sh/uv/) for dependency management.

### Requirements

- Python
- uv

### Installation

Clone the repository:

```bash
git clone https://github.com/valvarezcazador/NeuralNetworksFromScratch.git
cd NeuralNetworksFromScratch
```

Install the dependencies:

```bash
uv sync
```

### Running the project

```bash
uv run python main.py
```

## References

- Nielsen, M. A. (2015). *Neural Networks and Deep Learning*. Determination Press. [Online book](http://neuralnetworksanddeeplearning.com/).

---

*This repository is an ongoing learning and implementation project.*

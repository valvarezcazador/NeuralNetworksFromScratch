Neural Networks from Scratch

This repository contains implementations of neural networks from first principles, following the book Neural Networks and Deep Learning by Michael Nielsen.

The main objective is to explore the mathematical foundations and computational aspects of neural networks, implementing the underlying algorithms using Python and NumPy.

Repository Structure

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

Implementations

The repository includes implementations of:

Feedforward Neural Networks.

Stochastic Gradient Descent (SGD).

Backpropagation algorithm.

Mini-batch training.

MNIST handwritten digit classification.

The algorithms are implemented using Python and NumPy, without relying on high-level deep learning frameworks.

Experiments

The experiments/ directory contains numerical experiments investigating different aspects of neural network training.

Experiment

Description

Learning Curves

Analysis of training and test performance.

Evolution of Parameters

Study of the evolution of weights and biases during training.

Network Architecture

Investigation of the influence of network width and architecture on classification performance.

Theoretical Notes

Alongside the implementations, I am developing a set of lecture notes covering the mathematical foundations of feedforward neural networks.

The notes include mathematical derivations, theoretical discussions, and illustrative examples.

Status: Work in Progress.

The latest version is available here:

Notes on Feedforward Neural Networks (PDF)

Installation and Usage

The project uses Python, NumPy and uv for dependency management.

Requirements

Python

uv

Installation

Clone the repository:

git clone https://github.com/valvarezcazador/NeuralNetworksFromScratch.git
cd NeuralNetworksFromScratch

Install the dependencies:

uv sync

Running the project

uv run python main.py

References

Nielsen, M. A. (2015). Neural Networks and Deep Learning. Determination Press. Online book.

This repository is an ongoing learning and implementation project.
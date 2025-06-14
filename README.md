# Mo-net

## Overview

This project is a simple implementation of a deep learning library. The goal of
this project is to implement deep learning concepts and functions by hand using
jax/numpy without using popular libraries such as PyTorch or TensorFlow by
understanding the underlying maths and reading papers.

![Dashboard](/assets/dashboard.png)

![Infer](/assets/after-grokking.png)

## Usage

```shell
uv sync
uv run train --quickstart mnist_mlp
```

## Features

- Implementations for some normalisation types.
- Implementation of an Adam optimiser.
- Data parallelism using multiple CPU cores for parallel training.
- Web server monitoring interface.
- Logging to a database.

![cli](./assets/cli-interface.png)

## Motivation

PyTorch and TensorFlow exist. Why do this? I found that following recipes to
build neural networks using existing libraries did not teach me the concepts at
a level of understanding I sought. So I went another way. Using just an
understanding of the underlying mathematics, and reading the relevant papers, I
seek to understand the significance of different deep learning techniques by
iterating upon implementations of them. Rapidly, one learns what works, what is
important to be super precise in implementing, but also, surprisingly, just how
resilient neural networks can be to implementation errors. The process of
ensuring correctness of implementation can only be learnt by doing.
Debugging neural networks is very different from debugging procedural code.

## Performance

This library is not performant. Nor is it meant to be. Its primary reason for
existing is pedagogical. And as a result, many of the implementation details are
far simpler than is the case for proper, production-grade deep learning
libraries. Attempts have been made to improve performance where possible, and
this process has been very instructive in its own right.

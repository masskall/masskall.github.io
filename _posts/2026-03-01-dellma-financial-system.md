title: "LLM-Based Financial Decision Support System under Uncertainty"
date: 2026-03-01
categories:
  - Projects
tags:
  - Prompt Engineering
  - LLM
  - Decision Theory
  - Quantitative Analysis
---

### Project Overview
Inspired by cutting-edge research from ICLR 2025 (DeLLMa framework), this project explores the application of Large Language Models (LLMs) in complex, highly uncertain financial environments. Instead of treating the AI as a "black box" predictor, this project focused on building a **Human-Auditable** decision-making pipeline.

### Core Architecture & Backtesting
* **State Forecasting:** Utilized historical stock market data (24-month horizon) to generate probabilistic state predictions.
* **Utility Elicitation:** Translated abstract financial goals (e.g., maximizing monthly ROI while controlling drawdown) into concrete mathematical utility functions.
* **Quantitative Backtesting:** Ran Monte Carlo simulations to estimate the Expected Utility of various investment actions across massive state spaces.

### Methodological Innovation
Unlike traditional Zero-Shot or Chain-of-Thought (CoT) prompting, this system integrates classical **Decision Theory**. The AI does not just output a "buy/sell" command; it outputs a detailed probability matrix and expected utility calculation that human analysts can audit.

### Code Snippet (Expected Utility Calculation)
```python
import numpy as np

def calculate_expected_utility(state_probs, action_utilities):
    """
    Calculate Expected Utility using Monte Carlo estimation.
    state_probs: dict of probabilities for each market state
    action_utilities: matrix of utilities for each action given a state
    """
    expected_utilities = {}
    for action in action_utilities.keys():
        # E[U] = Sum( P(s) * U(a, s) )
        eu = sum(state_probs[state] * util for state, util in action_utilities[action].items())
        expected_utilities[action] = eu
        
    best_action = max(expected_utilities, key=expected_utilities.get)
    return best_action, expected_utilities

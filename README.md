# Dynamic Recovery Window Model in Pediatric Surgical Emergencies Using Lactate Clearance, Capillary Refill, and Delay Sensitive Perfusion State Transitions


A mathematical proof-of-concept framework for studying time-dependent recovery potential in pediatric surgical emergencies.

The model represents intervention delay as a dynamic interaction among lactate, capillary refill, perfusion reserve, recovery reserve, injury accumulation, latent physiological regime transitions, and intervention timing.

## Overview

Children requiring emergency surgery may deteriorate while awaiting definitive intervention. The effect of waiting, however, may depend not only on elapsed time but also on the evolving physiological state and remaining recovery capacity.

This repository implements a literature-parameterized Dynamic Recovery Window model that combines:

- nonlinear ordinary differential equations for continuous physiological states;
- a four-state continuous-time Markov chain for latent regime transitions;
- delay-dependent recovery and injury processes;
- global uncertainty and sensitivity analysis;
- scenario-based stress testing;
- numerical robustness analysis; and
- a finite-horizon optimal-stopping decision layer.

The framework is intended as a mathematical and computational proof-of-concept rather than a validated clinical prediction model.

## Model Structure

The continuous state vector is

\[
x(t) = [L(t), C(t), P(t), R(t), I(t)]^T,
\]

where:

- `L(t)` = blood lactate;
- `C(t)` = capillary refill time;
- `P(t)` = perfusion reserve;
- `R(t)` = recovery reserve;
- `I(t)` = injury burden.

An additional delay-load process `D(t)` represents the accumulating effect of waiting before intervention and its decay after intervention.

The continuous system is coupled to four latent physiological regimes:

1. Compensated
2. Vulnerable
3. Decompensating
4. Refractory

These regimes are represented through a state-dependent continuous-time Markov chain.

## Dynamic Recovery Window

A terminal recovery index is defined as

\[
G = [P R (1-I)]^{1/3}.
\]

The model classifies terminal states into reversible, partially reversible, and irreversible regions using predefined structural boundaries.

The Dynamic Recovery Window represents the latest evaluated intervention delay for which the simulated terminal state remains within the predefined reversible class.

This quantity is a model-derived mathematical boundary and must not be interpreted as a clinically validated safe-delay threshold.

## Optimal-Stopping Layer

A separate finite-horizon decision layer evaluates candidate intervention delays using a provisional objective function combining terminal perfusion reserve, recovery reserve, residual injury, lactate burden, waiting cost, and intervention cost.

The recovery-window boundary and the optimal-stopping solution are distinct mathematical quantities:

- the recovery window describes the boundary of the predefined reversible region;
- the optimal-stopping solution identifies the candidate delay maximizing the provisional decision objective.

The decision weights and penalties are structural assumptions and have not been calibrated using patient-level outcomes.

## Computational Analyses

The repository includes:

- reference trajectory simulation;
- intervention-delay continuation analysis;
- scenario stress testing;
- Latin-hypercube uncertainty analysis;
- partial rank correlation coefficient (PRCC) sensitivity analysis;
- numerical solver robustness checks;
- optimal-stopping surface analysis; and
- internal synthetic model-recovery analysis using Akaike's information criterion (AIC).

## Repository Structure

```text
pediatric-dynamic-recovery-window-model/
├── code/          # Python source code
├── data/          # Model input and literature-reference data
├── figures/       # Generated manuscript figures
├── manuscript/    # Full manuscript
├── model/         # Model parameter files
├── results/       # Numerical model outputs
├── CITATION.cff   # Citation metadata
├── LICENSE        # MIT License
└── README.md

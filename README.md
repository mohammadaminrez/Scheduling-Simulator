# Queue Simulation System

A sophisticated queue simulation system that implements different scheduling policies and distributions for analyzing queue behavior under various conditions.

## Overview

This project simulates a system with multiple servers and queues, implementing different scheduling policies and probability distributions. It's particularly useful for studying queue behavior under various load conditions and scheduling strategies.

### Key Features

- Multiple queue implementations:
  - Standard FIFO queues with exponential distributions
  - Weibull distribution-based queues
  - LIFO (Last-In-First-Out) preemptive queues
- Configurable parameters:
  - Number of servers/queues
  - Arrival rate (λ)
  - Service rate (μ)
  - Number of queue choices (d)
  - Shape parameter for Weibull distribution
- Visualization capabilities:
  - Queue length distribution plots
  - Performance metrics
  - CSV data export

## Requirements

- Python 3.x
- Required Python packages:
  - matplotlib
  - collections
  - random
  - argparse
  - csv
  - logging

## Usage

The simulation can be run with various command-line arguments to configure the behavior:

```bash
python queue_sim.py [options]
```

### Basic Options

- `--lambd`: Arrival rate (default: [0.5, 0.7, 0.9, 0.99])
- `--mu`: Service rate (default: 1)
- `--max-t`: Maximum simulation time (default: 1,000,000)
- `--n`: Number of queues (default: 10)
- `--d`: Number of queue choices (default: 5)
- `--shape`: Shape parameter for Weibull distribution (default: 0.5)
- `--plot_interval`: Interval for collecting plot data points (default: 1)
- `--seed`: Random seed for reproducibility
- `--verbose`: Enable verbose logging
- `--csv`: Output file for CSV results

### Examples

1. Run basic simulation with default parameters:
```bash
python queue_sim.py
```

2. Run Weibull distribution simulation with custom parameters:
```bash
python queue_sim_weibull.py --lambd 0.8 --mu 1 --n 20 --d 3 --shape 0.7
```

3. Run LIFO preemptive simulation:
```bash
python queue_sim_weibull_lifo.py --lambd 0.9 --mu 1 --n 15 --d 4
```

## Output

The simulation provides:
1. Average time spent in the system (W)
2. Theoretical expectations (when applicable)
3. Queue length distribution plots
4. Optional CSV output with detailed metrics

## Implementation Details

The system implements three main variants:

1. **Standard Queue (queue_sim.py)**
   - Uses exponential distributions
   - FIFO (First-In-First-Out) scheduling
   - Basic supermarket model implementation

2. **Weibull Queue (queue_sim_weibull.py)**
   - Uses Weibull distributions for both arrivals and service times
   - Configurable shape parameter
   - FIFO scheduling

3. **LIFO Preemptive Queue (queue_sim_weibull_lifo.py)**
   - Uses Weibull distributions
   - LIFO (Last-In-First-Out) scheduling
   - Preemptive policy where new jobs can interrupt running jobs

## Contributing

Feel free to submit issues and enhancement requests! 
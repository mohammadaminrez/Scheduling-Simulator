# Queue Simulation System

A flexible queue simulation system for analyzing queue behavior under various scheduling policies and distributions, now consolidated into a single script: `queue_sim.py`.

## Overview

This project simulates a system with multiple servers and queues, supporting different scheduling policies and probability distributions. It is useful for studying queue behavior under various load conditions and scheduling strategies.

### Key Features

- Multiple queue policies:
  - FIFO (First-In-First-Out)
  - LIFO (Last-In-First-Out) with preemption
- Multiple distributions:
  - Exponential (memoryless)
  - Weibull (configurable shape parameter)
- Configurable parameters:
  - Number of servers/queues
  - Arrival rate (λ)
  - Service rate (μ)
  - Number of queue choices (d)
  - Shape parameter for Weibull distribution
- Visualization:
  - Queue length distribution plots
  - Performance metrics
  - Optional CSV data export

## Requirements

- Python 3.x
- Required Python packages:
  - matplotlib
  - collections (standard library)
  - random (standard library)
  - argparse (standard library)
  - csv (standard library)
  - logging (standard library)

Install matplotlib if needed:
```bash
pip install matplotlib
```

## Usage

All simulation types and options are now available via `queue_sim.py`:

```bash
python queue_sim.py [options]
```

### Main Options

- `--lambd`: Arrival rate (default: [0.5, 0.9, 0.95, 0.99])
- `--mu`: Service rate (default: 1)
- `--max-t`: Maximum simulation time (default: 1,000)
- `--n`: Number of queues/servers (default: 1,000)
- `--d`: Number of queue choices (default: [1, 2, 5, 10])
- `--shape`: Shape parameter for Weibull distribution (default: 1)
- `--plot_interval`: Interval for collecting plot data points (default: 1)
- `--seed`: Random seed for reproducibility
- `--verbose`: Enable verbose logging
- `--csv`: Output file for CSV results
- `--queue-policy`: Queue policy (`fifo` or `lifo`, default: `fifo`)
- `--distribution`: Distribution type (`exponential` or `weibull`, default: `exponential`)

### Examples

1. **Run basic simulation with default parameters:**
   ```bash
   python queue_sim.py
   ```

2. **Run Weibull distribution simulation with custom parameters:**
   ```bash
   python queue_sim.py --distribution weibull --shape 0.7 --lambd 0.8 --mu 1 --n 20 --d 3
   ```

3. **Run LIFO preemptive simulation:**
   ```bash
   python queue_sim.py --queue-policy lifo --distribution weibull --lambd 0.9 --mu 1 --n 15 --d 4
   ```

## Output

The simulation provides:
- Average time spent in the system (W)
- Theoretical expectations (when applicable)
- Queue length distribution plots
- Optional CSV output with detailed metrics

## Implementation Details

All features are implemented in `queue_sim.py`:
- **Queue Policy:** FIFO or LIFO (preemptive)
- **Distributions:** Exponential or Weibull (with configurable shape)
- **Supermarket Model:** Jobs choose the shortest of `d` randomly selected queues

## Contributing

Feel free to submit issues and enhancement requests! 
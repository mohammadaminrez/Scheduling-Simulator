#!/usr/bin/env python3

import argparse
import csv
import collections
import logging
import matplotlib.pyplot as plt
from random import expovariate, sample, seed

from discrete_event_sim import Simulation, Event
from workloads import weibull_generator

# columns saved in the CSV file
CSV_COLUMNS = ['lambd', 'mu', 'max_t', 'n', 'd', 'shape', 'queue_policy', 'distribution', 'w']


class Queues(Simulation):
    """Simulation of a system with n servers and n queues.

    The system has n servers with one queue each. Jobs arrive at rate lambd and are served at rate mu.
    When a job arrives, according to the supermarket model, it chooses d queues at random and joins
    the shortest one.

    Features:
    - Queue Policy: FIFO (First In First Out) or LIFO (Last In First Out) with preemption
    - Distribution: Exponential (memoryless) or Weibull distribution
        - Weibull shape = 1: Same as exponential distribution
        - Weibull shape < 1: Heavy-tailed distribution
        - Weibull shape > 1: More uniform/bell-shaped distribution
    """

    def __init__(self, lambd, mu, n, d, queue_policy='fifo', distribution='exponential', 
                 plot_interval=1, shape=1):
        super().__init__()
        self.running = [None] * n  # if not None, the id of the running job (per queue)
        self.queues = [collections.deque() for _ in range(n)]  # queues of the system
        self.arrivals = {}  # dictionary mapping job id to arrival time
        self.completions = {}  # dictionary mapping job id to completion time
        self.remaining_time = {}  # dictionary mapping job id to remaining service time (for LIFO)
        self.lambd = lambd
        self.n = n
        self.d = d
        self.mu = mu
        self.queue_policy = queue_policy.lower()
        self.distribution = distribution.lower()
        self.shape = shape
        self.arrival_rate = lambd * n  # frequency of new jobs is proportional to the number of queues
        self.plots = []  # Store queue lengths for plotting
        
        # Set up distribution generators
        if self.distribution == 'weibull':
            self.arrival_gen = weibull_generator(shape, 1/self.arrival_rate)
            self.service_gen = weibull_generator(shape, 1/mu)
            self.schedule(self.arrival_gen(), Arrival(0))
        else:  # exponential
            self.schedule(expovariate(self.arrival_rate), Arrival(0))
            
        self.schedule(0, MonitorQueues(plot_interval))

    def schedule_arrival(self, job_id):
        """Schedule the arrival of a new job."""
        if self.distribution == 'weibull':
            self.schedule(self.arrival_gen(), Arrival(job_id))
        else:  # exponential
            self.schedule(expovariate(self.arrival_rate), Arrival(job_id))

    def schedule_completion(self, job_id, queue_index):
        """Schedule the completion of a job."""
        if self.distribution == 'weibull':
            if self.queue_policy == 'lifo' and job_id in self.remaining_time:
                self.schedule(self.remaining_time[job_id], Completion(job_id, queue_index))
            else:
                self.schedule(self.service_gen(), Completion(job_id, queue_index))
        else:  # exponential
            self.schedule(expovariate(self.mu), Completion(job_id, queue_index))

    def queue_len(self, i):
        """Return the length of the i-th queue."""
        return (self.running[i] is not None) + len(self.queues[i])


class Arrival(Event):
    """Event representing the arrival of a new job."""

    def __init__(self, job_id):
        self.id = job_id

    def process(self, sim: Queues):
        sim.arrivals[self.id] = sim.t  # set the arrival time of the job
        sample_queues = sample(range(sim.n), sim.d)  # sample d queues at random
        queue_index = min(sample_queues, key=sim.queue_len)  # pick the shortest one

        if sim.running[queue_index] is None:
            # Server is idle: run job immediately
            sim.running[queue_index] = self.id
            sim.schedule_completion(self.id, queue_index)
        else:
            # Server is busy: add to queue
            sim.queues[queue_index].append(self.id)

        # Schedule the next arrival
        sim.schedule_arrival(self.id + 1)



class Completion(Event):
    """Job completion."""

    def __init__(self, job_id, queue_index):
        self.job_id = job_id
        self.queue_index = queue_index

    def process(self, sim: Queues):
        queue_index = self.queue_index

        assert sim.running[queue_index] == self.job_id  # sanity check
        sim.completions[self.job_id] = sim.t
        queue = sim.queues[queue_index]

        if queue:  # queue is not empty
            if sim.queue_policy == 'fifo':
                new_job_id = queue.popleft()
            else:  # lifo
                new_job_id = queue.pop()
            sim.running[queue_index] = new_job_id
            sim.schedule_completion(new_job_id, queue_index)
        else:
            sim.running[queue_index] = None  # no job is running



class MonitorQueues(Event):
    """Event for monitoring queue lengths."""
    def __init__(self, interval=1):
        self.interval = interval

    def process(self, sim: Queues):
        # Record queue lengths for plotting
        for i in range(sim.n):
            sim.plots.append(sim.queue_len(i))
        # Schedule next monitoring event
        sim.schedule(self.interval, self)


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--lambd', type=float, default=[0.5,0.9,0.95,0.99], help="arrival rate")
    parser.add_argument('--mu', type=float, default=1, help="service rate")
    parser.add_argument('--max-t', type=float, default=1_000, help="maximum time to run the simulation")
    parser.add_argument('--n', type=int, default=1_000, help="number of servers")
    parser.add_argument('--d', type=int, default=[1,2,5,10], help="number of queues to sample")
    parser.add_argument('--csv', help="CSV file in which to store results")
    parser.add_argument("--seed", help="random seed")
    parser.add_argument("--verbose", action='store_true')
    parser.add_argument("--plot_interval", type=float, default=1, help="how often to collect data points for the plot")
    parser.add_argument("--queue-policy", choices=['fifo', 'lifo'], default='fifo',
                       help="queue policy: FIFO (First In First Out) or LIFO (Last In First Out) with preemption")
    parser.add_argument("--distribution", choices=['exponential', 'weibull'], default='exponential',
                       help="distribution type for arrival and service times")
    parser.add_argument("--shape", type=float, default=1,
                       help="shape parameter for Weibull distribution (only used when distribution=weibull)")
    args = parser.parse_args()

    if args.seed:
        seed(args.seed)
    if args.verbose:
        logging.basicConfig(format='{levelname}:{message}', level=logging.INFO, style='{')

    # Create a 2x2 grid of subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    title = f"{args.distribution.capitalize()} Distribution {args.queue_policy.upper()}"
    if args.distribution == 'weibull':
        title += f" - shape={args.shape}"
    title += f" - n={args.n} max-t={args.max_t}"
    fig.suptitle(title, fontsize=14)
    axes = axes.flatten()

    # Print results header
    print(f"{'λ':>8} | {'d':>4} | {'Avg Time (W)':>15} | {'Theoretical (d=1)':>20}")
    print("-"*70)

    for d_idx, d in enumerate(args.d):
        ax = axes[d_idx]
        for lambd in args.lambd:
            sim = Queues(lambd, args.mu, args.n, d, args.queue_policy, args.distribution,
                        args.plot_interval, args.shape)
            sim.run(args.max_t)
            

            completions = sim.completions
            W = ((sum(completions.values()) - sum(sim.arrivals[job_id] for job_id in completions))
                / len(completions))
            
            theoretical = ""
            if args.mu == 1 and lambd != 1 and (args.distribution == 'exponential' or 
                                               (args.distribution == 'weibull' and args.shape == 1)):
                theoretical = f"{1 / (1 - lambd):.2f}"

            # Print results in a table format (without status)
            print(f"{lambd:8.2f} | {d:4d} | {W:15.2f} | {theoretical:>20}")

            if args.csv is not None:
                with open(args.csv, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([lambd, args.mu, args.max_t, args.n, d, args.shape,
                                   args.queue_policy, args.distribution, W])

            # Plot results
            array = []
            fractions = []
            indexes = [i for i in range(0,15)]
            for i in indexes:
                count = 0
                for x in sim.plots:
                    if x >= i:
                        count += 1
                array.append(count)

            max_value = max(array)
            for i in array:
                fractions.append(float(i) / max_value)
            
            x_ticks = [2,4,6,8,10,12,14]
            y_ticks = [0.0,0.2,0.4,0.6,0.8,1.0]
            style = ['solid','dashed','dashdot','dotted']
            
            ax.plot(indexes[1:], fractions[1:], label=f"λ : {lambd}", linestyle=style[args.lambd.index(lambd)])
            ax.set_title(f"{d} choices - {args.distribution} service times - {args.queue_policy.upper()}")
            ax.set_xlabel("Queue length")
            ax.set_ylabel("Fraction of queues with at least that size")
            ax.set_ylim(min(fractions), 1)
            ax.legend(loc=0, prop={'size': 6})
            ax.grid(True)
            ax.set_xticks(x_ticks)
            ax.set_yticks(y_ticks)

    plt.tight_layout()
    plt.show()
    
    
if __name__ == '__main__':
    main() 
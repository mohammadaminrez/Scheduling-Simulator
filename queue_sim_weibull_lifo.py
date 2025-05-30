#!/usr/bin/env python3

import argparse
import csv
import collections
import logging
import matplotlib.pyplot as plt
from random import sample, seed

from discrete_event_sim import Simulation, Event
from workloads import weibull_generator

# columns saved in the CSV file
CSV_COLUMNS = ['lambd', 'mu', 'max_t', 'n', 'd', 'shape', 'w']


class QueuesLIFO(Simulation):
    """Simulation of a system with n servers and n queues using Weibull distributions with LIFO preemptive policy.

    The system has n servers with one queue each. Jobs arrive at rate lambd and are served at rate mu.
    When a job arrives, according to the supermarket model, it chooses d queues at random and joins
    the shortest one. The LIFO preemptive policy means that when a new job arrives, it preempts the
    currently running job, which is then pushed to the front of the queue.

    The Weibull distribution is used for both inter-arrival times and service times, with a shape parameter
    that controls the distribution's behavior:
    - shape = 1: Same as exponential distribution (memoryless)
    - shape < 1: Heavy-tailed distribution (few very large jobs, many small ones)
    - shape > 1: More uniform/bell-shaped distribution
    """

    def __init__(self, lambd, mu, n, d, plot_interval=1, shape=1):
        super().__init__()
        self.running = [None] * n  # if not None, the id of the running job (per queue)
        self.queues = [collections.deque() for _ in range(n)]  # LIFO queues of the system
        self.arrivals = {}  # dictionary mapping job id to arrival time
        self.completions = {}  # dictionary mapping job id to completion time
        self.remaining_time = {}  # dictionary mapping job id to remaining service time
        self.lambd = lambd
        self.n = n
        self.d = d
        self.mu = mu
        self.shape = shape  # shape parameter for Weibull distribution
        self.arrival_rate = lambd * n  # frequency of new jobs is proportional to the number of queues
        self.plots = []  # Store queue lengths for plotting
        
        # Create Weibull generators for arrivals and service times
        self.arrival_gen = weibull_generator(shape, 1/self.arrival_rate)
        self.service_gen = weibull_generator(shape, 1/mu)
        
        self.schedule(self.arrival_gen(), Arrival(0))  # schedule the first arrival
        self.schedule(0, MonitorQueues(plot_interval))  # schedule monitoring

    def schedule_arrival(self, job_id):
        """Schedule the arrival of a new job using Weibull distribution."""
        self.schedule(self.arrival_gen(), Arrival(job_id))

    def schedule_completion(self, job_id, queue_index):
        """Schedule the completion of a job using Weibull distribution."""
        if job_id not in self.remaining_time:
            self.remaining_time[job_id] = self.service_gen()
        self.schedule(self.remaining_time[job_id], Completion(job_id, queue_index))

    def queue_len(self, i):
        """Return the length of the i-th queue.
        
        Notice that the currently running job is counted even if it is not in self.queues[i]."""
        return (self.running[i] is not None) + len(self.queues[i])


class Arrival(Event):
    """Event representing the arrival of a new job."""

    def __init__(self, job_id):
        self.id = job_id

    def process(self, sim: QueuesLIFO):
        sim.arrivals[self.id] = sim.t  # set the arrival time of the job
        sample_queues = sample(range(sim.n), sim.d)  # sample the id of d queues at random
        queue_index = min(sample_queues, key=sim.queue_len)  # shortest queue among the sampled ones

        if sim.running[queue_index] is not None:
            # Preempt the currently running job
            preempted_job = sim.running[queue_index]
            # Calculate remaining time for preempted job
            if preempted_job in sim.remaining_time:
                sim.remaining_time[preempted_job] -= (sim.t - sim.arrivals[preempted_job])
            sim.queues[queue_index].appendleft(preempted_job)  # Add to front of queue (LIFO)
        
        # Set the new job as running
        sim.running[queue_index] = self.id
        sim.schedule_completion(self.id, queue_index)
        
        # schedule the arrival of the next job
        sim.schedule_arrival(self.id + 1)


class Completion(Event):
    """Job completion."""

    def __init__(self, job_id, queue_index):
        self.job_id = job_id
        self.queue_index = queue_index

    def process(self, sim: QueuesLIFO):
        queue_index = self.queue_index
        if sim.running[queue_index] != self.job_id:
            return  # Job was preempted, ignore this completion event
        
        sim.completions[self.job_id] = sim.t
        queue = sim.queues[queue_index]
        if queue:  # queue is not empty
            sim.running[queue_index] = new_job_id = queue.popleft()  # assign the first job in the queue
            sim.schedule_completion(new_job_id, queue_index)  # schedule its completion
        else:
            sim.running[queue_index] = None  # no job is running on the queue


class MonitorQueues(Event):
    """Event for monitoring queue lengths."""
    def __init__(self, interval=1):
        self.interval = interval

    def process(self, sim: QueuesLIFO):
        # Record queue lengths for plotting
        for i in range(sim.n):
            sim.plots.append(sim.queue_len(i))
        # Schedule next monitoring event
        sim.schedule(self.interval, self)


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--lambd', type=float, default=[0.5,0.7,0.9,0.99], help="arrival rate")
    parser.add_argument('--mu', type=float, default=1, help="service rate")
    parser.add_argument('--max-t', type=float, default=1_000_000, help="maximum time to run the simulation")
    parser.add_argument('--n', type=int, default=1, help="number of servers")
    parser.add_argument('--d', type=int, default=[1,2,5,10], help="number of queues to sample")
    parser.add_argument('--csv', help="CSV file in which to store results")
    parser.add_argument("--seed", help="random seed")
    parser.add_argument("--verbose", action='store_true')
    parser.add_argument("--plot_interval", type=float, default=1, help="how often to collect data points for the plot")
    parser.add_argument("--shape", type=float, default=1, help="shape parameter for Weibull distribution")
    args = parser.parse_args()

    # params = [getattr(args, column) for column in CSV_COLUMNS[:-1]]
    # corresponds to params = [args.lambd, args.mu, args.max_t, args.n, args.d]

    # if any(x <= 0 for x in params):
    #     logging.error("lambd, mu, max-t, n and d must all be positive")
    #     exit(1)

    if args.seed:
        seed(args.seed)
    if args.verbose:
        logging.basicConfig(format='{levelname}:{message}', level=logging.INFO, style='{')

    # Create a 2x2 grid of subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f"Weibull Distribution (shape={args.shape}) - LIFO", fontsize=14)
    axes = axes.flatten()

    for d_idx, d in enumerate(args.d):
        ax = axes[d_idx]
        print(f"d={d}, shape={args.shape}")
        for lambd in args.lambd:
            if lambd >= args.mu:
                logging.warning("The system is unstable: lambda >= mu") 

            sim = QueuesLIFO(lambd, args.mu, args.n, d, args.plot_interval, args.shape)
            sim.run(args.max_t)

            completions = sim.completions
            W = ((sum(completions.values()) - sum(sim.arrivals[job_id] for job_id in completions))
                / len(completions))
            print(f"λ={lambd}: Average time spent in the system: {W}")
            if args.mu == 1 and lambd != 1 and args.shape == 1:
                print(f"Theoretical expectation for random server choice (d=1): {1 / (1 - lambd)}")

            if args.csv is not None:
                with open(args.csv, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([lambd, args.mu, args.max_t, args.n, d, args.shape, W])

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
            ax.set_title(f"{d} choices - Weibull service times (shape={args.shape}) - LIFO")
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
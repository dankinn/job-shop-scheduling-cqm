from time import time
import warnings

from tabulate import tabulate
import argparse
from dimod import ConstrainedQuadraticModel, Binary, Integer, SampleSet
from dwave.system import LeapHybridCQMSampler
import pandas as pd
import numpy as np


import sys
sys.path.append('./src')
from utils.utils import print_cqm_stats
from model_data import JobShopData
from utils.greedy import GreedyJobShop

class JobShopSchedulingAssignmentCQM():
    """Builds and solves a Job Shop Scheduling problem using CQM.
    """
    def __init__(self, model_data: JobShopData, random_samples: dict):
        """Initializes the JobShopSchedulingCQM class.

        Args:
            model_data (JobShopData): A JobShopData object that holds the data 
                for this job shop scheduling problem.
            random_samples: a dictionary of task start times for each task, in
                the form {task: set(start_times)}
        """        
        self.model_data = model_data
        self.cqm = None
        self.x = {}
        self.y = {}
        self.best_sample = {}
        self.solution = {}
        self.completion_time = 0
        self.random_samples = random_samples


    def define_cqm_model(self) -> None:
        """Define CQM model."""
        self.cqm = ConstrainedQuadraticModel()


    def define_variables(self) -> None:
        """Define CQM variables.
        """
        # Define make span as an integer variable
        makespan_lower_bound = int(min([max(v)[1] for v in self.random_samples.values()]) * .8)
        makespan_upper_bounds = max([max(v)[1] for v in self.random_samples.values()])
        self.cqm.add_variable('INTEGER', 'makespan',  lower_bound=makespan_lower_bound, upper_bound=makespan_upper_bounds)

        # Define binary variable indicating whether task sample i is selected for task t
        self.x = {}
        for task, start_times in self.random_samples.items():
            for idx, _ in enumerate(start_times):
                var_name = 'x{}_{}'.format(task, idx)
                self.x[(task, idx)] = var_name
                self.cqm.add_variable(vartype='BINARY', v=var_name)


    def define_objective_function(self) -> None:
        """Define objective function, which is to minimize
        the makespan of the schedule."""
        self.cqm.set_objective([('makespan', 1)])


    def add_precedence_constraints(self, model_data: JobShopData) -> None:
        """Precedence constraints ensures that all operations of a job are
        executed in the given order.

        Args:
            model_data: a JobShopData data class
        """
        prec_count = 0
        for job in model_data.jobs: 
            for prev_task, curr_task in zip(model_data.job_tasks[job][:-1], model_data.job_tasks[job][1:]):

                for idx in range(len(self.random_samples[curr_task])):
                    
                    start = self.random_samples[curr_task][idx][0]
                    invalid_prev_idcs = [i for i, prev_times in enumerate(self.random_samples[prev_task]) if prev_times[1] > start]
                    if len(invalid_prev_idcs) > 0:
                        prec_constraint = []
                        for prev_task_idx in invalid_prev_idcs:
                            prec_constraint.append(('x{}_{}'.format(prev_task, prev_task_idx), 'x{}_{}'.format(curr_task, idx), 1))
                        prec_count += 1
                        self.cqm.add_constraint_from_iterable(prec_constraint, label='prec_ctr{}_{}'.format(curr_task, idx), sense='==', rhs=0)
        print ('Added {} precedence constraints'.format(prec_count))


    def add_quadratic_overlap_constraint(self, model_data: JobShopData) -> None:
        """Add quadratic constraints to ensure that no two jobs can be scheduled
         on the same machine at the same time.

         Args:
             model_data: a JobShopData data class
        """
        overlap_count = 0
        for j in model_data.jobs:
            for k in model_data.jobs:
                if j < k:
                    for i in model_data.resources:
                        task_k = model_data.get_resource_job_tasks(job=k, resource=i)
                        task_j = model_data.get_resource_job_tasks(job=j, resource=i)
                        k_times = self.random_samples[task_k]
                        j_times = self.random_samples[task_j]
                        for k_idx, k_time in enumerate(k_times):
                            overlaps = []
                            for j_idx, j_time in enumerate(j_times):
                                if k_time[1] > j_time[0] and k_time[0] < j_time[1]:
                                    overlaps.append(j_idx)
                                elif j_time[1] > k_time[0] and j_time[0] < k_time[1]:
                                    overlaps.append(j_idx)
                                    
                            if len(overlaps) > 0:
                                constraint = []
                                for j_idx in overlaps:
                                    constraint.append(('x{}_{}'.format(task_j, j_idx), 'x{}_{}'.format(task_k, k_idx), 1))
                                overlap_count += 1
                                self.cqm.add_constraint_from_iterable(constraint, label='overlap_ctr{}_{}_{}'.format(task_j, task_k, k_idx), sense='==', rhs=0)
        print ('Added {} overlap constraints'.format(overlap_count))


    def add_makespan_constraint(self, model_data: JobShopData) -> None:
        """Ensures that the make span is at least the largest completion time of
        the last operation of all jobs.

        Args:
            model_data: a JobShopData data class
        """
        for job in model_data.jobs:
            last_job_task = model_data.job_tasks[job][-1]
            last_job_vars = [('x{}_{}'.format(last_job_task, idx), -finish) for idx, (_, finish) in enumerate(self.random_samples[last_job_task])]
            last_job_vars.append(('makespan', 1))
            self.cqm.add_constraint_from_iterable(last_job_vars, label='makespan_ctr{}'.format(job), sense='>=', rhs=0)

    
    def choose_one_sample(self) -> None:
        """Ensures that exactly one sample is chosen for each task."""
        for task in self.random_samples.keys():
            self.cqm.add_discrete_from_iterable(['x{}_{}'.format(task, idx) for idx in range(len(self.random_samples[task]))], 
                                                label='choose_one_ctr{}'.format(task))


    def call_cqm_solver(self, time_limit: int,  profile: str) -> None:
        """Calls CQM solver to solve the job shop scheduling assignment problem.

        Args:
            time_limit (int): time limit in second
            profile (str): The profile variable to pass to the Sampler. Defaults to None.
            See documentation at 
            https://docs.ocean.dwavesys.com/en/stable/docs_cloud/reference/generated/dwave.cloud.config.load_config.html#dwave.cloud.config.load_config
        """
        sampler = LeapHybridCQMSampler(profile=profile)
        raw_sampleset = sampler.sample_cqm(self.cqm, time_limit=time_limit, label='Job Shop Assignment')
        self.feasible_sampleset = raw_sampleset.filter(lambda d: d.is_feasible)
        num_feasible = len(self.feasible_sampleset)
        if num_feasible > 0:
            best_samples = \
                self.feasible_sampleset.truncate(min(10, num_feasible))
        else:
            self.solution = {}
            self.solution_makespan = -1
            print ('\n\n No feasible solutions found. \n\n')
            warnings.warn("Warning: Did not find feasible solution")
            return

        print(" \n" + "=" * 30 + "BEST SAMPLE SET" + "=" * 30)
        print(best_samples)
        self.best_sample = best_samples.first.sample
        self.solution = {}
        for var, var_name in self.x.items():
            val = self.best_sample[var_name]
            if val == 1:
                self.solution[var[0]] = self.random_samples[var[0]][var[1]]

        self.solution_makespan = max([v[1] for v in self.solution.values()])


    
    def solution_as_dataframe(self) -> pd.DataFrame:
        """This function returns the solution as a pandas DataFrame

        Returns:
            pd.DataFrame: A pandas DataFrame containing the solution
        """        
        df_rows = []
        for (j, i), (task, start, dur) in self.solution.items():
            df_rows.append([j, task, start, start + dur, i])
        df = pd.DataFrame(df_rows, columns=['Job', 'Task', 'Start', 'Finish', 'Resource'])
        return df
    

def generate_random_greedy_samples(job_data: JobShopData, num_samples: int=100, keep_pct=1) -> dict:
    """This function generates random samples using the greedy algorithm; it will keep the
    top keep_pct percent of samples.

    Args:
        job_data (JobShopData): An instance of the JobShopData class
        num_samples (int, optional): The number of samples to take (number of times
            the GreedyJobShop algorithm is run). Defaults to 100.
        keep_pct (int, optional): The % of samples to keep, between 0 and 1. Defaults to 1.

    Returns:
        dict: A dictionary of task times, in the form {task: [start_time, end_time]}. Only
            unique task times will be returned for each task.
    """    
    start = time()
    solutions = []
    task_times = {task: [] for task in job_data.get_tasks()}
    for _ in range(num_samples):
        greedy = GreedyJobShop(job_data)
        task_assignments = greedy.solve()
        [task_times[task].append(task_assignments[task]) for task in job_data.get_tasks()]
        solutions.append(max([v[1] for v in task_assignments.values()]))
    end = time()

    sorted_solutions = [x for x in solutions]
    sorted_solutions.sort()
    kth_greedy = sorted_solutions[int(num_samples * keep_pct)]
    keep_idcs = [i for i, x in enumerate(solutions) if x <= kth_greedy]
    task_times = {task: [task_times[task][i] for i in keep_idcs] for task in job_data.get_tasks()}
    best_greedy = min(solutions)

    print('Generated {} samples in {} seconds'.format(num_samples, end-start))
    best_greedy = min(solutions)
    print ('Best greedy solution: {}'.format(best_greedy))
    task_times = {task: list(set(task_time)) for task, task_time in task_times.items()}
    return task_times, best_greedy
    

def run_shop_scheduler(
    job_data: JobShopData,
    solver_time_limit: int = 60,
    verbose: bool = True,
    profile: str = None,
    num_trials: int=100,
    trial_keep_pct: float=0.01
    ) -> None:
    """This function runs the job shop scheduler on the given data.

    Args:
        job_data (JobShopData): A JobShopData object that holds the data for this job shop 
            scheduling problem.
        solver_time_limit (int, optional): Upperbound on how long the schedule can be; leave empty to 
            auto-calculate an appropriate value. Defaults to None.
        use_mip_solver (bool, optional): Whether to use the MIP solver instead of the CQM solver.
            Defaults to False.
        verbose (bool, optional): Whether to print verbose output. Defaults to True.
        profile (str, optional): The profile variable to pass to the Sampler. Defaults to None.
        num_trials (int, optional): The number of trials to run the greedy algorithm. Defaults to 100.

    """

    greedy_samples, best_greedy = generate_random_greedy_samples(job_data, num_samples=int(num_trials/trial_keep_pct), keep_pct=trial_keep_pct)
    for task in greedy_samples.keys():
        _, upper_bound = job_data.get_task_time_bounds(task, best_greedy)
        greedy_samples[task] = [x for x in greedy_samples[task] if x[0] <= upper_bound]

    model_building_start = time()
    model = JobShopSchedulingAssignmentCQM(model_data=job_data, random_samples=greedy_samples)
    model.define_cqm_model()
    model.define_variables()
    model.add_precedence_constraints(job_data)
    model.add_quadratic_overlap_constraint(job_data)
    model.choose_one_sample()
    model.add_makespan_constraint(job_data)
    model.define_objective_function()

    if verbose:
        print_cqm_stats(model.cqm)
    model_building_time = time() - model_building_start
    solver_start_time = time()

    model.call_cqm_solver(time_limit=solver_time_limit, profile=profile)

    solver_time = time() - solver_start_time
    if verbose:
        print(" \n" + "=" * 55 + "SOLUTION RESULTS" + "=" * 55)
        print(tabulate([["Best CQM Makespan", "Best Greedy Makespan",
                        "Model Building Time (s)", "Solver Call Time (s)",
                        "Total Runtime (s)"],
                        [model.solution_makespan, 
                         best_greedy,
                         int(model_building_time), 
                         int(solver_time),
                         int(solver_time +  model_building_time)
                         ]],
                    headers="firstrow"))
    


if __name__ == "__main__":
    """Modeling and solving Job Shop Scheduling using CQM solver with the greedy
    assignment method"""

    # Instantiate the parser
    parser = argparse.ArgumentParser(
        description='Job Shop Scheduling Assignment Model Using LeapHybridCQMSampler',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-instance', type=str,
                        help='path to the input instance file; ',
                        default='input/instance5_5.txt')

    parser.add_argument('-tl', type=int,
                        help='time limit in seconds')
    
    parser.add_argument('-no_verbose', action='store_true', default=False,
                        help='Whether to exclude verbose output')
    
    parser.add_argument('-profile', type=str,
                        help='The profile variable to pass to the Sampler. Defaults to None.',
                        default=None)
    
    parser.add_argument('-num_trials', type=int,
                        help='number of trials to run',
                        default=100)
    
    parser.add_argument('-keep_pct', type=float,
                        help='The % of samples to keep, between 0 and 1. Defaults to 1.',
                        default=0.01)
    
    args = parser.parse_args()
    input_file = args.instance
    time_limit = args.tl

    job_data = JobShopData()
    job_data.load_from_file(input_file)

    run_shop_scheduler(job_data,
                       time_limit,
                       verbose=not args.no_verbose,
                       profile=args.profile,
                       num_trials=args.num_trials,
                       trial_keep_pct=args.keep_pct)
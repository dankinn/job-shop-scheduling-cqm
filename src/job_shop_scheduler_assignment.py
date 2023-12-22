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
from utils.utils import print_cqm_stats, write_solution_to_file
import utils.plot_schedule as job_plotter
import utils.mip_solver as mip_solver
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
        self.makespan = {}
        self.best_sample = {}
        self.solution = {}
        self.completion_time = 0
        self.random_samples = random_samples


    def define_cqm_model(self) -> None:
        """Define CQM model."""
        self.cqm = ConstrainedQuadraticModel()


    def define_variables(self, model_data: JobShopData) -> None:
        """Define CQM variables.

        Args:
            model_data: a JobShopData data class
        """
        # Define make span as an integer variable
        # makespan_lower_bound = int(min([max(v)[1] for v in self.random_samples.values()]) * .9)
        makespan_lower_bound = 0
        makespan_upper_bounds = max([max(v)[1] for v in self.random_samples.values()])
        self.makespan = Integer("makespan", lower_bound=makespan_lower_bound, upper_bound=makespan_upper_bounds)
        self.cqm.add_variable('INTEGER', 'makespan',  lower_bound=makespan_lower_bound, upper_bound=makespan_upper_bounds)

        # Define binary variable indicating whether task sample i is selected for task t
        self.x = {}
        for task, start_times in self.random_samples.items():
            for idx, start_time in enumerate(start_times):
                var_name = 'x{}_{}'.format(task, idx)
                self.x[(task, idx)] = var_name
                self.cqm.add_variable(vartype='BINARY', v=var_name)
                # self.cqm.add_variable('BINARY', self.x[(task, idx)])

    def define_objective_function(self) -> None:
        """Define objective function, which is to minimize
        the makespan of the schedule."""
        self.cqm.set_objective([('makespan', 1)])
        # last_job_vars = []
        # for job in self.model_data.jobs:
        #     last_job_task = self.model_data.job_tasks[job][-1]
        #     # last_job_ends = [self.random_samples[last_job_task][idx][1] for idx in range(len(self.random_samples[last_job_task]))]
        #     last_job_vars.extend([('x{}_{}'.format(last_job_task, idx), finish) for idx, (_, finish) in enumerate(self.random_samples[last_job_task])])
        # self.cqm.set_objective(last_job_vars)


    def add_precedence_constraints(self, model_data: JobShopData) -> None:
        """Precedence constraints ensures that all operations of a job are
        executed in the given order.

        Args:
            model_data: a JobShopData data class
        """
        for job in model_data.jobs:  # job
            for prev_task, curr_task in zip(model_data.job_tasks[job][:-1], model_data.job_tasks[job][1:]):

                for idx in range(len(self.random_samples[curr_task])):
                    
                    start = self.random_samples[curr_task][idx][0]
                    invalid_prev_idcs = [i for i, prev_times in enumerate(self.random_samples[prev_task]) if prev_times[1] > start]
                    if len(invalid_prev_idcs) > 0:
                        prec_constraint = [('x{}_{}'.format(prev_task, prev_task_idx), 'x{}_{}'.format(curr_task, idx), 1) \
                                           for prev_task_idx in invalid_prev_idcs]
                        self.cqm.add_constraint_from_iterable(prec_constraint, label='prec_ctr{}_{}'.format(curr_task, idx), sense='==', rhs=0)


    def add_quadratic_overlap_constraint(self, model_data: JobShopData) -> None:
        """Add quadratic constraints to ensure that no two jobs can be scheduled
         on the same machine at the same time.

         Args:
             model_data: a JobShopData data class
        """
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
                                constraint = [('x{}_{}'.format(task_j, j_idx), 'x{}_{}'.format(task_k, k_idx), 1) for j_idx in overlaps]
                                self.cqm.add_constraint_from_iterable(constraint, label='overlap_ctr{}_{}_{}'.format(task_j, task_k, k_idx), sense='==', rhs=0)


    def add_makespan_constraint(self, model_data: JobShopData) -> None:
        """Ensures that the make span is at least the largest completion time of
        the last operation of all jobs.

        Args:
            model_data: a JobShopData data class
        """
        for job in model_data.jobs:
            last_job_task = model_data.job_tasks[job][-1]
            # last_job_ends = [self.random_samples[last_job_task][idx][1] for idx in range(len(self.random_samples[last_job_task]))]
            last_job_vars = [('x{}_{}'.format(last_job_task, idx), -finish) for idx, (_, finish) in enumerate(self.random_samples[last_job_task])]
            last_job_vars.append(('makespan', 1))
            self.cqm.add_constraint_from_iterable(last_job_vars, label='makespan_ctr{}'.format(job), sense='>=', rhs=0)
            # for idx, task_time in enumerate((self.random_samples[last_job_task])):
            
            #     self.cqm.add_constraint(
            #     [('makespan', 1), ('x{}_{}'.format(last_job_task, idx),-task_time[1])],
            #     rhs=0, sense='>=',
            #     label='makespan_ctr{}_{}'.format(job, idx))

    
    def choose_one_sample(self) -> None:
        """Ensures that exactly one sample is chosen for each task."""
        for task in self.random_samples.keys():
            self.cqm.add_constraint_from_iterable([('x{}_{}'.format(task, idx), 1) for idx in range(len(self.random_samples[task]))],
                                                  rhs=1, sense='==',
                                                  label='choose_one_ctr{}'.format(task))


    def call_cqm_solver(self, time_limit: int, model_data: JobShopData, profile: str) -> None:
        """Calls CQM solver.

        Args:
            time_limit (int): time limit in second
            model_data (JobShopData): a JobShopData data class
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
            warnings.warn("Warning: Did not find feasible solution")
            best_samples = raw_sampleset.truncate(10)

        print(" \n" + "=" * 30 + "BEST SAMPLE SET" + "=" * 30)
        print(best_samples)

        self.best_sample = best_samples.first.sample

        self.solution = {}
        for var, var_name in self.x.items():
            val = self.best_sample[var_name]
            if val == 1:
                self.solution[var[0]] = self.random_samples[var[0]][var[1]]

        self.completion_time = max([v[1] for v in self.solution.values()])
        print ('Completion time: {}'.format(self.completion_time))
        return self.completion_time


    
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
    

def generate_random_greedy_samples(job_data: JobShopData, num_samples: int=100) -> dict:
    start = time()
    solutions = []
    task_times = {task: set() for task in job_data.get_tasks()}
    for x in range(num_samples):
        greedy = GreedyJobShop(job_data)
        task_assignments = greedy.solve()
        [task_times[task].add(task_assignments[task]) for task in job_data.get_tasks()]
        solutions.append(max([v[1] for k,v in task_assignments.items()]))
    
    end = time()
    print('Generated {} samples in {} seconds'.format(num_samples, end-start))
    best_greedy = min(solutions)
    print ('Best greedy solution: {}'.format(best_greedy))
    import pdb
    pdb.set_trace()
    task_times = {task: list(task_time) for task, task_time in task_times.items()}
    return task_times, best_greedy
    

def run_shop_scheduler(
    job_data: JobShopData,
    solver_time_limit: int = 60,
    verbose: bool = False,
    out_sol_file: str = None,
    out_plot_file: str = None,
    profile: str = None,
    num_trials: int=100
    ) -> pd.DataFrame:
    """This function runs the job shop scheduler on the given data.

    Args:
        job_data (JobShopData): A JobShopData object that holds the data for this job shop 
            scheduling problem.
        solver_time_limit (int, optional): Upperbound on how long the schedule can be; leave empty to 
            auto-calculate an appropriate value. Defaults to None.
        use_mip_solver (bool, optional): Whether to use the MIP solver instead of the CQM solver.
            Defaults to False.
        verbose (bool, optional): Whether to print verbose output. Defaults to False.
        out_sol_file (str, optional): Path to the output solution file. Defaults to None.
        out_plot_file (str, optional): Path to the output plot file. Defaults to None.
        profile (str, optional): The profile variable to pass to the Sampler. Defaults to None.

    Returns:
        pd.DataFrame: A DataFrame that has the following columns: Task, Start, Finish, and
        Resource.
    """    
    greedy_samples, best_greedy = generate_random_greedy_samples(job_data, num_samples=num_trials)

    model_building_start = time()
    model = JobShopSchedulingAssignmentCQM(model_data=job_data, random_samples=greedy_samples)
    model.define_cqm_model()
    model.define_variables(job_data)
    model.add_precedence_constraints(job_data)
    model.add_quadratic_overlap_constraint(job_data)
    model.choose_one_sample()
    model.add_makespan_constraint(job_data)
    model.define_objective_function()

    if verbose:
        print_cqm_stats(model.cqm)
    model_building_time = time() - model_building_start
    solver_start_time = time()

    completion_time = model.call_cqm_solver(time_limit=solver_time_limit, model_data=job_data, profile=profile)
    return completion_time, best_greedy
    sol = model.best_sample
    solver_time = time() - solver_start_time

    if verbose:
        print(" \n" + "=" * 55 + "SOLUTION RESULTS" + "=" * 55)
        print(tabulate([["Completion Time", "Max Possible Make-Span",
                        "Model Building Time (s)", "Solver Call Time (s)",
                        "Total Runtime (s)"],
                        [model.completion_time, 
                         model.max_makespan,
                         int(model_building_time), 
                         int(solver_time),
                         int(solver_time +  model_building_time)
                         ]],
                    headers="firstrow"))
    
    # Write solution to a file.
    if out_sol_file is not None:
        write_solution_to_file(
            job_data, model.solution, model.completion_time, out_sol_file)

    # Plot solution
    if out_plot_file is not None:
        job_plotter.plot_solution(job_data, model.solution, out_plot_file)

    df = model.solution_as_dataframe()
    return df


if __name__ == "__main__":
    """Modeling and solving Job Shop Scheduling using CQM solver."""

    # Instantiate the parser
    parser = argparse.ArgumentParser(
        description='Job Shop Scheduling Using LeapHybridCQMSampler',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-instance', type=str,
                        help='path to the input instance file; ',
                        default='input/instance5_5.txt')

    parser.add_argument('-tl', type=int,
                        help='time limit in seconds')

    parser.add_argument('-os', type=str,
                        help='path to the output solution file',
                        default='output/solution.txt')

    parser.add_argument('-op', type=str,
                        help='path to the output plot file',
                        default='output/schedule.png')
    
    parser.add_argument('-verbose', action='store_true', default=True,
                        help='Whether to print verbose output')
    
    parser.add_argument('-profile', type=str,
                        help='The profile variable to pass to the Sampler. Defaults to None.',
                        default=None)
    
    
    # Parse input arguments.
    args = parser.parse_args()
    input_file = args.instance
    time_limit = args.tl
    out_plot_file = args.op
    out_sol_file = args.os

    job_data = JobShopData()
    job_data.load_from_file(input_file)

    completion_times = []
    greedy_times = []
    for i in range(100):
        completion_time, greedy_time = run_shop_scheduler(job_data, time_limit, verbose=args.verbose, profile=args.profile, num_trials=2000)
        completion_times.append(completion_time)
        greedy_times.append(greedy_time)
        print ('done with iteration {}'.format(i))
        print ('completion time: {}'.format(completion_time))
        print ('greedy time: {}'.format(greedy_time))
    import pdb
    pdb.set_trace()
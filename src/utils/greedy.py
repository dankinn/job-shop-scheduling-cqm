'''
This file will greedily generate a solution for the job shop problem
'''
import argparse
import numpy as np
import time

import pandas as pd
import sys
sys.path.append('./src')
from model_data import JobShopData

class GreedyJobShop:

    def __init__(self, model_data: JobShopData):
        """Initializes the GreedyJobShop class.

        Args:
            model_data (JobShopData): A JobShopData object that holds 
                the data for this job shop
        """        
        self.model_data = model_data
    

    def solve(self) -> dict:
        '''
        This solves the job shop scheduling problem using the 
        following strategy:
        1. Randomly select a job with open tasks
        2. Select the first open task for that job
        3. Assing the task to its required resource at the earliest time
        4. Repeat until all tasks are assigned
        '''
        def resource_gap_finder(resource_schedule: list, min_start: int, gap_duration: int):
            """helper function that checks to see if there is a gap in the 
            resource_schedule of at least gap_duration that starts after min_start

            Args:
                resource_schedule (list): _description_
                min_start (int): _description_
                gap_duration (int): _description_

            Returns:
                _type_: _description_
            """            
            for i in range(len(resource_schedule) - 1):
                if resource_schedule[i+1]['start'] > min_start and \
                    resource_schedule[i+1]['start'] - max(min_start, resource_schedule[i]['finish']) >= gap_duration:
                    return max(min_start, resource_schedule[i]['finish']), i+1
            return max(min_start, resource_schedule[-1]['finish']), len(resource_schedule)
  
        self.resource_schedules = {resource: [] for resource in self.model_data.resources}
        self.job_schedules = {job: [] for job in self.model_data.jobs}
        self.last_task_scheduled = {job: -1 for job in self.model_data.jobs}
        self.task_assignments = {}
        unfinished_jobs = [x for x in self.model_data.jobs]
        
        unfinished_jobs = np.array([x for x in self.model_data.jobs])
        np.random.shuffle(unfinished_jobs)
        not_yet_finished = np.ones(len(unfinished_jobs))
        idx = 0
        while sum(not_yet_finished) > 0:
            #skip with prob 0.1
            if np.random.rand() < 0.1:
                idx += 1
                continue
            job = unfinished_jobs[idx % len(unfinished_jobs)]
            if not_yet_finished[idx % len(unfinished_jobs)] == 0:
                idx += 1
                continue
            task = self.model_data.job_tasks[job][self.last_task_scheduled[job] + 1]
            resource = task.resource

            if len(self.job_schedules[job]) == 0:
                min_job_time = 0
            else:
                min_job_time = self.job_schedules[job][-1]['finish']
            if len(self.resource_schedules[resource]) == 0:
                min_resource_time = max(0, min_job_time)
                resource_pos = 0
            else: 
                min_resource_time, resource_pos = resource_gap_finder(self.resource_schedules[resource], min_job_time, task.duration)

            start_time = min_resource_time
            finish_time = task.duration + start_time
            self.resource_schedules[resource].insert(resource_pos, {'start': start_time, 'finish': finish_time, 'task': task})
            self.job_schedules[job].append({'start': start_time, 'finish': finish_time, 'task': task})
            self.task_assignments[task] = (start_time, finish_time)
            self.last_task_scheduled[job] += 1
            if self.last_task_scheduled[job] == len(self.model_data.job_tasks[job]) - 1:
                # unfinished_jobs.remove(job)
                not_yet_finished[idx % len(unfinished_jobs)] = 0
            idx += 1
            # if idx % len(unfinished_jobs) == 0:
            #     new_order = [x for x in range(len(unfinished_jobs))]
            #     np.random.shuffle(new_order)
            #     unfinished_jobs = unfinished_jobs[new_order]
            #     not_yet_finished = not_yet_finished[new_order]
        return self.task_assignments


    def solution_as_dataframe(self, solution) -> pd.DataFrame:
        """This function returns the solution as a pandas DataFrame

        Returns:
            pd.DataFrame: A pandas DataFrame containing the solution
        """        
        df_rows = []
        for (j, i), (task, start, dur) in solution.items():
            df_rows.append([j, task, start, start + dur, i])
        df = pd.DataFrame(df_rows, columns=['Job', 'Task', 'Start', 'Finish', 'Resource'])
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
    
    parser.add_argument('-use_mip_solver', action='store_true',
                        help='Whether to use the MIP solver instead of the CQM solver')
    
    parser.add_argument('-verbose', action='store_true', default=True,
                        help='Whether to print verbose output')
    
    parser.add_argument('-allow_quad', action='store_true',
                        help='Whether to allow quadratic constraints')
    
    parser.add_argument('-profile', type=str,
                        help='The profile variable to pass to the Sampler. Defaults to None.',
                        default=None)
    
    parser.add_argument('-max_makespan', type=int,
                        help='Upperbound on how long the schedule can be; leave empty to auto-calculate an appropriate value.',
                        default=None)
    
    # Parse input arguments.
    args = parser.parse_args()
    input_file = args.instance
    time_limit = args.tl
    out_plot_file = args.op
    out_sol_file = args.os
    allow_quadratic_constraints = args.allow_quad

    job_data = JobShopData()
    job_data.load_from_file(input_file)

    start = time.time()
    solutions = []
    task_start_times = {task: set() for task in job_data.get_tasks()}
    for x in range(1000):
        greedy = GreedyJobShop(job_data)
        task_assignments = greedy.solve()
        [task_start_times[task].add(task_assignments[task][0]) for task in job_data.get_tasks()]
        solutions.append(max([v[1] for k,v in task_assignments.items()]))
    end = time.time()
    print(end - start)
    import pdb
    pdb.set_trace()
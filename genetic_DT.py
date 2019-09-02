import random
import numpy as np
from joblib import dump, load
dt=load('DT_version1.joblib')

class genetic_optimisation():
    generations = 10
    pop_size = 5000
    num_params = 4
    num_application = 1
    chance = 0.5
    overhead = 0.5 # in GB
    storage_fraction = 0.5 # percentage of executor's memory to be reserved for storage
    executor_instance_space=np.arange(2,9)
    exe_storage_memory = np.linspace(91454405632, 214544056320, 10000) # in byte
    driver_storage_memory = np.linspace(5345440563, 3745440563, 10000) # in byte
    # User Input
    # num_jobs = random.choice(np.arange(1, 20))
    num_jobs=11
    num_completed_tasks = np.linspace(10, 246240, 5000) # directly related to the no. of jobs and partition of the data
    # user Input
    # app_name = random.choice(['risk', 'measure', 'claim', 'deduplicator'])
    app_name = 'risk'
    app_name_map = {'risk': 0, 'measure': 1, 'claim': 2, 'deduplicator': 3} # label encoding for categorical data
    median_map = {'risk': 9.634, 'measure': 45.78, 'claim': 0.23, 'deduplicator': 33.25} # median time mapping(in min)
    std_map = {'risk': 93.362178, 'measure': 120.920939, 'claim': 0.032087, 'deduplicator': 25.079993} # standard deviation maping(in min)
    median_time_job_ratio = median_map[app_name] / num_jobs

    def __init__(self):
        """It will initialize the population and starts optimising the params on given resources"""
        self.populations = []
        for i in range(self.pop_size):
            self.populations.append(self.initialisation())
        for generation in range(self.generations):
            self.scored_populations = self.fitness(self.populations)
            self.populations = []
            for i in range(int(self.pop_size / 2)):
                item1 = self.selection_2_cross(self.scored_populations)
                random.shuffle(self.scored_populations)
                item2 = self.selection_2_cross(self.scored_populations)
                item1, item2 = self.crossover(item1, item2)
                item1, item2 = self.mutation([item1, item2])
                self.populations.extend([item1, item2])
            print("In generation {0} average time taken is {1}".format(generation,
                                                                                self.finding(self.scored_populations)))


    def initialisation(self):
        """This initializes the population randomly with possible values"""

        self.col = {"num_exe": self.executor_instance_space,
                    "exe_mem": self.exe_storage_memory,
                    "driver_mem": self.driver_storage_memory,
                    "partitions": self.num_completed_tasks}
        lis1 = []
        for i in ["num_exe", "exe_mem", 'driver_mem', 'partitions']:
            lis1.append(random.choice(self.col[i]))
        return np.array(lis1)


    def mem_estimator(self):
        """it will return the amount of data to be cached in memory"""
        # using standard datatypes involved times frequency or using extrapolationn Plus some buffer
        return 600

    def predict_execution_time(self, part_input):
        """job execution time modeling which give the expected execution time(in min)"""
        inputParams=[part_input[0], part_input[1], part_input[2], self.num_jobs, part_input[3], self.app_name_map[self.app_name],
                     self.median_map[self.app_name], self.std_map[self.app_name], self.median_time_job_ratio]
        return dt.predict([inputParams])[0]


    def fitness(self, populations):
        """this fitness function will calculate the deviation from the desired outcome based on constrains"""
        # populations is a list of dataframes in the form [dataframe,...]
        lis = []
        for schedule in populations:
            expected_time = self.predict_execution_time(schedule)
            lis.append([schedule, expected_time])
            # lis.append([schedule, expected_time])
        return lis

    def selection_2_cross(self, items):
        """Select the best application's params to crossover with another best application's params to create new generation"""
        distances = [i[1] for i in items]
        minn = min(distances)
        temp = sorted(distances)
        median = temp[int(len(distances) / 2)]
        n = random.uniform(minn, median)
        for schedule, weight in items:
            if weight <= n:
                return schedule

    def crossover(self, item1, item2):
        """Single point crossover between selected applications params"""
        one_point = int(random.uniform(1, 4))
        return (
        np.concatenate((item1[:one_point], item2[one_point:])), np.concatenate((item2[:one_point], item1[one_point:])))

    def mutation(self, items):
        """Single poing mutation in the application params"""
        for item in items:
            if random.random() >= self.chance:
                point_2_change = int(self.num_params * random.random())
                mapping = {0: "num_exe", 1: "exe_mem", 2: "driver_mem", 3: "partitions"}
                item[point_2_change]=random.choice(self.col[mapping[point_2_change]])
        return items

    def finding(self, scored_populations):
        """finding the average fitness score of the population in a given generation"""
        distances = [i[1] for i in scored_populations]
        # print(min(distances))
        # min_value = min(distances)
        # print('For the current generation, parameter values with minimum time taken:',scored_populations[distances.index(min(distances))])
        # indexes = [i for i, x in enumerate(distances) if x == min_value]
        return sum(distances) / len(distances)

    def opt_executor_memory(self):
        """shows information for the optimised parameters"""
        distances = [i[1] for i in self.scored_populations]
        params = self.scored_populations[distances.index(min(distances))][0]
        time_taken = self.scored_populations[distances.index(min(distances))][1]
        print('For Application:', self.app_name)
        print('Num of jobs:',self.num_jobs)
        print('No. of executor instance needed:', params[0])
        print('executor memory to be demanded(GB):',((params[1]/params[0])/self.storage_fraction)/1073741824 + self.overhead)
        print("Driver's memory(GB):", (params[2]/self.storage_fraction)/1073741824 + self.overhead)
        print("number of partitions required(assuming no subsequent Job's tasks are skipped):",params[3]/self.num_jobs)
        print('Time Taken(Based on model prediction with RMSE of 58 min):',time_taken)

if __name__=="__main__":
    raman = genetic_optimisation()
    raman.opt_executor_memory()

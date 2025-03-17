import json
import os
from typing import List, Callable

class Task():
    def __init__(self, func, params, refresh=False):
        self.func: Callable[[any], any] = func
        self.func_name: str = func.__name__
        self.params: dict = params
        self.refresh: bool = refresh

    def run(self, index, pipeline_params, step_before_results, step_before_executed, directory):
        task_name = f"{index}_{self.func_name}"
        file_name = f"{directory}/{task_name}.json"

        if not step_before_executed and not self.refresh and os.path.exists(file_name) and task_name in pipeline_params and pipeline_params[task_name] == self.params:
            with open(file_name, "r") as f:
                return False, json.load(f)

        results = self.func(step_before_results, **self.params)
        with open(file_name, "w") as f:
            json.dump(results, f)
        
        return True, results

class Pipeline():
    MAIN_FOLDER = ".pipelines"

    def __init__(self, pipeline_name, tasks=[], initial_data=None):
        self.pipeline_folder = f"{Pipeline.MAIN_FOLDER}/{pipeline_name}"
        self.pipeline_execution_file = f"{self.pipeline_folder}/execution.json"
        self.tasks: List[Task] = tasks
        self.initial_data = initial_data
        
        if not os.path.exists(Pipeline.MAIN_FOLDER):
            os.mkdir(Pipeline.MAIN_FOLDER)
        
        if not os.path.exists(self.pipeline_folder):
            os.mkdir(self.pipeline_folder)
    
    def run(self, clean_cache=False):
        if os.path.exists(self.pipeline_execution_file):
            with open(self.pipeline_execution_file, "r") as f:
                pipeline_params = json.load(f)
        else:
            pipeline_params = {}
        
        if clean_cache:
            self.clean_cache()
        
        results = self.initial_data
        executed = False
        for i, task in enumerate(self.tasks):
            print(f"[{i + 1}/{len(self.tasks)}] - {task.func_name}")

            executed, results = task.run(i, pipeline_params, results, executed, self.pipeline_folder)

            if executed:
                pipeline_params[f"{i}_{task.func_name}"] = task.params
                
                with open(self.pipeline_execution_file, "w") as f:
                    json.dump(pipeline_params, f, indent=2)

                print("=> Executed")
            else:
                print("=> Skiped")
    
    def get_data_from_step(self, id):
        for file in os.listdir(self.pipeline_folder):
            if ".json" in file and file.startswith(f"{id}_"):
                with open(f"{self.pipeline_folder}/{file}", "r") as f:
                    return json.load(f)
    
    def clean_cache(self):
        for file in [f"{self.pipeline_folder}/{file}" for file in os.listdir(self.pipeline_folder) if ".json" in file and file != "execution.json"]:
            os.remove(file)
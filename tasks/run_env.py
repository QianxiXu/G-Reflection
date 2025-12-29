"""
搞一个环境执行的代码
"""
import os



os.environ["OPENAI_API_BASE"] = "https://antchat.alipay.com/v1"
os.environ["OPENAI_API_KEY"] = "TDM6IgMVUcG9sfHeweMMgrUD4ptayo8J"
MODEL = "Qwen3-235B-A22B-Instruct-2507"



import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import shutil
import yaml
from dataclasses import dataclass, field
import argparse
import random
from tqdm import tqdm
import mas
# from mas.agents import Agent
# from mas.module_map import module_map
# from mas.reasoning import ReasoningBase
# from mas.memory import MASMemoryBase
# from mas.llm import LLMCallable, GPTChat, get_price
# from mas.mas import MetaMAS
# from mas.utils import EmbeddingFunc

from envs import BaseEnv, BaseRecorder, get_env, get_recorder, get_task
from mas_workflow import get_mas
from prompts import get_dataset_system_prompt, get_task_few_shots
from utils import get_model_type





with open('tasks/configs.yaml') as reader:
    CONFIG: dict = yaml.safe_load(reader)

WORKING_DIR: str = None

@dataclass
class TaskManager:
    task_name: str              # task name
    mas_type: str               # type of mas
    memory_type: str            # memory type
    tasks: list[dict]           # all tasks
    env: BaseEnv                # interative datatset environment
    recorder: BaseRecorder      # record experiment results
    # mas: MetaMAS                # multi-agent system
    mas_config: dict = field(default_factory=dict)   # mas configs
    mem_config: dict = field(default_factory=dict)   # memory configs


def build_task(task: str, mas_type: str, memory_type: str, max_steps: int) -> TaskManager:

    with open("/Users/liuyi/Documents/Docker_env/Agent/GMemory-main/tasks/env_configs/alfworld_config.yaml") as reader:
        config = yaml.safe_load(reader)

    env: BaseEnv = get_env(task, config, max_steps)
    # recorder: BaseRecorder = get_recorder(task, working_dir=WORKING_DIR, namespace='total_task')
    recorder = None
    tasks: list[dict] = get_task(task)
    # mas_workflow: MetaMAS = get_mas(mas_type)
    mas_workflow = None
    mas_config: dict = CONFIG.get(mas_type, {})

    return TaskManager(
        task_name=task,
        mas_type=mas_type,
        memory_type=memory_type,
        tasks=tasks,
        env=env,
        recorder=recorder,
        # mas=mas_workflow,
        mas_config=mas_config
    )  

def run_task(task_manager: TaskManager) -> None:

    
    pre_action_dict = [
        '> think: First, I need to find a tomato. A tomato is more likely to appear in fridge (1), countertop (1-3), cabinet (1-6), drawer (1-3), garbagecan (1), or shelf (1-3). I can check one by one, starting with fridge 1.',
        "think: First, I need to find a tomato. A tomato is more likely to appear in fridge (1), countertop (1-3), cabinet (1-6), drawer (1-3), garbagecan (1), or shelf (1-3). I can check one by one, starting with fridge 1.",
        "> go to fridge 1  ",
        "> open fridge 1",
        "> go to countertop 1",
        "> think: Now I find a tomato (1). Next, I need to take it.",
        "> take tomato 1 from countertop 1",
        "> think: Now I take a tomato (1). Next, I need to go to a fridge (1) and cool it.",
        "> go to fridge 1",
        "> cool tomato 1 with fridge 1",
        "> think: Now I cool a tomato (1). Next, I need to put it in/on microwave 1.",
        "> go to microwave 1",
        "> open microwave 1",
        "> put tomato 1 in/on microwave 1"
    ]

    pre_action_dict = [
            "go to garbagecan 1", 
            "take tomato 2 from garbagecan 1", 
            "go to fridge 1", 
            "go to fridge 2", 
            "cool tomato 2 with fridge 1", 
            "go to microwave 1", 
            "open microwave 1", 
            "move tomato 2 to microwave 1"
        ]

    # pre_action_dict = [
    #     '> think: To solve the task, I need to find and take a tomato, then cool it with fridge, then put it in/on microwave.',
    #     "> think: First I need to find a tomato. A tomato is more likely to appear in fridge (1), countertop (1-3), cabinet (1-6), drawer (1-3), garbagecan (1), shelf (1-3), or sinkbasin (1). I can check one by one, starting with fridge 1.",
    #     "> go to fridge 1",
    #     "> go to countertop 1",
    #     "> think: Now I find a tomato (1). Next, I need to take it.",
    #     "> take tomato 1 from countertop 1",
    #     "> think: Now I take a tomato (1). Next, I need to go to a fridge (1) and cool it.",
    #     "> go to fridge 1",
    #     "> cool tomato 1 with fridge 1",
    #     "> think: Now I cool a tomato (1). Next, I need to put it in/on microwave 1.",
    #     "> go to microwave 1",
    #     "> open microwave 1",
    #     "> put tomato 1 in/on microwave 1"
    # ]

    env = task_manager.env
    env.reset()
    for task_id, task_config in tqdm(enumerate(task_manager.tasks), total=len(task_manager.tasks), desc="Running Tasks"):
        
        if task_id != 2:
            continue
        task_main, task_description = env.set_env(task_config) # 任务目标 看到一个碗 环境描述，你在一个环境里面，周围有一些东西
        
        for i in range(len(pre_action_dict)):
            take_action = pre_action_dict[i]

            action = env.process_action(take_action)

            observation, reward, done = env.step(action)

        

            step_message: str = f'Act {i + 1}: {action}\nObs {i + 1}: {observation}'
            print(step_message)
            print()
            if done:
                print("Task finished!")
                break
        
           
       



if __name__ == '__main__':
    # settings
    random.seed(42)

    parser = argparse.ArgumentParser(description='Run tasks with specified modules.')
    parser.add_argument('--task', type=str, choices=['alfworld', 'fever', 'pddl'])
    parser.add_argument('--mas_type', type=str, choices=['autogen', 'macnet', 'dylan'])
    parser.add_argument('--mas_memory', type=str, default='none', help='Specify mas memory module')
    parser.add_argument('--reasoning', type=str, default='io', help='Specify reasoning module')
    parser.add_argument('--model', type=str, default='gpt-3.5-turbo-0125', help='Specify the LLM model type')
    parser.add_argument('--max_trials', type=int, default=50, help='max number of steps')
    parser.add_argument('--successful_topk', type=int, default=1, help='Number of successful trajs to be retrieved from memory.')
    parser.add_argument('--failed_topk', type=int, default=0, help='Number of failed trajs to be retrieved from memory.')
    parser.add_argument('--insights_topk', type=int, default=3, help='Number of insights to be retrieved from memory.')
    parser.add_argument('--threshold', type=float, default=0.0, help='threshold for traj similarity.')
    parser.add_argument('--use_projector', action='store_true', help='whether to use role projector.')
    parser.add_argument('--hop', type=int, default=1, help='hop for traj similarity.')

  

    args = parser.parse_args()
 

    task: str = args.task
    mas_type: str = args.mas_type
    max_trials: int = args.max_trials
    model_type: str = args.model
    mas_memory_type: str = args.mas_memory
    reasoning_type: str = args.reasoning
    
 
    
    # run tasks
    task_configs: TaskManager = build_task(task, mas_type, mas_memory_type, max_trials)

    run_task(task_configs)
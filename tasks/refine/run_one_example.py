"""
在一个单个样例中反复执行 查看效果
"""

import os
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'


# proxy = "http://127.0.0.1:7890" 

# os.environ["http_proxy"] = proxy
# os.environ["https_proxy"] = proxy



# os.environ["OPENAI_API_BASE"] = "https://api.openai-hk.com/v1"
# # os.environ["OPENAI_API_KEY"] = "hk-ure66g1000059923aeac882630a75dd4b11a2b5eec0e2c17"
# os.environ["OPENAI_API_KEY"] = "hk-10t3l5100005993509f283dc8f4a81a4a55e3d3e191ee8d4"  # self 
# MODEL = "gpt-4o-mini"
# # MODEL = "gpt-5-mini"

os.environ["OPENAI_API_BASE"] = "https://antchat.alipay.com/v1"
os.environ["OPENAI_API_KEY"] = "TDM6IgMVUcG9sfHeweMMgrUD4ptayo8J"
# MODEL = "Qwen3-235B-A22B-Instruct-2507"
# MODEL = "Qwen2.5-7B-Instruct"
MODEL = "Qwen3-8B"




print('------------ Cur Model is ------------', MODEL)
# print('------------ Cur Prefix is ------------', Prefix)

import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import shutil
import yaml
from dataclasses import dataclass, field
import argparse
import random
from tqdm import tqdm

import mas
from mas.agents import Agent
from mas.module_map import module_map
from mas.reasoning import ReasoningBase
from mas.memory import MASMemoryBase
from mas.llm import LLMCallable, GPTChat, get_price
from mas.mas import MetaMAS
from mas.utils import EmbeddingFunc

from envs import BaseEnv, BaseRecorder, get_env, get_recorder, get_task
from mas_workflow import get_mas
from prompts import get_dataset_system_prompt, get_task_few_shots
from utils import get_model_type
from step1_detact import ParallelGraphBuilder
from step1_detact import get_trajectory_from_logs


print(f"ok")


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
    mas: MetaMAS                # multi-agent system
    mas_config: dict = field(default_factory=dict)   # mas configs
    mem_config: dict = field(default_factory=dict)   # memory configs


def build_task(task: str, mas_type: str, memory_type: str, max_steps: int) -> TaskManager:

    with open(CONFIG.get(task).get('env_config_path')) as reader:
        config = yaml.safe_load(reader)

    env: BaseEnv = get_env(task, config, max_steps)
    recorder: BaseRecorder = get_recorder(task, working_dir=WORKING_DIR, namespace='total_task')
    tasks: list[dict] = get_task(task)
    mas_workflow: MetaMAS = get_mas(mas_type)
    mas_config: dict = CONFIG.get(mas_type, {})

    return TaskManager(
        task_name=task,
        mas_type=mas_type,
        memory_type=memory_type,
        tasks=tasks,
        env=env,
        recorder=recorder,
        mas=mas_workflow,
        mas_config=mas_config
    )   

def build_mas(
    task_manager: TaskManager,
    reasoning: str = None,
    mas_memory: str = None,
    llm_type: str = None,
) -> None:
    
    embed_func = EmbeddingFunc(CONFIG.get('embedding_model', "sentence-transformers/all-MiniLM-L6-v2")) 
    reasoning_module_type, mas_memory_module_type = module_map(reasoning, mas_memory)

    llm_model: LLMCallable = GPTChat(model_name=llm_type)
    reasoning_module: ReasoningBase = reasoning_module_type(llm_model=llm_model)
    mas_memory_module: MASMemoryBase = mas_memory_module_type(
        namespace=mas_memory,
        global_config=task_manager.mem_config,
        llm_model=llm_model,
        embedding_func=embed_func 
    )
    
    task_manager.mas.add_observer(task_manager.recorder)  
    task_manager.mas.build_system(reasoning_module, mas_memory_module, task_manager.env, task_manager.mas_config)

def run_task(task_manager: TaskManager, args = None, run_task_ids = None, builder: ParallelGraphBuilder = None, max_iter = 10, is_first_re = False, first_re_path = None) -> None:

    # 将轨迹和反思写入到本地文件中 命名方式就在 WORKING_DIR 后面建一个文件夹 就叫做 task_{task_id}_iter_{iter_id}
    record_out_dir_root = os.path.join(WORKING_DIR, 'refine_records')
    os.makedirs(record_out_dir_root, exist_ok=True)
    
    reflection_path_list = [
        "/data/G-Memory/GMemory-main/.db/Qwen3-8B_test_vision_no_GT_new/alfworld/macnet/g-memory/split_no_log_no_spatial_info_dict.json"
    ]
    target_test_index = [0,1,2]


    if is_first_re:
        reflection_advice = "" # 初始化为空

        # for cur_path in reflection_path_list:
        #     for idx in target_test_index:
        #         records = get_trajectory_from_logs(cur_path)
        #         builder.get_log(records, target_test_index=idx)
        #         reflection_results = builder.return_reflection()
        #         for key, value in reflection_results.items():
        #             print(f"{key}: {value}")
        #         reflection_advice = reflection_results.get('guidance', '')

        with open(first_re_path, 'r') as f:
            first_re_data = json.load(f)
            records = first_re_data['records']
        builder.get_log([records], target_test_index=0)
        reflection_results = builder.return_reflection()
        reflection_advice = reflection_results.get('guidance', '')
        for key, value in reflection_results.items():
            print(f"{key}: {value}")

    else:
        reflection_advice = "" # 初始化为空

    # exit(0)

    task_manager.recorder.dataset_begin()
    
    task_n = 0
    task_success = 0
    success_iters_set = list()
    success_first_set = list()


    
    for task_id, task_config in tqdm(enumerate(task_manager.tasks), total=len(task_manager.tasks), desc="Running Tasks"):
        if run_task_ids is not None and task_id not in run_task_ids:
            continue

        task_n += 1
        record_out_dir = os.path.join(record_out_dir_root, f'task_{task_id}')
        os.makedirs(record_out_dir, exist_ok=True)
        reflection_advice = "" # 每个新任务开始前初始化为空
        builder.reset_graph()
        
        for iter_id in range(max_iter):
            print(f"----- Refinement Iteration {iter_id} for Task {task_id} -----")

            if args.is_no_reflection:
                print("Reflection is disabled. -------------------")
                reflection_advice = ""

            task_manager.recorder.task_begin(task_id, task_config)  

            task_main, task_description = task_manager.mas.env.set_env(task_config) # 任务目标 看到一个碗 环境描述，你在一个环境里面，周围有一些东西
            few_shots: list[str] = get_task_few_shots(
                dataset=task_manager.task_name, # 指定任务 
                task_config=task_config, # 这个任务的一些参数
                few_shots_num=CONFIG.get(task_manager.task_name).get('few_shots_num', 0)
            )
            task_config.update(task_main=task_main, task_description=task_description, few_shots=few_shots, args=args, reflection_advice=reflection_advice)
            
            task_instruction: str = get_dataset_system_prompt(task_manager.task_name, task_config=task_config)
            for agent in task_manager.mas.agents_team.values():    
                task_manager.recorder.log(f'------------ MAS Agent: {agent.name} ------------')
                task_manager.recorder.log(agent.add_task_instruction(task_instruction))
            records = []
            if not args.is_no_step_reflection:
                reward, done, records = task_manager.mas.schedule(task_config, records, builder) 
            else:
                reward, done, records = task_manager.mas.schedule(task_config, records) 
                
            if not args.is_no_reflection:
                builder.get_log([records], target_test_index=0)
                reflection_results = builder.return_reflection()

                cur_record = {
                    'task_id': task_id,
                    'iter_id': iter_id,
                    'done': done,
                    'reflection_advice': reflection_advice,
                    'records': records,
                    'merge_steps': builder._return_merge_steps(),
                    'reflection_results': reflection_results
                }
            elif not args.is_no_step_reflection:
                cur_record = {
                    'task_id': task_id,
                    'iter_id': iter_id,
                    'done': done,
                    'reflection_advice': reflection_advice,
                    'records': records,
                    'merge_steps': '',
                    'reflection_results': ''
                }
            with open(os.path.join(record_out_dir, f'task_{task_id}_iter_{iter_id}.json'), 'w') as f:
                json.dump(cur_record, f, indent=4)
        
            task_manager.recorder.task_end(reward, done)       

            if not args.is_no_reflection:
                reflection_advice = reflection_results.get('guidance', '')
            
            if not args.is_no_step_reflection:
                builder.get_log([records], target_test_index=0)
                builder.next_graph()

            if done:
                task_success += 1
                # builder.reset_graph()
                if iter_id == 0:
                    success_first_set.append(task_id)
                else:
                    success_iters_set.append(task_id)
                break
    
    task_manager.recorder.dataset_end()
    print(f"Total tasks run: {task_n}")
    print(f"Total tasks success: {task_success}")
    print(f"Success Rate: {task_success / task_n * 100:.2f}%")
    print(f"Success on first iteration tasks: {len(success_first_set)}")
    print(f"Success on any iteration tasks: {len(success_iters_set)}")
    print("-"*50)
    print(f"Success on first iteration tasks: {success_first_set}")
    print("-"*50)
    print(f"Success on any iteration tasks: {success_iters_set}")

def print_args(args):
    for arg, value in vars(args).items():
        print(f"{arg}: {value}")


if __name__ == '__main__':
    # settings
    random.seed(42)

    parser = argparse.ArgumentParser(description='Run tasks with specified modules.')
    parser.add_argument('--task', type=str, default='alfworld', choices=['alfworld', 'fever', 'pddl', 'hotpotqa'])
    parser.add_argument('--mas_type', type=str, default='autogen', choices=['autogen', 'macnet', 'dylan'])
    parser.add_argument('--mas_memory', type=str, default='g-memory', help='Specify mas memory module')
    parser.add_argument('--Prefix', type=str, default='', help='Specify the prefix for working dir')
    parser.add_argument('--reasoning', type=str, default='io', help='Specify reasoning module')
    # parser.add_argument('--model', type=str, default='gpt-3.5-turbo-0125', help='Specify the LLM model type')
    parser.add_argument('--max_trials', type=int, default=30, help='max number of steps')
    parser.add_argument('--successful_topk', type=int, default=1, help='Number of successful trajs to be retrieved from memory.')
    parser.add_argument('--failed_topk', type=int, default=0, help='Number of failed trajs to be retrieved from memory.')
    parser.add_argument('--insights_topk', type=int, default=3, help='Number of insights to be retrieved from memory.')
    parser.add_argument('--threshold', type=float, default=0.0, help='threshold for traj similarity.')
    parser.add_argument('--use_projector', action='store_true', help='whether to use role projector.')
    parser.add_argument('--hop', type=int, default=1, help='hop for traj similarity.')

    parser.add_argument('--no_use_insights', action='store_true', help='whether to use insights.')  
    parser.add_argument('--no_use_successful_shots', action='store_true', help='whether to use use_successful_shots.')  # 
    
    parser.add_argument('--is_no_reflection', action='store_true')
    parser.add_argument('--reflection_type', type=str, default='specific', choices=['specific', 'general'])    
    
    parser.add_argument('--is_no_step_reflection', action='store_true')

    args = parser.parse_args()
    args.model = MODEL

    print_args(args)
    # exit(0)

    task: str = args.task
    mas_type: str = args.mas_type
    max_trials: int = args.max_trials
    model_type: str = args.model
    mas_memory_type: str = args.mas_memory
    reasoning_type: str = args.reasoning

    args.no_use_insights = True
    args.no_use_successful_shots = True


    if args.no_use_insights:
        print(f"不使用 insights 进行任务辅助！")
    else:
        print(f"使用 insights 进行任务辅助！")
    if args.no_use_successful_shots:
        print(f"不使用 successful shots 进行任务辅助！")
    else:
        print(f"使用 successful shots 进行任务辅助！")

    
    # dir
    if args.Prefix != "":
        WORKING_DIR = os.path.join('/data/G-Memory/GMemory-main/tasks/refine/out/logs', get_model_type(model_type) + f"{args.Prefix}", task, mas_type, f'{mas_memory_type}')
    else:
        WORKING_DIR = os.path.join('/data/G-Memory/GMemory-main/tasks/refine/out/logs', get_model_type(model_type), task, mas_type, f'{mas_memory_type}')
    # if os.path.exists(WORKING_DIR):
    #     shutil.rmtree(WORKING_DIR)
    os.makedirs(WORKING_DIR, exist_ok=True)

    # 将 args 保存到 WORKING_DIR/cur_run_config.yaml 中
    with open(os.path.join(WORKING_DIR, 'cur_run_config.yaml'), 'w') as f:
        yaml.dump(vars(args), f)

    # run tasks
    task_configs: TaskManager = build_task(task, mas_type, mas_memory_type, max_trials)
    task_configs.mas_config['successful_topk'] = args.successful_topk   # 1
    task_configs.mas_config['failed_topk'] = args.failed_topk           # 0
    task_configs.mas_config['insights_topk'] = args.insights_topk       # 3
    task_configs.mas_config['threshold'] = args.threshold               # 0.0
    task_configs.mas_config['use_projector'] = args.use_projector       # False
    task_configs.mem_config.update(
        working_dir=WORKING_DIR,
        hop=args.hop
    )
    # run_task_ids = [5, 8, 9, 40, 45, 91, 93, 101, 120, 124]
    run_task_ids = [9, 40, 93]

    build_mas(task_configs, reasoning_type, mas_memory_type, model_type)

    # builder = ParallelGraphBuilder(
    #     model_name=MODEL, 
    #     base_url=os.environ["OPENAI_API_BASE"], 
    #     api_key=os.environ["OPENAI_API_KEY"], 
    #     max_workers = 1
    # )
    builder = ParallelGraphBuilder(
        model_name=MODEL, 
        base_url=os.environ["OPENAI_API_BASE"], 
        api_key=os.environ["OPENAI_API_KEY"], 
        max_workers = 1, 
        reflection_type = args.reflection_type
    )

    run_task(task_configs, args, run_task_ids, builder, is_first_re=False, first_re_path="/data/G-Memory/GMemory-main/tasks/refine/out/logs/Qwen3-8B_test2/alfworld/macnet/g-memory/refine_records/task_5_iter_0.json")

    # # postprocess
    # completion_tokens, prompt_tokens, _ = get_price()
    # task_configs.recorder.log(f'completion_tokens:{completion_tokens}, prompt_tokens:{prompt_tokens}, price={completion_tokens*15/1000000+prompt_tokens*5/1000000}')

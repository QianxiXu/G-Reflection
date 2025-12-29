import re
import ast
import pprint # 用于格式化输出，使其更易读
import json 
import os
from pathlib import Path

def remove_timestamps(text_input):
    """
    独立封装的函数：
    移除字符串中长相为 "YYYY-MM-DD HH:MM:SS" 的时间字段。
    
    参数:
    text_input (str): 包含时间戳的原始字符串。
    
    返回:
    str: 移除了时间戳的字符串。
    """
    if not isinstance(text_input, str):
        return text_input
    
    # 匹配 "YYYY-MM-DD HH:MM:SS" 格式，以及其后可能跟随的 " -"
    # \s* 匹配时间戳前后的任何空白
    timestamp_pattern = re.compile(r'\s*\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}( -)?\s*')
    
    # 将匹配到的时间戳替换为单个空格
    cleaned_text = timestamp_pattern.sub(' ', text_input)
    
    # 额外清理，移除可能因替换而产生的多余空白
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
    
    return cleaned_text

# def parse_log(filepath='log.log'):
#     """
#     解析 Alfworld 日志文件，按 Task -> Trial 的嵌套结构提取信息。

#     参数:
#     filepath (str): log.log 文件的路径。

#     返回:
#     list: 
#         一个列表，每个元素都是一个 Task 字典。
#         Task 字典: {
#             'task_id': int,
#             'init_user_pxrompt': str,
#             'trials': List[Trial 字典]
#         }
#         Trial 字典: {
#             'trial_id': int,
#             'spatial_info_dict': dict,
#             'final_decision_result': str,
#             'step_result': str
#         }
#     """
#     import re
# import ast
# import pprint # 用于格式化输出，使其更易读

def remove_timestamps(text_input):
    """
    独立封装的函数：
    移除字符串中长相为 "YYYY-MM-DD HH:MM:SS" 的时间字段。
    
    参数:
    text_input (str): 包含时间戳的原始字符串。
    
    返回:
    str: 移除了时间戳的字符串。
    """
    if not isinstance(text_input, str):
        return text_input
    
    # 匹配 "YYYY-MM-DD HH:MM:SS" 格式，以及其后可能跟随的 " -"
    # \s* 匹配时间戳前后的任何空白
    timestamp_pattern = re.compile(r'\s*\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}( -)?\s*')
    
    # 将匹配到的时间戳替换为单个空格
    cleaned_text = timestamp_pattern.sub(' ', text_input)
    
    # 额外清理，移除可能因替换而产生的多余空白
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
    
    return cleaned_text

def parse_log(filepath='log.log'):
    """
    解析 Alfworld 日志文件，按 Task -> Trial 的嵌套结构提取信息。

    参数:
    filepath (str): log.log 文件的路径。

    返回:
    list: 
        一个列表，每个元素都是一个 Task 字典。
        Task 字典: {
            'task_id': int,
            'init_user_pxrompt': str,
            'trials': List[Trial 字典]
        }
        Trial 字典: {
            'trial_id': int,
            'spatial_info_dict': dict,
            'final_decision_result': str,
            'step_result': str
        }
    """
    
    def _clean_text(text):
        """内部辅助函数，用于清理提取的日志文本"""
        if not isinstance(text, str):
            return text
        
        # --- (已更正) ---
        # 1. 移除 标签
        # 正确的正则表达式是 r'\\'
        text = re.sub(r'\\', '', text)
        
        # 2. 移除 YYYY-MM-DD HH:MM:SS - 格式的时间戳
        text = re.sub(r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}( -)?', '', text)
        
        # 3. 移除换行符并替换为空格
        text = text.replace('\n', ' ').strip()
        
        # 4. 压缩多个空格为单个
        text = re.sub(r'\s+', ' ', text)
        return text

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            log_content = f.read()
    except FileNotFoundError:
        print(f"错误: 未在 '{filepath}' 找到日志文件。")
        return []
    except Exception as e:
        print(f"读取文件时出错: {e}")
        return []

    all_tasks_data = []
    
    # 层级 1：按 '---------- Task: \d+ ----------' 切分任务
    # [1:] 用于跳过日志文件开头到第一个 Task 之间的任何内容
    # re.split(pattern, string) 会返回一个列表，
    # 捕获组 (\d+) 也会被包含在结果中
    task_blocks = re.split(r'---------- Task: (\d+) ----------', log_content)[1:]
    # 1. 使用 '---------- Task: \d+ ----------' 作为分隔符来切分不同的任务
    # task_blocks = re.split(r'---------- Task: \d+ ----------', log_content)
    # task_blocks 现在是 [task_id_str, task_content, task_id_str, task_content, ...]
    # 我们需要将它们配对
    for i in range(0, len(task_blocks), 2):
        task_id = int(task_blocks[i])
        task_content = task_blocks[i+1]
        
        task_data = {
            'task_id': task_id,
            'init_user_pxrompt': None,
            'trials': [] # 存储该任务的所有 trial
        }

        # 层级 1 数据：提取 'init_user_pxrompt'
        prompt_match = re.search(
            r'init_user_pxrompt-------->:(.*?)init_user_pxrompt end --------<',
            task_content,
            re.DOTALL
        )
        if prompt_match:
            clean_prompt = prompt_match.group(1)
            # 清理 prompt 文本（移除 source 标签和多余空白）
            # clean_prompt = re.sub(r'\', '', raw_prompt)
            # clean_prompt = re.sub(r'\n+', '\n', raw_prompt).strip()
            task_data['init_user_pxrompt'] = clean_prompt
        else:
            print(f"警告: 在 Task {task_id} 中未找到 'init_user_pxrompt'。")

        # 层级 1 数据：提取 结果
        # result_match = re.search(
        #     fr'- now_the_trial (\d+) is end -----------------------------------------------------------<(.*?)---------- Task: {task_id + 1} ----------',
        #     task_content,
        #     re.DOTALL
        # )
        result_match = re.search(
            r'.*- now_the_trial (\d+) is end -----------------------------------------------------------<(.*)$',
            task_content,
            re.DOTALL
        )
        if result_match:
            task_result = _clean_text(result_match.group(2))
            task_data['task_result'] = task_result
        else:
            print(f"警告: 在 Task {task_id} 中未找到任务结果。")

        # 层级 2：在 Task 内部按 Trial 切分
        # re.finditer 会查找所有不重叠的匹配
        # (\d+) 捕获 trial 编号, (.*?) 捕获中间内容, \1 确保结束编号与开始编号一致
        trial_matches = re.finditer(
            r"- now_the_trial is (\d+) ----------------------------------------------------------->:(.*?)- now_the_trial \1 is end -----------------------------------------------------------<",
            task_content,
            re.DOTALL
        )
        
        for trial_match in trial_matches:
            trial_id = int(trial_match.group(1))
            trial_content = trial_match.group(2) # 两个标记之间的所有内容

            trial_data = {
                'trial_id': trial_id,
                'spatial_info_dict': None,
                'final_decision_result': None,
                'step_result': None
            }
            
            # --- 层级 2 数据提取 ---
            # (在 trial_content 内部查找)

            # 1. 提取 'spatial_info' 字典
            spatial_match = re.search(r"spatial_info for agent dicision is :(.*?)temporal_info for agent dicision is :", trial_content, re.DOTALL)
            if spatial_match:
                dict_str = spatial_match.group(1).strip()
                if dict_str:
                    dict_start = dict_str.find('{')
                    dict_end = dict_str.rfind('}')
                    if dict_start != -1 and dict_end != -1 and dict_end > dict_start:
                        clean_dict_str = dict_str[dict_start : dict_end + 1]
                        try:
                            trial_data['spatial_info_dict'] = ast.literal_eval(clean_dict_str)
                        except (ValueError, SyntaxError, TypeError) as e:
                            print(f"--- 警告: Task {task_id}, Trial {trial_id} 字典解析失败: {e} ---")
                    else:
                        print(f"--- 警告: Task {task_id}, Trial {trial_id} 未找到有效字典结构 ---")

            # 2. 提取 '决策执行节点的最终结果为-->'
            final_decision_match = re.search(r"决策执行节点的最终结果为-->:(.*?)本步骤执行的结果为-->:", trial_content, re.DOTALL)
            if final_decision_match:
                trial_data['final_decision_result'] = _clean_text(final_decision_match.group(1))

            # 3. 提取 '本步骤执行的结果为-->'
            # 搜索范围是 "本步骤执行..." 到 trial_content 的末尾 (因为 trial_content 已被 "end" 标记截断)
            step_result_match = re.search(r"本步骤执行的结果为-->:(.*)", trial_content, re.DOTALL)
            if step_result_match:
                trial_data['step_result'] = _clean_text(step_result_match.group(1))
            
            task_data['trials'].append(trial_data)
        
        all_tasks_data.append(task_data)

    return all_tasks_data

def run_parse_log():
    # 假设 log.log 与此脚本在同一目录下
    # log_file_path = '/Users/liuyi/Documents/Docker_env/Agent/GMemory-main/.db/Qwen3-235B-A22B-Instruct-2507_test_vision/alfworld/macnet/g-memory/total_task.log'
    log_file_path = "/data/G-Memory/GMemory-main/.db/gpt-4o-mini_test_vision_no_GT/hotpot/macnet/g-memory/total_task.log"
    
    parsed_data = parse_log(log_file_path)
    
    if parsed_data:
        print(f"成功解析了 {len(parsed_data)} 个任务。\n")
        
        # --- (新增) 保存到 JSON 文件 ---
        try:
            # 1. 获取 log.log 所在的目录
            # os.path.abspath 获取绝对路径，os.path.dirname 获取目录
            base_directory = os.path.dirname(os.path.abspath(log_file_path))
            
            # 2. 定义输出文件名
            output_filename = 'log.json'
            
            # 3. 组合成完整的输出路径
            output_filepath = os.path.join(base_directory, output_filename)
            
            # 4. 写入 JSON 文件
            with open(output_filepath, 'w', encoding='utf-8') as f:
                # indent=4 使 JSON 文件格式化，易于阅读
                # ensure_ascii=False 确保中文等字符正确写入，而不是被转义
                json.dump(parsed_data, f, indent=4, ensure_ascii=False)
                
            print(f"--- 成功将解析结果保存到: {output_filepath} ---")

        except Exception as e:
            print(f"--- 错误：保存 JSON 文件失败 ---")
            print(f"{e}")
        # --- 保存功能结束 ---
        exit(0)
        # 使用 pprint 格式化打印结果
        pp = pprint.PrettyPrinter(indent=2, width=120)
        
        print("--- 'remove_timestamps' 函数使用示例 ---")
        test_string = "> think: 任务完成 2025-11-16 10:50:39 - 这是一个测试"
        print(f"  原始字符串: {test_string}")
        print(f"  清理后字符串: {remove_timestamps(test_string)}")
        
        print("\n\n--- 整体解析结果 (只显示 Task 0 结构) ---")
        # 只打印第一个任务，避免刷屏
        if parsed_data:
            pp.pprint(parsed_data[0]) 
        
        
        # --- 示例：展示如何访问数据 ---
        print("\n\n--- 访问示例 ---")
        
        # 打印 Task 0 的 'init_user_pxrompt' (前200字符)
        if len(parsed_data) > 0 and parsed_data[0]['init_user_pxrompt']:
            print(f"\n** Task 0 的 'init_user_pxrompt' (前200字符):**")
            print(parsed_data[0]['init_user_pxrompt'][:200] + "...")
            
        # 打印 Task 0 的第一个 Trial (Trial 0)
        if len(parsed_data) > 0 and len(parsed_data[0]['trials']) > 0:
            print(f"\n** Task 0, Trial 0 的 'spatial_info_dict':**")
            pp.pprint(parsed_data[0]['trials'][0]['spatial_info_dict'])
            
            print(f"\n** Task 0, Trial 0 的 'final_decision_result':**")
            pp.pprint(parsed_data[0]['trials'][0]['final_decision_result'])
            
            print(f"\n** Task 0, Trial 0 的 'step_result':**")
            pp.pprint(parsed_data[0]['trials'][0]['step_result'])
        
        # 打印 Task 1 的 Trial 数量
        if len(parsed_data) > 1:
             print(f"\n** Task 1 共有 {len(parsed_data[1]['trials'])} 个 Trials。**")

    else:
        print("未解析到任何数据。请检查 'log.log' 文件是否存在且内容正确。")


def load_all_pddls_in_task(gt_traj_file_path: str) -> dict:
    """
    读取给定 game.tw-pddl 文件所在的任务目录中，
    所有 trial 目录下的 game.tw-pddl 文件内容。

    参数:
    gt_traj_file_path (str): 任何一个 trial 目录下的 game.tw-pddl 文件路径。
                             例如: ".../look_at_obj_in_light-Bowl-None-DeskLamp-308/trial_T.../game.tw-pddl"

    返回:
    dict: 一个字典，映射 {pddl_file_path: pddl_json_content}
    """
    
    # 1. 实现你的逻辑：cur_task_path = str(gt_traj_file).split('/')[:-2]
    #    我们使用 Path.parent.parent 会更健壮
    try:
        task_dir_path = Path(gt_traj_file_path).parent.parent
    except Exception as e:
        print(f"获取任务目录失败: {e}")
        return {}

    if not task_dir_path.is_dir():
        print(f"错误: 任务目录 {task_dir_path} 不存在或不是一个目录。 input {gt_traj_file_path}")
        return {}

    # print(f"--- 正在扫描任务目录: {task_dir_path} ---")
    
    all_pddl_data = {}

    # 2. 读取 cur_task_path 文件夹下的所有文件名 (非递归)
    #    os.listdir() 完美符合这个要求
    for entry_name in os.listdir(task_dir_path):
        trial_dir_path = os.path.join(task_dir_path, entry_name)
        
        # 3. 确保我们只查看 trial 目录 (检查是否是文件夹)
        if os.path.isdir(trial_dir_path):
            
            # 4. 构建 game.tw-pddl 的路径
            pddl_file_path = os.path.join(trial_dir_path, "game.tw-pddl")
            if pddl_file_path == gt_traj_file_path:
                continue
            
            # 5. 检查文件是否存在
            if os.path.exists(pddl_file_path):
                try:
                    # 6. 读取 game.tw-pddl 的内容 (使用 json.load)
                    with open(pddl_file_path, 'r', encoding='utf-8') as f:
                        game_data = json.load(f)
                        
                    all_pddl_data[pddl_file_path.split('/')[-2]] = game_data['walkthrough']
                    
                    
                except json.JSONDecodeError:
                    print(f"[错误] {pddl_file_path} 不是一个有效的 JSON。")
                except Exception as e:
                    print(f"[错误] 读取 {pddl_file_path} 失败: {e}")
            else:
                # 这个 trial 文件夹下没有 game.tw-pddl
                print(f"[跳过] {trial_dir_path} (未找到 game.tw-pddl)")
        else:
            # 这不是一个目录 (例如可能是 .DS_Store 等文件)
            print(f"[跳过] {entry_name} (这不是一个目录), input {gt_traj_file_path}")

    # print(f"--- 扫描完成 ---")
    return all_pddl_data

def get_GT_data():
    out_data = {}
    out_path = "/Users/liuyi/Documents/Docker_env/Agent/GMemory-main/.db"
    test_data_path = "/Users/liuyi/Documents/Docker_env/Agent/GMemory-main/data/alfworld/alfworld_tasks_suffix.json"
    with open(test_data_path, 'r') as f:
        test_data = json.load(f)

    for index, data in enumerate(test_data):
        goal = data['goal']
        gamefile = data['gamefile']
        # gt_file = "".join(gamefile.split('/')[:-1]) + 'traj_data.json'
        gt_file = Path(gamefile).parent / 'traj_data.json'
        gt_traj_file = Path(gamefile).parent / 'game.tw-pddl'
        cur_gt = json.load(open(gt_file, 'r'))
        cur_gt_traj = open(gt_traj_file, 'r').read()
        cur_gt_traj = json.loads(cur_gt_traj)
        all_pddl_data = load_all_pddls_in_task(str(gt_traj_file))
        out_data[index] = {
            'goal': goal,
            'gamefile': gamefile,
            'gt_traj': cur_gt_traj['walkthrough'],
            'other_gt_trajs': all_pddl_data,
            "plan": cur_gt['plan']['high_pddl'],
            'turk_annotations': cur_gt['turk_annotations'],
        }
    json.dump(out_data, open(os.path.join(out_path, 'alfworld_gt_data.json'), 'w'), indent=4)


# --- 脚本主执行区 ---
if __name__ == "__main__":
    print()
    run_parse_log()
    # get_GT_data()

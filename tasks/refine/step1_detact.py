import json
import re
import networkx as nx
from typing import List, Dict, Tuple
import concurrent.futures
from functools import partial
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import os
from pyvis.network import Network
from refine_prompts import *
from collections import defaultdict
import torch
import mas.llm as llm

from mas.utils import load_config 

from sentence_transformers import SentenceTransformer, util
import torch

# 如果你有 OpenAI Key，请填入；如果没有，保持为 None，代码会使用 Mock 数据演示流程
OPENAI_API_KEY = "TDM6IgMVUcG9sfHeweMMgrUD4ptayo8J"
OPENAI_API_BASE = "https://antchat.alipay.com/v1"

CONFIG: dict = load_config("configs/configs.yaml")
LLM_CONFIG: dict = CONFIG.get("llm_config", {})
USE_OLLAMA = LLM_CONFIG.get("use_ollama", True)



class ParallelGraphBuilder:
    def __init__(self, model_name: str = "gpt-4o", base_url: str = OPENAI_API_BASE, api_key: str = OPENAI_API_KEY, max_workers: int = 2, reflection_type: str = 'specific'):
        self.raw_logs = None
        self.structured_steps = []

        self.graphs = [nx.DiGraph()]
        self.curr_graph = 0
        self.model_name = model_name
        self.base_url = base_url
        self.api_key = api_key
        from openai import OpenAI
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        self.max_workers = max_workers # 并发数量
        self.window_size = 5 # 设置滑动窗口大小
        self.reflection_type = reflection_type # 'specific' or 'general'
        
    def next_graph(self):
        self.graphs[self.curr_graph] = nx.DiGraph()
        self.build_graph_parallel()
        self.curr_graph += 1
        self.graphs.append(nx.DiGraph())
    
    def reset_graph(self):
        self.graphs = []
        self.graphs.append(nx.DiGraph())
        self.curr_graph = 0
    
    def get_log(self, raw_logs: List[List[str]], target_test_index = None):
        """
        获取日志内容
        """
        self.raw_logs = raw_logs
        self.structured_steps = []
        for log in raw_logs:
            cur_structured_steps = self._parse_logs(log)
            self.structured_steps.append(cur_structured_steps)
        self.test_index = None
        if target_test_index is not None:
            self.structured_steps = self.structured_steps[target_test_index]
            self.test_index = target_test_index

         # --- 新增：步骤重组 ---
        self.merged_steps = self._merge_steps(self.structured_steps)
        
    def get_graph_guidance(self):
        """
        通过找到与当前运行轨迹最相似的图，总结guidance
        目前的相似度计算方法是把图转化成文字然后embedding算相似度
        """
        max_sim = 0.0
        max_sim_id = 0
        for curr_id in range(0, self.curr_graph):
            curr_sim = self._graph_similarity(graph_id=curr_id)
            if curr_sim > max_sim:
                max_sim = curr_sim
                max_sim_id = curr_id
        curr_graph_text = self.export_graph_for_diagnosis()
        guide_graph_text = self.export_graph_for_diagnosis(max_sim_id)
        prompt = f"""
You are an expert Task Strategist. Your goal is to provide actionable guidance for an AI Agent based on its historical experience.

[Historical Experience]
Below are the most similar past cases found in the memory:
{guide_graph_text}

[Instruction]
Based on the similarities and differences between the current trajectory and past experiences:
1. Identify potential pitfalls or successful patterns from the history.
2. Provide a concise "Guidance" for the NEXT step. 
3. If the past experience failed, tell the agent what to avoid. If it succeeded, tell it what to replicate.

Output your response in the following format:
- Analysis: <briefly compare current and past>
- Guidance: <specific instruction for the next step>
"""
        if USE_OLLAMA:
            response = self.client.chat.completions.create(
                model=self.model_name, 
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that generates task guidance."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0 # 较低的随机性以保证逻辑严谨
            )
            return response.choices[0].message.content
        
        else:
            messages = [
                        {"role": "system", "content": "You are a helpful assistant that generates task guidance."},
                        {"role": "user", "content": prompt}
                    ]
            input_text = llm._qwen_tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                        enable_thinking=False
                    )
            inputs = llm._qwen_tokenizer(
                input_text,
                return_tensors="pt"
            ).to(llm._qwen_model.device)
            with torch.no_grad():
                outputs = llm._qwen_model.generate(
                    **inputs,
                    max_new_tokens=4096,
                    do_sample=False,
                    temperature=None
                )

            output_text = llm._qwen_tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[-1]:],
                skip_special_tokens=True
            )
            
            return output_text
        
    def _graph_similarity(self, graph_id: int):
        """
        计算当前的图与graph_id代表的图的相似度
        
        """
        assert graph_id >= 0
        curr_graph_text = self.export_graph_for_diagnosis()
        past_graph_text = self.export_graph_for_diagnosis(graph_id)
        embedding_model = SentenceTransformer('/data/models/nomic-embed-text-v1.5', trust_remote_code=True)
        curr_embeddings = embedding_model.encode(curr_graph_text, convert_to_tensor=True)
        past_embeddings = embedding_model.encode(past_graph_text, convert_to_tensor=True)
        cosine_scores = util.cos_sim(curr_embeddings, past_embeddings)
        return cosine_scores
    
    def _parse_logs(self, logs: List[str]) -> List[Dict]:
        """
        将原始的字符串 List 解析为结构化的字典列表。
        处理格式： "Act X: ... Obs X: ..."
        """
        structured_data = []
        
        # 处理 Step 0 (Task Description)
        if logs:
            structured_data.append({
                "step_id": 0,
                "type": "Task",
                "action_text": "Task Description & Initial Observation",
                "observation_text": "",
                "content": logs[0],
            })

        # 处理后续 Steps
        # 正则匹配模式：Act [数字]: [内容] Obs [数字]: [内容]
        # pattern = re.compile(r"Act (\d+): (.*?) Obs \1: (.*)", re.DOTALL)
        pattern = re.compile(r"Act (\d+): (.*?)\s+Obs \1: (.*)", re.DOTALL)
        for i, log_str in enumerate(logs[1:], start=1):
            match = pattern.search(log_str)
            if match:
                step_num = int(match.group(1))  # 数字
                action_content = match.group(2).strip() # 动作内容
                obs_content = match.group(3).strip()    # 观察内容
                # 细分：如果是 think，单独标记
                act_type = "Think" if action_content.startswith("think:") else "Action"
                structured_data.append({
                    "step_id": i,
                    "type": act_type,
                    "action_text": action_content,
                    "observation_text": obs_content,
                    "raw": log_str
                })

        return structured_data

    def _merge_steps(self, raw_steps: List[Dict]) -> List[Dict]:
        """
        【核心修改】步骤重组逻辑
        将 Think 和随后的 Action 合并为一个逻辑步骤 (Merged Step)。
        """
        merged = []
        current_block = None
        
        # 重新编号计数器
        new_id_counter = 0

        for step in raw_steps:
            s_type = step['type']
            
            if s_type == 'Task':
                # Task 单独作为第 0 步
                merged.append({
                    "merged_id": new_id_counter,
                    "type": "Task",
                    "content": step['content'],
                    "original_step_ids": [step['step_id']] # 映射关系
                })
                new_id_counter += 1
            
            elif s_type == 'Think':
                # 遇到新的 Think，无论之前是否有 Block，都必须结束之前的，开启新的
                # 因为 Think 代表了新的认知过程开始
                if current_block:
                    merged.append(current_block)
                    new_id_counter += 1
                
                # 开启一个新的 Block
                current_block = {
                    "merged_id": new_id_counter,
                    "type": "CognitiveUnit",
                    "think_text": step['action_text'],
                    "actions": [], # 修改为列表，存储多个 Action
                    "original_step_ids": [step['step_id']] # 记录 Think ID
                }
            
            elif s_type == 'Action':
                # Action 归属于当前的 Think Block
                if current_block:
                    # 将 Action 添加到当前 Block 的 actions 列表中
                    current_block['actions'].append({
                        "action_text": step['action_text'],
                        "observation_text": step['observation_text'],
                        "step_id": step['step_id']
                    })
                    current_block['original_step_ids'].append(step['step_id']) # 追加 Action ID
                else:
                    # 边缘情况：如果没有 Think 直接 Action (通常不会发生，或者是第一个步骤就是 Action)
                    # 创建一个没有 Think 的 Block
                    current_block = {
                        "merged_id": new_id_counter,
                        "type": "CognitiveUnit",
                        "think_text": "No explicit thought recorded.",
                        "actions": [{
                            "action_text": step['action_text'],
                            "observation_text": step['observation_text'],
                            "step_id": step['step_id']
                        }],
                        "original_step_ids": [step['step_id']]
                    }
                    # 注意：这里不立即 append，因为后面可能还有连续的 Action

        # 循环结束后，处理最后一个遗留的 Block
        if current_block:
            merged.append(current_block)

        return merged


    def _generate_window_prompt(self, start_idx: int, reflection_type = 'specific') -> str:
        """
        基于【重组后的步骤】生成 Prompt
        适配新的 merged_steps 结构 (包含 actions 列表)
        """
        end_idx = start_idx + self.window_size
        window_steps = self.merged_steps[start_idx : end_idx]
        
        # 1. 获取任务描述
        task_desc = "Task description not found."
        if len(self.merged_steps) > 0 and self.merged_steps[0]['type'] == 'Task':
            task_desc = self.merged_steps[0]['content']

        # 2. 构建窗口内的轨迹字符串
        history_str = ""
        for step in window_steps:
            m_id = step['merged_id']
            
            if step['type'] == "Task":
                history_str += f"=== [Step {m_id}] SYSTEM TASK CONFIGURATION ===\n"
                history_str += f"Content: {step['content']}\n"
                history_str += f"==================================================\n\n"
            else:
                # 展示 Think -> Actions -> Obs 的组合
                history_str += f"[Step {m_id}] COGNITIVE UNIT:\n"
                history_str += f"  THOUGHT: {step['think_text']}\n"
                
                # 遍历该 Think 下的所有 Actions
                actions = step.get('actions', [])
                if not actions:
                    history_str += f"\n"
                else:
                    for i, act in enumerate(actions):
                        # 如果有多个动作，可以标记为 Action 1, Action 2...
                        prefix = "ACTION" if len(actions) == 1 else f"ACTION {i+1}"
                        history_str += f"      {prefix}:  {act['action_text']}\n"
                        history_str += f"      RESULT:  {act['observation_text']}\n"
                
                history_str += "\n"

        # 3. 调用 get_naive_prompt
        # prompt = get_naive_prompt(task_desc, history_str)
        # return prompt, 'window'
        prompt_spe, prompt_ger = get_naive_prompt(task_desc, history_str)
        if reflection_type == 'specific':
            return prompt_spe, 'window'
        elif reflection_type == 'general':
            return prompt_ger, 'window'
        else:
            raise ValueError(f"Unknown reflection_type: {reflection_type}")

    def _analyze_single_window(self, start_idx: int) -> Dict:
        """
        单个步骤的分析函数，将被放入线程池运行
        """

        prompt, step_type = self._generate_window_prompt(start_idx, self.reflection_type)
        print('-'*50)
        print(prompt)
        print('-'*50)
        # step_id = self.structured_steps[step_idx]['step_id']
        # prompt, step_type = self._generate_prompt_for_step(step_idx, 'test')

        # 简单的重试机制
        max_retries = 3
        for attempt in range(max_retries):
            try:
                if USE_OLLAMA:
                    extra_body = {}
                    if "Qwen3-8B" in self.model_name:
                        extra_body = {"chat_template_kwargs": {"enable_thinking": False}}

                    response = self.client.chat.completions.create(
                        model=self.model_name, 
                        messages=[
                            {"role": "system", "content": "You are a helpful assistant that outputs strict JSON."},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=0,
                        response_format={"type": "json_object"},
                        extra_body=extra_body
                    )
                    response_json = response.choices[0].message.content
                        # 清理 Markdown 标记
                    response_json = re.sub(r"```json\s*|\s*```", "", response_json).strip()
                    data = json.loads(response_json)
                    
                    
                else:
                    messages = [
                        {"role": "system", "content": "You are a helpful assistant that outputs strict JSON."},
                        {"role": "user", "content": prompt}
                    ]
                                        
                    prompt = llm._qwen_tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                        enable_thinking=False
                    )
                    inputs = llm._qwen_tokenizer(
                        prompt,
                        return_tensors="pt"
                    ).to(llm._qwen_model.device)
                    with torch.no_grad():
                        outputs = llm._qwen_model.generate(
                            **inputs,
                            max_new_tokens=4096,
                            do_sample=False,
                            temperature=None
                        )

                    output_text = llm._qwen_tokenizer.decode(
                        outputs[0][inputs["input_ids"].shape[-1]:],
                        skip_special_tokens=True
                    )
                
                    # 清理 Markdown 标记
                    response_json = re.sub(r"```json\s*|\s*```", "", output_text).strip()
                    data = json.loads(response_json)
                
                # return {
                #     "window_start": start_idx,
                #     "problematic_steps": data.get("problematic_steps", [])
                # }
                return {
                    "window_start": start_idx,
                    "step_id": data.get("step_id"),
                    "reasoning": data.get("reasoning"),
                    "guidance" : data.get('correction_suggestion')
                }

            except Exception as e:
                print(f"Error in window {start_idx} (Attempt {attempt+1}): {e}")
                time.sleep(1)
        
        # 失败返回空
        # return {"window_start": start_idx, "problematic_steps": [], "error": "Max retries exceeded"}
        return {"window_start": start_idx, "step_id": None, "reasoning": None, "guidance": None}

    # --- 指标计算函数 1: 绝对票数 ---
    def _calculate_raw_votes(self, results: List[Dict], is_print: bool = False) -> Tuple[Dict, Dict]:
        """
        计算每个 Merged Step 的总得票数
        Returns:
            vote_counts: {merged_id: count}
            vote_reasons: {merged_id: [reasons]}
        """
        vote_counts = defaultdict(int)
        vote_reasons = defaultdict(list)

        # 临时存储详细信息用于打印: {merged_id: [{"reason": str, "window_start": int}, ...]}
        vote_details = defaultdict(list)

        for res in results:
            window_start = res.get("window_start", -1)
            probs = res.get("problematic_steps", [])
            for item in probs:
                s_id = item.get("step_id") # 这里是 Merged ID
                reason = item.get("reason")
                if s_id is not None:
                    vote_counts[s_id] += 1
                    vote_reasons[s_id].append(reason)
                    vote_details[s_id].append({
                        "reason": reason,
                        "window_start": window_start
                    })

        
        if is_print:
            print("\n" + "="*60)
            print("=== RAW VOTE DETAILS (Sorted by Vote Count) ===")
            print("="*60)
            
            # 按得票数降序排序
            sorted_votes = sorted(vote_counts.items(), key=lambda x: x[1], reverse=True)
            
            for m_id, count in sorted_votes:
                # 获取步骤内容
                merged_step_data = next((s for s in self.merged_steps if s['merged_id'] == m_id), None)
                content_str = "Unknown Step"
                if merged_step_data:
                    think = merged_step_data.get('think_text', '')
                    actions = merged_step_data.get('actions', [])
                    act_texts = [a['action_text'] for a in actions]
                    act_str = " -> ".join(act_texts) if act_texts else "No Action"
                    content_str = f"Think: {think}\nActions: {act_str}"

                print(f"\n[Merged Step {m_id}] - Total Votes: {count}")
                print(f"Content:\n{content_str}")
                print("-" * 30)
                print("Vote Details:")
                
                # 打印每一票的详情
                for detail in vote_details[m_id]:
                    print(f"  • Window Start: {detail['window_start']}")
                    print(f"    Reason: {detail['reason']}")
                print("-" * 60)

        return vote_counts, vote_reasons

    # --- 指标计算函数 2: 投票比例 (Score) ---
    def _calculate_vote_ratios(self, results: List[Dict], vote_counts: Dict) -> List[Dict]:
        """
        计算投票比例 = 得票数 / 曝光次数
        """
        exposure_counts = defaultdict(int)
        total_merged_steps = len(self.merged_steps)

        # 统计曝光次数 (分母)
        for res in results:
            window_start = res.get("window_start", 0)
            window_end = min(window_start + self.window_size, total_merged_steps)
            
            for idx in range(window_start, window_end):
                # 只有 CognitiveUnit (非 Task) 才参与分母统计
                if self.merged_steps[idx]['type'] == 'CognitiveUnit':
                    m_id = self.merged_steps[idx]['merged_id']
                    exposure_counts[m_id] += 1
        
        # 计算最终得分
        step_scores = []
        for m_id, exposures in exposure_counts.items():
            votes = vote_counts.get(m_id, 0)
            score = 0.0
            if exposures > 0:
                score = votes / exposures
            
            step_scores.append({
                "merged_id": m_id,
                "score": score,
                "votes": votes,
                "exposures": exposures
            })
        
        # 排序：按分数降序，分数相同按票数降序
        step_scores.sort(key=lambda x: (x['score'], x['votes']), reverse=True)
        return step_scores

    def return_reflection(self) -> str:
        """
        解析完整轨迹 返回反思内容和指导意见
        """

        total_merged_steps = len(self.merged_steps)
        print(f"Total Merged Steps: {total_merged_steps} (Original: {len(self.structured_steps)})")

        # 1. 生成滑动窗口 (基于 Merged Steps) 这次窗口大小直接拉到满
        if total_merged_steps <= self.window_size:
            window_starts = [0]
        else:
            self.window_size = total_merged_steps # 这次窗口大小直接拉到满
            window_starts = list(range(0, total_merged_steps - self.window_size + 1))

        # 2. 并行执行
        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = []
            print(f">>> Submitting window tasks...")
            for start_idx in tqdm(window_starts, desc="Submitting"):
                future = executor.submit(self._analyze_single_window, start_idx)
                futures.append(future)
                time.sleep(0.1) # 避免并发过快

            print(f">>> Waiting for results...")
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Analyzing Windows"):
                try:
                    results.append(future.result())
                except Exception as e:
                    print(f"Unexpected error: {e}")

        # 解析结果
        return results[0]

    def _return_merge_steps(self) -> List[Dict]:
        """
        返回重组后的步骤
        """
        return self.merged_steps
    

    def build_graph_parallel(self):
        """
        并行执行滑动窗口分析，并统计投票结果
        """
        total_merged_steps = len(self.merged_steps)
        print(f"Total Merged Steps: {total_merged_steps} (Original: {len(self.structured_steps)})")

        # 1. 生成滑动窗口 (基于 Merged Steps)
        if total_merged_steps <= self.window_size:
            window_starts = [0]
        else:
            window_starts = list(range(0, total_merged_steps - self.window_size + 1))

        print(f"Total Merged Steps: {total_merged_steps}, Window Size: {self.window_size}")
        print(f"Generated {len(window_starts)} sliding windows for analysis.")

        # 2. 并行执行
        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = []
            print(f">>> Submitting window tasks...")
            for start_idx in tqdm(window_starts, desc="Submitting"):
                future = executor.submit(self._analyze_single_window, start_idx)
                futures.append(future)
                time.sleep(0.1) # 避免并发过快

            print(f">>> Waiting for results...")
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Analyzing Windows"):
                try:
                    results.append(future.result())
                except Exception as e:
                    print(f"Unexpected error: {e}")

         # 3. 指标计算
        vote_counts, vote_reasons = self._calculate_raw_votes(results, is_print=True)
        step_scores = self._calculate_vote_ratios(results, vote_counts)

        # 4. 输出报告 (映射回原始步骤)
        print("\n" + "="*60)
        print("=== PROBLEMATIC STEP ANALYSIS (Merged Steps) ===")
        print("="*60)

        final_report = []

        for item in step_scores:
            m_id = item['merged_id']
            score = item['score']
            
            if score == 0: continue # 过滤掉无错步骤

            # 获取重组后的步骤信息
            merged_step_data = next((s for s in self.merged_steps if s['merged_id'] == m_id), None)
            
            if merged_step_data:
                # 提取原始 ID 用于展示
                original_ids = merged_step_data.get('original_step_ids', [])
                think_content = merged_step_data.get('think_text', '')
                
                # 处理 actions 列表用于展示
                actions_list = merged_step_data.get('actions', [])
                # 提取所有动作文本
                action_texts = [a['action_text'] for a in actions_list]
                # 拼接显示，如果有多个动作，用 " -> " 连接
                action_display = " -> ".join(action_texts) if action_texts else "No Action"

                print(f"\n[Merged Step {m_id}] -> Original Steps: {original_ids}")
                print(f"Score: {score:.2f} (Votes: {item['votes']} / Exposures: {item['exposures']})")
                print(f"Thought: {think_content}")
                print(f"Actions: {action_display}")
                
                reasons = list(set(vote_reasons[m_id]))
                print("Reasons:")
                for r in reasons[:2]:
                    print(f" - {r}")

                final_report.append({
                    "merged_step_id": m_id,
                    "original_step_ids": original_ids,
                    "score": score,
                    "votes": item['votes'],
                    "exposures": item['exposures'],
                    "content": {
                        "think": think_content,
                        "actions": action_texts # 保存为列表，方便后续处理
                    },
                    "reasons": reasons
                })
        

        # 写入本地：
        results.sort(key=lambda x: x.get('step_id', 0))
        with open("/data/G-Memory/GMemory-main/tasks/refine/out/logs/window_analysis_results.json", "w") as f:
            json.dump(results, f, indent=2)
        
        # exit(0)
        
        # 按得分降序排列，得分相同时按得票数降序
        step_scores.sort(key=lambda x: (x['score'], x['votes']), reverse=True)

     

        # 保存最终统计结果
        output_path = f"/data/G-Memory/GMemory-main/tasks/refine/out/voting_results_{self.test_index}.json"
        with open(output_path, 'w') as f:
            json.dump(final_report, f, indent=2)
        print(f"\nScoring results saved to: {output_path}")

       

        # exit(0) 

        # 1. 添加所有节点
        for step in self.structured_steps:
            self.graphs[self.curr_graph].add_node(step['step_id'], **step)
        
        # 2. 并发执行 LLM 分析 (跳过 Node 0，从 Node 1 开始)
        tasks = list(range(1, len(self.structured_steps)))
        # tasks = [len(self.structured_steps) -1] # 就做最后一步，所有步骤测试
        # tasks = [15]
        results_map = {} # 使用字典存储结果，方便后续按 ID 查找
        
        print(f"Starting parallel analysis for {len(tasks)} steps with {self.max_workers} workers...")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = []
            
            # 阶段 A: 慢速提交任务 (Rate Limiting)
            print(f">>> Submitting tasks (interval: 0.2s)...")
            for i in tqdm(tasks, desc="Submitting Tasks"):
                future = executor.submit(self._analyze_single_window, i)
                futures.append(future)
                time.sleep(0.20)  # 【关键点 2】强制每隔 0.2 秒提交一个任务，避免瞬间 QPS 爆炸

            # 阶段 B: 获取结果 (Progress Bar)
            # 使用 tqdm 包装 as_completed，这样每当一个任务完成，进度条就会走一格
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="AI Analysis Progress"):
                try:
                    res = future.result()
                    # 现在 res 里面肯定有 'step_id' 了，因为我们在 _analyze_single_step 里注入了
                    if 'step_id' in res:
                        results_map[res['step_id']] = res
                    else:
                        print(f"Error: Result missing step_id: {res}")
                except Exception as e:
                    print(f"Unexpected error in thread: {e}")


        
        # 3. 收集结果并添加边
        # 将 map 转回 list 并排序，方便保存文件
        results_list = list(results_map.values())
        results_list.sort(key=lambda x: x.get('step_id', 0))
        for item in results_list:
            print(f"Step {item['step_id']} analysis result: \n{item['reasoning']}\n\n")
        # exit(0)
        
        # 分别将依赖 和 不足定位 信息 写入本地
        output_path = f"/data/G-Memory/GMemory-main/tasks/refine/out/step_results_{self.test_index}.json"
        # output_path2 = "/Users/liuyi/Documents/Docker_env/Agent/GMemory-main/tasks/refine/out/step_Error_results.json"
        with open(output_path, 'w') as f:
            json.dump(results_list, f, indent=2)
        print(f"Analysis results saved to {output_path}")
        # exit(0)


        # 遍历所有步骤（跳过 Step 0 Task），根据类型分别处理
        for i in range(1, len(self.structured_steps)):
            step = self.structured_steps[i]
            step_id = step['step_id']
            step_type = step['type']
            
            # 从 map 中获取结果
            llm_res = results_map.get(step_id, {})
            
            # 记录 LLM 的推理/理由 (通用)
            # 注意：Action 的 prompt 里没有 reason 字段，Think 里有 root_cause_vote
            # 这里做一个简单的兼容处理
            if step_type == 'Think':
                # Action 步骤只有关键信息提取 没有理由
                # 初始化默认值
                reason = ""
                # 检查是否存在 root_cause_vote (通常是 Think 节点)
                # 注意：Prompt 中提到如果步骤是健全的，vote 可能是 null，所以要检查是否为 dict
                if 'root_cause_vote' in llm_res and isinstance(llm_res['root_cause_vote'], dict):
                    vote_data = llm_res['root_cause_vote']
                    # 1. 提取嵌套的 reason
                    reason = vote_data.get('reason', '')
                    # 2. 记录完整的 root_cause_vote 对象到节点属性中
                    self.graphs[self.curr_graph].nodes[step_id]['root_cause_vote'] = vote_data

            self.graphs[self.curr_graph].nodes[step_id]['llm_reason'] = reason

            # ==========================================
            # 策略 A: Think 节点 -> 依赖由 LLM 决定
            # ==========================================
            if step_type == 'Think':
                deps = llm_res.get('target_step_dependencies', [])
                for source_id in deps:
                    if self.graphs[self.curr_graph].has_node(source_id):
                        self.graphs[self.curr_graph].add_edge(source_id, step_id, type="informational")

            # ==========================================
            # 策略 B: Action 节点 -> 依赖由程序逻辑决定 (紧跟的上一个 Think)
            # ==========================================
            elif step_type == 'Action':
                # 1. 处理实体提取
                elements = llm_res.get('key_elements', [])
                if elements: 
                    for elem_name in elements:
                        entity_id = f"entity_{step_id}_{elem_name}"
                        self.graphs[self.curr_graph].add_node(entity_id, type="Entity", label=elem_name, color="#90EE90")
                        self.graphs[self.curr_graph].add_edge(step_id, entity_id, type="context_of")

                # 2. 构建依赖边：回溯寻找最近的 Think 节点
                found_think = False
                for prev_idx in range(i - 1, -1, -1):
                    prev_step = self.structured_steps[prev_idx]
                    if prev_step['type'] == 'Think':
                        prev_id = prev_step['step_id']
                        self.graphs[self.curr_graph].add_edge(prev_id, step_id, type="caused_by")
                        found_think = True
                        break 
        
        


    def export_graph_for_diagnosis(self, graph_id = None):
        """
        将图导出为大模型可读的文本格式，用于下一步诊断
        """
        if graph_id is None:
            graph_id = self.curr_graph
        
        output = "=== IDG Structure ===\n"

        # 1. 筛选出主流程节点 (Task, Action, Think)，排除 Entity 节点
        # 假设 step_id 是整数，我们要按顺序打印
        main_nodes = [n for n in self.graphs[graph_id].nodes if self.graphs[graph_id].nodes[n].get('type') != 'Entity']
        main_nodes.sort() # 确保按步骤顺序输出


        for node in main_nodes:
            data = self.graphs[graph_id].nodes[node]
            # 格式化输出
            type_str = data.get('type', 'Task')

            content_sample = data.get('action_text', data.get('content', ''))
            obs_sample = data.get('observation_text', '')

            output += f"Node [{node}] ({type_str}): \"{content_sample}\"\n"
            if obs_sample:
                output += f"          OBSERVATION: {obs_sample}\n"

            # 2. [关键修改] 查找连接的 Entity 子节点作为 Context Elements
            # 逻辑：查找从当前 node 出发的所有邻居，如果类型是 Entity，提取其 label
            context_elements = []
            # graph.successors(node) 获取所有该节点指向的节点
            for neighbor in self.graphs[graph_id].successors(node):
                neighbor_data = self.graphs[graph_id].nodes[neighbor]
                if neighbor_data.get('type') == 'Entity':
                    # 获取实体名称，默认为 ID
                    label = neighbor_data.get('label', str(neighbor))
                    context_elements.append(label)
            
            if context_elements:
                output += f"  - Context Elements: {context_elements}\n"
            
            # 3. 处理依赖边 (入边)
            # 逻辑：只获取类型为 'informational' 或 'caused_by' 的入边，排除其他可能的边
            dependencies = []
            for u, v, attr in self.graphs[graph_id].in_edges(node, data=True):
                # 兼容之前的代码，边类型可能是 'caused_by' 或 'informational'
                if attr.get('type') in ['caused_by', 'informational']:
                    dependencies.append(u)
            
            if dependencies:
                output += f"  <- Dependent on: {dependencies}\n"
            # Node 0 是根节点，不需要依赖；其他节点如果没有依赖则报警
            elif node > 0:
                output += f"  <- [WARNING] NO DEPENDENCIES (ISOLATED)\n"
            
            output += "\n"
        return output
    
    def visualize_graph(self):
        """
        使用 PyVis 生成交互式 HTML 图谱
        修正版：使用内联资源解决进度条卡 0% 的问题
        """
        output_file=f"/data/G-Memory/GMemory-main/tasks/refine/out/agent_trace_{self.test_index}.html"

        from pyvis.network import Network
        import os

        # [关键修改] cdn_resources='in_line': 
        # 强制将 JS 库写入 HTML，防止因网络问题导致加载失败
        net = Network(
            height="800px", 
            width="100%", 
            directed=True, 
            select_menu=True, 
            filter_menu=True,
            cdn_resources='in_line' 
        )
        
        # 检查图是否为空
        if self.graphs[self.curr_graph].number_of_nodes() == 0:
            print("[Warning] Graph is empty! Nothing to visualize.")
            return

        # 将 NetworkX 图转换为 PyVis 图
        for node_id in self.graphs[self.curr_graph].nodes:
            node_attrs = self.graphs[self.curr_graph].nodes[node_id]
            node_type = node_attrs.get('type', 'Unknown')
            
            # 样式定义
            label = str(node_id) 
            title = "" 
            color = "#DDDDDD" 
            size = 10
            shape = "dot"
            
            # 根据类型定义样式
            if node_type == "Task":
                color = "#FF6B6B" # 红色
                size = 25
                title = f"Task: {node_attrs.get('content', '')}"
                label = f"Task"
                shape = "star"
                
            elif node_type == "Action":
                color = "#4D96FF" # 蓝色
                size = 20
                # Tooltip 显示详细信息
                title = (f"<b>Action:</b> {node_attrs.get('action_text', '')}<br>"
                         f"<b>Obs:</b> {node_attrs.get('observation_text', '')}<br>"
                         f"<b>Reason:</b> {node_attrs.get('llm_reason', '')}")
                label = f"Act {node_id}"
                
            elif node_type == "Think":
                color = "#FFD93D" # 黄色
                size = 15
                shape = "diamond"
                title = f"Think: {node_attrs.get('action_text', '')}"
                label = f"Think {node_id}"
                
            elif node_type == "Entity":
                color = "#6BCB77" # 绿色
                size = 10
                shape = "square"
                title = f"Context Object: {node_attrs.get('label', '')}"
                # 截断过长的标签
                raw_label = node_attrs.get('label', '')
                label = raw_label[:15] + "..." if len(raw_label) > 15 else raw_label
            
            # 添加节点
            # group 参数用于 PyVis 的图例和筛选功能
            net.add_node(node_id, label=label, title=title, color=color, size=size, shape=shape, group=node_type)

        # 添加边
        for u, v, attrs in self.graphs[self.curr_graph].edges(data=True):
            edge_type = attrs.get('type', 'unknown')
            color = "#AAAAAA"
            width = 1
            dashes = False
            title = attrs.get('reason', '')
            
            if edge_type == "informational" or edge_type == "caused_by":
                color = "#333333" # 深色实线
                width = 2
                arrows = "to"
            elif edge_type == "context_of":
                color = "#6BCB77" # 绿色虚线
                width = 1
                dashes = True 
                arrows = "" # 这种边通常不需要箭头，或者指向实体
            
            net.add_edge(u, v, title=title, color=color, width=width, dashes=dashes, arrows=arrows)

        # 调整物理引擎参数，防止节点乱飞，让它们更稳定
        net.barnes_hut(gravity=-2000, central_gravity=0.3, spring_length=150, spring_strength=0.05, damping=0.09)
        
        # 保存并打印路径
        try:
            # 确保文件名以 .html 结尾
            if not output_file.endswith('.html'):
                output_file += '.html'
            
            net.save_graph(output_file)
            print(f"Visualization successfully saved to: {os.path.abspath(output_file)}")
            print("Please open this file in your browser.")
        except Exception as e:
            print(f"Error saving visualization: {e}")
        


def get_trajectory_from_logs(input_path = "/data/G-Memory/GMemory-main/.db/Qwen3-8B_test_vision_no_GT_new/alfworld/macnet/g-memory/log.json"):
    """
    从解析出的日志文件里面获取轨迹数据
    """
    def fun_task_description(init_prompt):
        pattern = re.compile(
        r"## Your Turn: Take Action!\nUse the above examples and insights as a foundation, and now work on the following task:\n(.*?)\n\n>\n\n", 
        re.DOTALL
        )
        match = pattern.search(init_prompt)
        if match:
            task_desc = match.group(1).strip()
        else:
            task_desc = "Task description not found."
        return task_desc
        

    
    with open(input_path, 'r') as f:
        log_data = json.load(f)
    raw_logs_all = []
    for entry in log_data:
        raw_logs = []
        trials = entry.get('trials', [])
        init_prompt = entry.get('init_user_pxrompt', '')
        task_description = fun_task_description(init_prompt)
        raw_logs.append(task_description)
        task_result = entry.get('task_result', )
        if "You failed the task" not in task_result:
            continue
        for trial in trials:
            trial_id = trial.get('trial_id')
            step_result = trial.get('step_result')
            raw_logs.append(step_result)
        raw_logs_all.append(raw_logs)
    return raw_logs_all


if __name__== "__main__":
    # --- 使用你的数据进行测试 ---

    # 实例化并运行
    raw_logs = get_trajectory_from_logs()
    builder = ParallelGraphBuilder(model_name="Qwen3-8B", base_url=OPENAI_API_BASE, api_key=OPENAI_API_KEY, max_workers = 1)
    builder.get_log(raw_logs, target_test_index=0)

    builder.build_graph_parallel()  # Qwen3-8B
    # 打印结果供下一步使用
    graph_text = builder.export_graph_for_diagnosis()
    print(graph_text)
    builder.visualize_graph()
    
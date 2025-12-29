
# """

# prompt = f"""
# Role: You are a rigorous Agent Logic Auditor.
# Task: Analyze the **TARGET NODE** and identify its dependencies based ONLY on the provided **HISTORY**.

# **HISTORY (Strictly Chronological):**
# {history_str}

# **TARGET NODE (To Analyze):**
# {current_str}
        
# **INSTRUCTIONS:**
# 1. **Dependency Analysis:** List the Node IDs from History that provided necessary information or logical basis for the Target Node.
#     - **VALID ID CHECK:** You can ONLY cite Node IDs that explicitly appear in the **HISTORY** above. Valid IDs are: [{valid_ids_str}].
#    - **CRITICAL:** If the Target Node claims a fact (e.g., "I saw the soapbar") or makes a decision that is NOT supported by any node in History, you **MUST set the `dependencies` field in the JSON output to `[]` (an empty list)**. Do NOT invent dependencies. This explicitly indicates a Hallucination.
#    - Node 0 (Task) is usually a dependency for the first plan.

# 2. **Key Elements Extraction (Environment Skeleton):**
#    - **GOAL:** Extract environment elements ONLY for the **TARGET MOMENT**.
#    - **SCOPE:** **STRICTLY RESTRICTED** to the Target Node's content.
#    - **SOURCE:** Extract entities **STRICTLY from the "OBSERVATION" text** of the Target Node.
#    - **STRATEGY:** **EXHAUSTIVE LISTING**. Do NOT filter out items just because the agent didn't interact with them.
#    - **FORBIDDEN:** **Do NOT** include objects or locations mentioned in `HISTORY` unless they appear again in the Target Node. We do NOT want a cumulative state; we want a snapshot of *this specific step*.
#    - **NAVIGATION/EMPTY:** If the Observation says "Nothing happens" or "Empty", reflect that state (e.g., ["countertop 2 (Empty)"]).

# **OUTPUT FORMAT (JSON Only):**
# {{
#   "dependencies": [0],  // Example: [0] or [1, 3]. MUST be a subset of [{valid_ids_str}]. Output `[]` if no dependency.
#   "reason": "Explanation...",
#   "key_elements": ["obj1", "loc1"]
# }}
# """


def return_action_step(current_str):
    """
    Action 动作不做步骤依赖 做重要信息提取
    """
    prompt = f"""
Role: You are an Environment Context Modeler.
Task: Extract a precise and EXHAUSTIVE snapshot of the environment based on the **TARGET STEP**.

**TARGET Step (To Analyze):**
{current_str}

**INSTRUCTIONS:**
1. **Goal:** Build a "Context Skeleton" for the environment state at this exact moment.
2. **Source:** Extract entities **STRICTLY from the "OBSERVATION" text**.
   - Use the "ACTION" text only to understand context (e.g., if Obs says "Nothing happens", use Action to identify the location).
3. **Strategy: EXHAUSTIVE LISTING (No Filtering).**
   - List **ALL** physical objects, locations, and containers mentioned.
   - Do NOT filter out "irrelevant" background items (e.g., candles, spraybottles). We need the full noise level.
   - **State Marking:** - If a container is open/empty, mark it (e.g., "countertop 2 (Empty)").
     - If holding an item, mark it (e.g., "User (holding soapbar)").

**OUTPUT FORMAT (JSON Only):**
{{
  "key_elements": ["loc_name", "obj1", "obj2", "obj3"] // List EVERYTHING visible in Observation
}}
"""
    return prompt


def return_plan_step(task_description: str, history_str: str, current_str: str, valid_ids_str: str, cur_step_id: int):
    """
    """
#     prompt = f"""
# Role: You are a Cognitive Scientist analyzing an AI Agent's thought process using **Situation Awareness (SA) Theory**.
# Task: Analyze the **TARGET NODE** (Thinking/Planning) to identify Information Dependencies and pinpoint the **Earliest Cognitive Breakpoint**.

# **HISTORY (Strictly Chronological):**
# {history_str}

# **TARGET NODE (To Analyze):**
# {current_str}

# **INSTRUCTIONS:**

# **Part 1: Information Dependency (The "Why" of this thought)**
# - List Node IDs from History that provided the **logical basis** or **factual evidence** for the Target Node.
# - **HALLUCINATION CHECK:** If the Target Node claims a fact (e.g., "I saw the key") NOT supported by any node in History, return `[]`. This is a critical check.

# **Part 2: Cognitive Weakness Diagnosis (Root Cause Analysis)**
# - **Objective:** Find the **"Primal Error"**. Blame the **first moment** the agent's cognition detached from reality or logic, leading to the final failure.
# - **Theoretical Framework (Endsley’s SA Levels):**
#   Review the `HISTORY` leading up to the Target Node. Does the Target Node (or a recent predecessor) exhibit failure at any of these levels?

#   1.  **SA Level 1 Failure: Perceptual Blindness**
#       - *Definition:* The data was in the `OBSERVATION`, but the agent ignored it.
  
#   2.  **SA Level 2 Failure: Interpretation Error**
#       - *Definition:* The agent saw the data but assigned the wrong meaning (Cognitive Mismatch).
  
#   3.  **SA Level 3 Failure: Projection Deficit**
#       - *Definition:* Correct understanding, but failed to simulate future consequences (Inefficient Planning).
  
#   4.  **Metacognitive Failure: Monitoring Deficit**
#       - *Definition:* The agent is stuck or drifting but fails to trigger a "Self-Correction" or "Stop" signal.

# **Voting Mechanism:**
# - If the **TARGET NODE** itself represents the *start* or *continuation* of one of these failures, cast a vote for the **Earliest Step** where this logic chain began.
# - If the thought is sound and grounded, vote `null`.

# **OUTPUT FORMAT (JSON Only):**
# {{
#   "dependencies": [0, 2], 
#   "root_cause_vote": {{
#       "suspected_step_id": 3,  // The ID of the FIRST PAST step that is the "Root Cause" of the issue. `null` if no issue.
#       "weakness_type": "SA Level 1 Failure (Perceptual Blindness)", 
#       "reason": "At Step 3, the agent failed to perceive the 'soapbar' listed in the observation context, leading to the current redundant search."
#   }}
# }}
# """
    


    prompt = f"""
Role: You are a Cognitive Scientist analyzing an AI Agent's thought process using **Situation Awareness (SA) Theory**.
Task: Analyze the **TARGET NODE** (Thinking/Planning) within the context of its **HISTORY** to identify Information Dependencies and pinpoint the **Earliest Cognitive Breakpoint** (Root Error).

**HISTORY (Strictly Chronological):**
{history_str}

**TARGET NODE (To Analyze):**
{current_str}


**INSTRUCTIONS:**

**Part 1: Information Dependency (The "Why" of this thought)**
- List Node IDs from History that provided the **logical basis** or **factual evidence** for the Target Node.
- **VALID ID CHECK:** You can ONLY cite Node IDs that explicitly appear in the **HISTORY** above. Valid IDs are: [{valid_ids_str}].
- **HALLUCINATION CHECK:** If the Target Node claims a fact (e.g., "I saw the key") NOT supported by any node in History, you **MUST set the `target_node_dependencies` field in the JSON output to `[]` (an empty list)**. Do NOT invent dependencies. This explicitly indicates a Hallucination.


**Part 2: Cognitive Weakness Diagnosis (Root Cause Analysis)**
- **Scope:** Treat `HISTORY` and `TARGET NODE` as a single continuous timeline of steps.
- **Diagnosis Logic:**
  1. **Assess the Target Node:** Does this thought represent a failure in cognition (e.g., hallucination, loop, missing info, wrong interpretation)?
  2. **Trace Back (The "Primal Error"):**
     - If the Target Node is **Good/Sound**: Vote `null`.
     - If the Target Node is **Bad/Flawed**: You must find the **ORIGIN** of this flaw.
       - Is the Target Node the *first time* this error appears? -> Vote for **Target Node ID**.
       - Is the Target Node just *repeating* or *suffering from* an error that started earlier (e.g., a loop started 5 steps ago)? -> Vote for that **Historical Step ID**.
       
- **Theoretical Framework (Endsley’s SA Levels):**
  Use these criteria to identify the error type:
  1.  **SA Level 1 Failure: Perceptual Blindness** (Data present in Obs but ignored).
  2.  **SA Level 2 Failure: Interpretation Error** (Data seen but misunderstood/mismatched).
  3.  **SA Level 3 Failure: Projection Deficit** (Future simulation failed, causing loops/inefficiency).
  4.  **Metacognitive Failure: Monitoring Deficit** (Drifting without self-correction).

**OUTPUT FORMAT (JSON Only):**
{{
  "target_node_dependencies": [X], // Example: [0] or [1, 3]. 'X' MUST be a subset of [{valid_ids_str}]. Output `[]` if no dependency.
  "root_cause_vote": {{
      "suspected_step_id": Y,  // 'Y' is the ID of the **Earliest Step** (History OR Target) where the error logic originated. `null` if the Target Node is sound.
      "weakness_type": "SA Level * Failure (***)", // Select one error type from **Theoretical Framework (Endsley’s SA Levels):**
      "reason": "At Step Y, the agent ...."
  }}
}}
"""
    


    prompt = f"""
Role: You are an expert AI Analyst evaluating an Agent's reasoning capabilities.
Task: Perform two independent analyses on the provided agent trajectory:
1. Identify **Information Dependencies** for step {cur_step_id} (current step).
2. Detect the **Root Cause** of any errors or inefficiencies in the entire trajectory.

**All steps (Strictly Chronological):**
{history_str}

{current_str}

**INSTRUCTIONS:**

**Part 1: Information Dependency (Focus on STEP {cur_step_id} only)**
- Analyze the **STEP {cur_step_id}**. Identify which steps from the **PREVIOUS STEPS BEFORE STEP {cur_step_id}** provided the **logical basis** or **factual evidence** for this specific thought.
- **VALID ID CHECK:** You can ONLY cite step IDs that explicitly appear in the **PREVIOUS STEPS BEFORE STEP {cur_step_id}**. Valid IDs are: [{valid_ids_str}].
- **HALLUCINATION CHECK:** If the STEP {cur_step_id} claims a fact NOT supported by any node in History, you **MUST set the `target_step_dependencies` field to `[]`**. Do NOT invent dependencies.

**Part 2: Cognitive Weakness Diagnosis (Global Trajectory Analysis)**
- **Scope:** Treat the entire sequence as a single timeline.
- **Diagnosis Logic:**
  1. **Scan the All Steps of the Full Trajectory:** Look for the **most significant error** or distinct area for improvement across ALL steps.
  2. **Identify the Culprit Step:**
     - **Critical Impact:** Identify the specific step containing the error or weakness that had the **greatest negative impact** on the final failure.
     - **Refinement Value:** Select the step where, if it were refined or corrected, the Agent would have the **highest probability** of successfully completing the task.
     - **Chronological Priority:** If multiple steps appear to have equal impact on the failure, select the **earlier, more fundamental** step (the root cause) rather than later derivative errors.
  3. **Null Condition:** Vote `null` in **suspected_step_id** only if the **ENTIRE** trajectory is sound, logical, and efficient.

**RESPONSE FORMAT:**
  - **Output strictly raw JSON.**

**OUTPUT FORMAT (JSON Only):**
{{
  "target_step_dependencies": [X], // Example: [0] or [1, 3]. Dependencies for the STEP {cur_step_id}. Output `[]` if none.
  "root_cause_vote": {{
      "suspected_step_id": Y,  // 'Y' can be ANY ID from All steps. Select the specific step where the error actually resides. `null` if the WHOLE trajectory is perfect.
      "weakness_type": "Category string", // A concise summary of the diagnosed weakness. E.g., "Hallucination", "Logic Error", "Loop", "Inefficiency". 
      "reason": "At Step Y, the agent ...."
  }}
}}
"""
    
    return prompt

# def get_naive_prompt(task_desc, history_str):
#     prompt = f"""
# Role: You are an expert Agent Trajectory Auditor.
# Task: Analyze the provided **PARTIAL TRAJECTORY (Sliding Window)** of an Agent attempting a task. The Agent ultimately FAILED the task.

# **Context:**
# - This is a segment of a longer execution trace.
# - Your goal is to identify the **Top 3 Most Problematic Steps** WITHIN THIS SPECIFIC WINDOW that contributed to the failure or inefficiency.

# **Task Description:** 
# {task_desc}

# **Trajectory Segment (Window):**
# {history_str}


# **INSTRUCTIONS:**

# **1. How to Evaluate THINK Steps:**
# - **Framework:** Analyze based on the **Final Goal** ({task_desc}) and the **Context** (History & Observations).
# - **Criteria for Error/Weakness:**
#   - **Critical Impact:** Identify the specific thought that led the agent down a dead-end path.
#   - **Refinement Value:** Select the thought where, if it were corrected, the Agent would have the **highest probability** of success.
  
# **2. Select:** Identify the Top 3 most critical error steps within this window. If there are fewer than 3 errors, list only the real errors.

# **3. Format:** Output strictly in JSON format.

# **OUTPUT FORMAT (JSON Only):**
# {{
#   "problematic_steps": [
#     {{
#       "reason": "Brief Reason for why this step A is problematic.", 
#       "step_id": A
#     }},
#     {{
#       "reason": "Brief Reason for why this step B is problematic.",
#       "step_id": B,
#     }},
#     {{
#       "reason": "Brief Reason for why this step C is problematic.",
#       "step_id": C
#     }}
#   ]
# }}
# """
#     return prompt

# def get_naive_prompt(task_desc, history_str):
# #     prompt = f"""
# # Analyze the Agent's trajectory. 
# # Focus strictly on the consistency between the **Agent's Inventory (what it is holding)** and its **Thoughts**.

# # **Trajectory:**
# # {history_str}

# # **Rules for Error Detection:**
# # 1. **Inventory Awareness:** Did the agent act as if it was searching for an item it was ALREADY holding?
# # 2. **Goal Consistency:** Did the agent abandon a correct plan (e.g., "cool the tomato") because it misinterpreted an observation (e.g., "fridge is empty of tomatoes" implies failure to find, rather than readiness to place)?

# # **Requirement:**
# # Identify the step where the Agent lost track of its own inventory or status.

# # **OUTPUT FORMAT (JSON Only):**
# # {{
# #   "problematic_steps": [
# #     {{
# #       "step_id": int,
# #       "reason": "Concise explanation focusing on Inventory/State contradiction."
# #     }}
# #   ]
# # }}
# # """
    
# #     prompt = f"""
# # I will provide you with an Agent's generation trajectory. This Agent trajectory is part of a complete trajectory. The Agent failed to successfully complete the task objective within the specified step limit.

# # **Constraint: You must assume that the task is theoretically solvable and the environment contains all necessary objects. Do not attribute the failure to missing items or environment issues.**

# # Please pinpoint the step where the model first deviated, or the step requiring the most critical correction to enable the Agent to complete the task, along with your reasoning.

# # Additionally, provide a specific guidance or instruction to help other Agents successfully complete the task if they encounter the same issue you identified.

# # Trajectory:

# # {history_str}

# # Please strictly output the result in JSON format with the following fields:
# # {{
# #   "step_id": [The ID of the step],
# #   "reasoning": "[Your reasoning]",
# #   "guidance": "[The guidance for other Agents]"
# # }}"""

# #     prompt = f"""
# # You are an expert analyst for the "Alfworld" household environment. Your goal is to diagnose why an Agent failed to complete a task.

# # ### Environment & Action Rules (Reference)
# # The Agent operates in a simulated household. It must strictly follow specific syntactic structures. Use these rules to judge if the Agent's actions were valid.

# # **Allowed Actions & Syntax:**
# # NOTE:
# # - You must strictly follow the syntactic structure of the steps (where 'a' and 'b' are variables): 
# #   - 1. take a from b. 
# #   - 2. go to a. 
# #   - 3. open a.
# #   - 4. put a in/on b. 
# #       - ❗ ABSOLUTELY DO NOT use "in" or "on" alone. You MUST write "in/on" together as a combined phrase.
# #       - Example ✅: put apple in/on garbage(or any other objs).
# #       - Example ❌: put apple in garbage(or any other objs). (WRONG)
# #       - Example ❌: put apple on garbage(or any other objs). (WRONG)
# #   - 5. clean a with b. 
# #   - 6. heat a with b. 
# #   - 7. cool a with b. 
# #   - 8. use a. 
# #   - 9. think: xxx

# # **Any output by the Agent not strictly matching these formats is considered an error.**

# # ### Analysis Task
# # I will provide you with the **complete** generation trajectory of an Agent. The Agent failed to successfully complete the task objective within the specified step limit.

# # **Constraint: You must assume that the task is theoretically solvable and the environment contains all necessary objects. Do not attribute the failure to missing items or environment issues. Focus your analysis solely on the Agent's Weakness in reasoning or planning.**

# # Please pinpoint the step where the model first deviated, or the step requiring the most critical correction to enable the Agent to complete the task, along with your reasoning. Since this is the full history, prioritize identifying the **earliest root cause** of the failure.

# # Additionally, provide a specific guidance or instruction to help other Agents successfully complete the task if they encounter the same issue you identified.

# # Full Trajectory:
# # {history_str}

# # Please strictly output the result in JSON format with the following fields:
# # {{
# #   "step_id": [The ID of the step],   // Identify the earliest root cause step
# #   "reasoning": "[Your reasoning]",  // Explanation of why this step is critical.
# #   "guidance": "[The guidance for other Agents]" // Provide clear, actionable advice to avoid the same pitfall.
# # }}"""
    
#     prompt = f"""
# ### Role & Context
# You are an expert analyst for the "Alfworld" household environment. An Agent has failed to complete a task within the limited steps.

# ### Environment & Action Rules (Reference)
# The Agent operates in a simulated household. It must strictly follow specific syntactic structures. Use these rules to judge if the Agent's actions were valid.

# **Allowed Actions & Syntax:**
# NOTE:
# - You must strictly follow the syntactic structure of the steps (where 'a' and 'b' are variables): 
#   - 1. take a from b. 
#   - 2. go to a. 
#   - 3. open a.
#   - 4. put a in/on b. 
#       - ❗ ABSOLUTELY DO NOT use "in" or "on" alone. You MUST write "in/on" together as a combined phrase.
#       - Example ✅: put apple in/on garbage(or any other objs).
#       - Example ❌: put apple in garbage(or any other objs). (WRONG)
#       - Example ❌: put apple on garbage(or any other objs). (WRONG)
#   - 5. clean a with b. 
#   - 6. heat a with b. 
#   - 7. cool a with b. 
#   - 8. use a. 
#   - 9. think: xxx

# **Constraint:** The task is theoretically solvable. Do not blame the environment.

# ### Input Data
# Full Trajectory:
# {history_str}

# ### Analysis Task
# 1. Identify the **earliest root cause** step where the Agent's reasoning or planning failed introducing the final failure of the task.
# 2. Return a correction suggestion strategy containing: describing the specific situation where the agent made a wrong decision; providing a concise, more efficient new plan of action that accounts for the agent's mistake with reference to specific actions that you suggest to take within the limited steps. (Prioritize efficiency. The task must be completed within strict step limits.)


# ### Output Specification
# Strictly output the result in JSON format:

# {{
#   "step_id": [The ID of the step],   // Identify the earliest root cause step
#   "reasoning": "[Brief analysis of the error.]",
#   "correction_suggestion": "[A correction suggestion strategy]"
# }}
# """


#     return prompt

def get_naive_prompt(task_desc, history_str):
#     prompt = f"""
# Analyze the Agent's trajectory. 
# Focus strictly on the consistency between the **Agent's Inventory (what it is holding)** and its **Thoughts**.

# **Trajectory:**
# {history_str}

# **Rules for Error Detection:**
# 1. **Inventory Awareness:** Did the agent act as if it was searching for an item it was ALREADY holding?
# 2. **Goal Consistency:** Did the agent abandon a correct plan (e.g., "cool the tomato") because it misinterpreted an observation (e.g., "fridge is empty of tomatoes" implies failure to find, rather than readiness to place)?

# **Requirement:**
# Identify the step where the Agent lost track of its own inventory or status.

# **OUTPUT FORMAT (JSON Only):**
# {{
#   "problematic_steps": [
#     {{
#       "step_id": int,
#       "reason": "Concise explanation focusing on Inventory/State contradiction."
#     }}
#   ]
# }}
# """
    
#     prompt = f"""
# I will provide you with an Agent's generation trajectory. This Agent trajectory is part of a complete trajectory. The Agent failed to successfully complete the task objective within the specified step limit.

# **Constraint: You must assume that the task is theoretically solvable and the environment contains all necessary objects. Do not attribute the failure to missing items or environment issues.**

# Please pinpoint the step where the model first deviated, or the step requiring the most critical correction to enable the Agent to complete the task, along with your reasoning.

# Additionally, provide a specific guidance or instruction to help other Agents successfully complete the task if they encounter the same issue you identified.

# Trajectory:

# {history_str}

# Please strictly output the result in JSON format with the following fields:
# {{
#   "step_id": [The ID of the step],
#   "reasoning": "[Your reasoning]",
#   "guidance": "[The guidance for other Agents]"
# }}"""

#     prompt = f"""
# You are an expert analyst for the "Alfworld" household environment. Your goal is to diagnose why an Agent failed to complete a task.

# ### Environment & Action Rules (Reference)
# The Agent operates in a simulated household. It must strictly follow specific syntactic structures. Use these rules to judge if the Agent's actions were valid.

# **Allowed Actions & Syntax:**
# NOTE:
# - You must strictly follow the syntactic structure of the steps (where 'a' and 'b' are variables): 
#   - 1. take a from b. 
#   - 2. go to a. 
#   - 3. open a.
#   - 4. put a in/on b. 
#       - ❗ ABSOLUTELY DO NOT use "in" or "on" alone. You MUST write "in/on" together as a combined phrase.
#       - Example ✅: put apple in/on garbage(or any other objs).
#       - Example ❌: put apple in garbage(or any other objs). (WRONG)
#       - Example ❌: put apple on garbage(or any other objs). (WRONG)
#   - 5. clean a with b. 
#   - 6. heat a with b. 
#   - 7. cool a with b. 
#   - 8. use a. 
#   - 9. think: xxx

# **Any output by the Agent not strictly matching these formats is considered an error.**

# ### Analysis Task
# I will provide you with the **complete** generation trajectory of an Agent. The Agent failed to successfully complete the task objective within the specified step limit.

# **Constraint: You must assume that the task is theoretically solvable and the environment contains all necessary objects. Do not attribute the failure to missing items or environment issues. Focus your analysis solely on the Agent's Weakness in reasoning or planning.**

# Please pinpoint the step where the model first deviated, or the step requiring the most critical correction to enable the Agent to complete the task, along with your reasoning. Since this is the full history, prioritize identifying the **earliest root cause** of the failure.

# Additionally, provide a specific guidance or instruction to help other Agents successfully complete the task if they encounter the same issue you identified.

# Full Trajectory:
# {history_str}

# Please strictly output the result in JSON format with the following fields:
# {{
#   "step_id": [The ID of the step],   // Identify the earliest root cause step
#   "reasoning": "[Your reasoning]",  // Explanation of why this step is critical.
#   "guidance": "[The guidance for other Agents]" // Provide clear, actionable advice to avoid the same pitfall.
# }}"""
    
    prompt_spe = f"""
### Role & Context
You are an expert analyst for the "Alfworld" household environment. An Agent has failed to complete a task within the limited steps.

### Environment & Action Rules (Reference)
The Agent operates in a simulated household. It must strictly follow specific syntactic structures. Use these rules to judge if the Agent's actions were valid.

**Allowed Actions & Syntax:**
NOTE:
- You must strictly follow the syntactic structure of the steps (where 'a' and 'b' are variables): 
  - 1. take a from b. 
  - 2. go to a. 
  - 3. open a.
  - 4. put a in/on b. 
      - ❗ ABSOLUTELY DO NOT use "in" or "on" alone. You MUST write "in/on" together as a combined phrase.
      - Example ✅: put apple in/on garbage(or any other objs).
      - Example ❌: put apple in garbage(or any other objs). (WRONG)
      - Example ❌: put apple on garbage(or any other objs). (WRONG)
  - 5. clean a with b. 
  - 6. heat a with b. 
  - 7. cool a with b. 
  - 8. use a. 
  - 9. think: xxx

**Constraint:** The task is theoretically solvable. Do not blame the environment.

### Input Data
Full Trajectory:
{history_str}

### Analysis Task
1. Identify the **earliest root cause** step where the Agent's reasoning or planning failed introducing the final failure of the task.
2. Return a correction suggestion strategy containing: describing the specific situation where the agent made a wrong decision; providing a concise, more efficient new plan of action that accounts for the agent's mistake with reference to specific actions that you suggest to take within the limited steps. (Prioritize efficiency. The task must be completed within strict step limits.)


### Output Specification
Strictly output the result in JSON format:

{{
  "step_id": [The ID of the step],   // Identify the earliest root cause step
  "reasoning": "[Brief analysis of the error.]",
  "correction_suggestion": "[A correction suggestion strategy]"
}}
"""
    
    prompt_ger = f"""
### Role & Context
You are an expert analyst for the "Alfworld" household environment. An Agent has failed to complete a task within the limited steps.

### Environment & Action Rules (Reference)
The Agent operates in a simulated household. It must strictly follow specific syntactic structures. Use these rules to judge if the Agent's actions were valid.

**Allowed Actions & Syntax:**
NOTE:
- You must strictly follow the syntactic structure of the steps (where 'a' and 'b' are variables): 
  - 1. take a from b. 
  - 2. go to a. 
  - 3. open a.
  - 4. put a in/on b. 
      - ❗ ABSOLUTELY DO NOT use "in" or "on" alone. You MUST write "in/on" together as a combined phrase.
      - Example ✅: put apple in/on garbage(or any other objs).
      - Example ❌: put apple in garbage(or any other objs). (WRONG)
      - Example ❌: put apple on garbage(or any other objs). (WRONG)
  - 5. clean a with b. 
  - 6. heat a with b. 
  - 7. cool a with b. 
  - 8. use a. 
  - 9. think: xxx

**Constraint:** The task is theoretically solvable. Do not blame the environment.

### Input Data
Full Trajectory:
{history_str}

### Analysis Task
1. Identify the **earliest root cause** step where the Agent's reasoning or planning failed introducing the final failure of the task.
2. To stop other Agents from making similar errors, return a **generalizable** correction suggestion strategy. **Formulate a general rule or heuristic** derived from this failure. This strategy should:
   - Describe the abstract situation or pattern where the wrong decision was made.
   - Provide a guideline applicable to **the same or any similar situations** in the Alfworld environment.
   - **(Prioritize efficiency. The proposed heuristic must aim to optimize decision logic to complete tasks within strict step limits.)**

### Output Specification
Strictly output the result in JSON format:

{{
  "step_id": [The ID of the step],   // Identify the earliest root cause step
  "reasoning": "[Brief analysis of the error.]",
  "correction_suggestion": "[A generalizable strategy or heuristic applicable to the same or any similar problems, prioritizing efficiency]"
}}
"""

    return prompt_spe, prompt_ger

# query: str = f"""You will be given the history of a past experience in which you were placed in an environment and given a task to complete. You were unsuccessful in completing the task. Do not summarize your environment, but rather think about the strategy and path you took to attempt to complete the task. Devise a concise, new plan of action that accounts for your mistake with reference to specific actions that you should have taken. For example, if you tried A and B but forgot C, then devise a plan to achieve C with environment-specific actions. You will need this later when you are solving the same task. Give your plan after "Plan". Here are two examples:

# {FEW_SHOT_EXAMPLES}

# {scenario}"""
def get_model_type(model_name: str) -> str:
    
    valid_model_types: list[str] = [
        'gpt-4o-mini', 
        'gpt-5-mini', 
        'qwen2.5-7b', 
        'qwen2.5-14b',
        'qwen2.5-32b', 
        'qwen2.5-72b',
        'intern', 
        'deepseek-v3',
        'Qwen3-235B-A22B-Instruct-2507',
        "Qwen3-8B"
    ]

    for model_type in valid_model_types:
        if model_type.lower() in model_name.lower():
            return model_type
    
    return 'unknown'
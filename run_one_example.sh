cd /data/G-Memory/GMemory-main

nohup python tasks/refine/run_one_example.py \
  --task alfworld \
  --reasoning io \
  --mas_memory g-memory \
  --mas_type macnet \
  --max_trials 30 \
  --Prefix _test2 \
  --reflection_type general \
  --is_no_reflection \
  > ./sample_alfworld_test.out 2>&1 &

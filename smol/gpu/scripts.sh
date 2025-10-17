set -x
# initial nsys hello world
# python triton_elementwise_mul.py --size 20000000 --dtype float32 --iters 200 --warmup 50 --block 1024

# nsys profile -o nsys_triton_mul --stats=true python triton_elementwise_mul.py --size 200000000 --iters 100 --warmup 50

# # attn
# python attn.py --batch 4 --seq 1024 --heads 8 --dim 64 --dtype bf16 --iters 200 --warmup 50

# nsys profile -o nsys_attn --stats=true python attn.py --batch 4 --seq 1024 --heads 8 --dim 64 --dtype bf16 --iters 200 --warmup 50

# NCU - we want to run JUST the kernel we care about, ncu will handle the warm-ups, repeats etc
# ncu -f -o torch_sdpa_prefill python attn_kernel.py --batch_size 1 --q_seq_len 512 --k_seq_len 512 --num_heads 32 --head_dim 128 --kernel sdpa --dtype fp32
# ncu -f -o torch_sdpa_decode python attn_kernel.py --batch_size 128 --q_seq_len 1 --k_seq_len 512 --num_heads 32 --head_dim 128 --kernel sdpa --dtype fp32

# ncu -f -o custom_sdpa_prefill python attn_kernel.py --batch_size 1 --q_seq_len 512 --k_seq_len 512 --num_heads 32 --head_dim 128 --kernel custom_sdpa --dtype fp32
# ncu -f -o custom_sdpa_decode python attn_kernel.py --batch_size 128 --q_seq_len 1 --k_seq_len 512 --num_heads 32 --head_dim 128 --kernel custom_sdpa --dtype fp32

# ncu -f -o sdpa_kernel_prefill python attn_kernel.py --batch_size 1 --q_seq_len 512 --k_seq_len 512 --num_heads 32 --head_dim 128 --kernel kernel_sdpa_fwd --dtype fp32
# ncu -f -o sdpa_kernel_decode python attn_kernel.py --batch_size 128 --q_seq_len 1 --k_seq_len 512 --num_heads 32 --head_dim 128 --kernel kernel_sdpa_fwd --dtype fp32

# nsys
# nsys profile -f true -o torch_sdpa_prefill --stats=true python attn_kernel.py --batch_size 1 --q_seq_len 512 --k_seq_len 512 --num_heads 32 --head_dim 128 --kernel sdpa --dtype fp32 --warmup 2
# nsys profile -f true -o torch_sdpa_decode --stats=true python attn_kernel.py --batch_size 128 --q_seq_len 1 --k_seq_len 512 --num_heads 32 --head_dim 128 --kernel sdpa --dtype fp32 --warmup 2

# nsys profile -f true -o custom_sdpa_prefill --stats=true python attn_kernel.py --batch_size 1 --q_seq_len 512 --k_seq_len 512 --num_heads 32 --head_dim 128 --kernel custom_sdpa --dtype fp32 --warmup 2
# nsys profile -f true -o custom_sdpa_decode --stats=true python attn_kernel.py --batch_size 128 --q_seq_len 1 --k_seq_len 512 --num_heads 32 --head_dim 128 --kernel custom_sdpa --dtype fp32 --warmup 2

# nsys profile -f true -o sdpa_kernel_prefill --stats=true python attn_kernel.py --batch_size 1 --q_seq_len 512 --k_seq_len 512 --num_heads 32 --head_dim 128 --kernel kernel_sdpa_fwd --dtype fp32 --warmup 2
# nsys profile -f true -o sdpa_kernel_decode --stats=true python attn_kernel.py --batch_size 128 --q_seq_len 1 --k_seq_len 512 --num_heads 32 --head_dim 128 --kernel kernel_sdpa_fwd --dtype fp32 --warmup 2


# ncu with chunked sdpa
ncu  --set detailed -f -o sdpa_kernel_prefill python attn_kernel.py --batch_size 1 --q_seq_len 512 --k_seq_len 512 --num_heads 32 --head_dim 128 --kernel kernel_sdpa_fwd --dtype fp32
ncu  --set detailed -f -o sdpa_chunked_kernel_prefill_c512 python attn_kernel.py --batch_size 1 --q_seq_len 512 --k_seq_len 512 --num_heads 32 --head_dim 128 --kernel kernel_sdpa_chunked_fwd_512 --dtype fp32
ncu  --set detailed -f -o sdpa_chunked_kernel_prefill_c1024 python attn_kernel.py --batch_size 1 --q_seq_len 512 --k_seq_len 512 --num_heads 32 --head_dim 128 --kernel kernel_sdpa_chunked_fwd_1024 --dtype fp32
ncu  --set detailed -f -o sdpa_chunked_kernel_prefill_c2048 python attn_kernel.py --batch_size 1 --q_seq_len 512 --k_seq_len 512 --num_heads 32 --head_dim 128 --kernel kernel_sdpa_chunked_fwd_2048 --dtype fp32
ncu  --set detailed -f -o sdpa_chunked_kernel_prefill_c4096 python attn_kernel.py --batch_size 1 --q_seq_len 512 --k_seq_len 512 --num_heads 32 --head_dim 128 --kernel kernel_sdpa_chunked_fwd_4096 --dtype fp32
ncu  --set detailed -f -o flash_attn_kernel_prefill python attn_kernel.py --batch_size 1 --q_seq_len 512 --k_seq_len 512 --num_heads 32 --head_dim 128 --kernel kernel_flash_attention_fwd --dtype fp32

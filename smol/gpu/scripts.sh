# initial nsys hello world
python triton_elementwise_mul.py --size 20000000 --dtype float32 --iters 200 --warmup 50 --block 1024

nsys profile -o nsys_triton_mul --stats=true python triton_elementwise_mul.py --size 200000000 --iters 100 --warmup 50

# attn
python attn.py --batch 4 --seq 1024 --heads 8 --dim 64 --dtype bf16 --iters 200 --warmup 50

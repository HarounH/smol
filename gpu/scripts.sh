python triton_elementwise_mul.py --size 20000000 --dtype float32 --iters 200 --warmup 50 --block 1024

nsys profile -o nsys_triton_mul --stats=true python triton_elementwise_mul.py --size 200000000 --iters 100 --warmup 50
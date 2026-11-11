# Performance

## How to measure

```bash
make bench                                   # every model in models/ and models/gguf/
bash tests/run_benchmark.sh models/stories42M.bin models/gguf/tinyllama.gguf
./build/llm_bench models/stories42M.bin      # one model, detailed breakdown
OMP_NUM_THREADS=1 ./build/llm_bench models/stories42M.bin
```

`run_benchmark.sh` runs each model with 1 thread and with all cores, three
times each, and writes the median into
[`PERFORMANCE_RESULTS.md`](PERFORMANCE_RESULTS.md) together with the CPU,
RAM, compiler and date. For a GGUF-only breakdown per weight matrix, see
`./build/gguf_bench <model.gguf>` (`make gguf-tools`).

### Method

- **Deterministic workload:** a fixed prompt of 32 token IDs, then 64
  tokens of greedy generation with no early stop. Every run does the same
  work, so results can be compared across builds and machines.
- **Warm-up first:** one forward pass before timing pulls the memory-mapped
  weights in from disk, so the timings measure computation, not disk reads.
- **Timer:** `clock_gettime(CLOCK_MONOTONIC)` (wall-clock time).
- **Memory:** read from `/proc/self/status`. `VmRSS` is the resident set
  after loading; `VmHWM` is the peak resident set.

### Columns in the results table

| Column | Meaning |
|---|---|
| File (MB) | model file size = memory taken by the (mapped) weights |
| KV cache (MB) | `engine_kv_cache_bytes()`, for the 96 positions the benchmark allocates; see `MEMORY.md` for full-context sizes |
| Activations (MB) | one token's working buffers (RunState without the KV cache) |
| Peak RSS (MB) | the most physical memory the process used |
| Load (s) | `engine_open()`: header parsing, mapping, tokenizer, buffers |
| Prompt (tok/s) | prompt tokens processed per second |
| Generation (tok/s) | new tokens produced per second; this is the speed a user sees |

## Results on our development laptop

16 logical cores (likely 8 physical with hyper-threading), WSL 2 on Windows,
gcc `-O2 -fopenmp`. Exact CPU and RAM are in `PERFORMANCE_RESULTS.md`.

### Full benchmark (`make bench`, 32 prompt + 64 generated tokens)

| Model | Weights | Peak RSS | 1 thread | 16 threads |
|---|---:|---:|---:|---:|
| stories42M (llama2.c FP32) | 159 MB | 167 MB | **37.4 tok/s** | 32.7 tok/s |
| TinyLlama-1.1B-Chat (GGUF Q4_K_M) | 638 MB | 613 MB | 1.9 tok/s | 5.4 tok/s |

### TinyLlama: thread scaling and CPU-specific build (8 prompt + 32 generated tokens)

| Threads | `-O2` (default) | `-O2 -march=native` (`make NATIVE=1`) | Speed-up vs 1 thread |
|---:|---:|---:|---:|
| 1 | 1.93 tok/s (519 ms/token) | 2.03 tok/s (493 ms/token) | 1.0× |
| 2 | 3.68 tok/s | – | 1.9× |
| 4 | 5.10 tok/s | 5.50 tok/s | 2.6× |
| **8** | **8.03 tok/s** (125 ms/token) | **8.59 tok/s** (116 ms/token) | **4.2×** |
| 16 | 6.35 tok/s | – | 3.3× |

### What the numbers show

1. **The best thread count is the number of physical cores (8), not
   logical cores (16).** Scaling is almost perfect up to 2 threads (1.9×)
   and reaches 4.2× at 8. Going to 16 makes it *slower* (8.0 → 6.4 tok/s).
   Hyper-threads share their core's arithmetic units, so they add
   synchronisation overhead without adding compute. Recommended setting:
   `export OMP_NUM_THREADS=8`.
2. **Small models are faster single-threaded.** stories42M runs at 37 tok/s
   on 1 thread but 33 tok/s on 16. Its matrices are only 512 wide, so
   each matrix-vector product is a few microseconds of work, and starting and
   joining 16 threads for every product costs more than it saves.
   Parallelism only pays off once the per-product work is large, as with
   TinyLlama's 2048 × 5632 matrices.
3. **TinyLlama is limited by computation, not memory bandwidth.** At
   8.6 tok/s it reads 638 MB × 8.6 ≈ 5.5 GB/s of weights, far below what a
   laptop's RAM can deliver (tens of GB/s). The time goes into unpacking
   the 4- and 6-bit values in the Q4_K / Q6_K kernels.
4. **`-march=native` helps only about 5–8%.** GCC cannot auto-vectorise the
   bit-unpacking loops well, so allowing AVX2 changes little. The next
   step would be hand-written AVX2 intrinsics for the Q4_K / Q6_K dot
   products (see *Future work*).
5. **The cost per token follows model size.** TinyLlama does about 26× the
   work of stories42M per token (1.1 B vs 42 M parameters), and on 1 thread
   it is about 19× slower (37.4 vs 1.9 tok/s). The quantized kernels
   recover part of the gap because they read about 7× fewer bytes per weight than FP32.
6. **Memory ≈ file size.** Peak RSS is 167 MB for a 159 MB model and
   613 MB for a 638 MB one, because the weights are memory-mapped and the
   KV cache and activations add only a few MB (see `MEMORY.md`). RSS is
   slightly below the file size because the OS pages in only the parts of
   the file that are actually read.
7. **Load time:** 0.2 s (stories42M) and 1.9 s (TinyLlama). Both are
   dominated by reading the file through WSL from a OneDrive folder on the
   Windows drive. From the Linux file system, loading is close to instant.

### Reference: 2-core cloud build machine
Files with the exact shapes of the stories models (speed does not depend on
the weight values), 1 → 2 threads: stories15M 61 → 107 tok/s, stories42M
20 → 35 tok/s, stories110M 8 → 13 tok/s.

## Optimisations in the engine

| Optimisation | Where | Effect |
|---|---|---|
| `mmap()` the model file, no weight copies | `checkpoint_stories.c`, `gguf_model.c` | near-instant load (≈ 0.02 s); memory = file size |
| OpenMP row-parallel matrix-vector products | `matmul()` in `transformer.c`, `gguf_matvec()` | 4.2× on 8 cores (TinyLlama) |
| Fused kernels on quantized blocks (Q4_0, Q8_0, Q4_K, Q5_K, Q6_K) | `gguf_model.c` | no per-token dequantised copy of any matrix |
| KV cache reused between chat turns | `llm_generate -i` | each turn only processes its new tokens |
| Tokenizer merges in O(n log n) with a heap | `tokenizer.c` | a 20 KB prompt encodes in 0.01 s instead of 15 s; identical output |
| Optional `-march=native` | `make NATIVE=1` | lets GCC use AVX2; +5–8% measured on TinyLlama |

## Tips for faster runs

- Set `OMP_NUM_THREADS` to the number of **physical** cores (8 on our
  laptop gave 8.0 tok/s vs 6.4 tok/s with all 16 logical cores). Use
  `OMP_NUM_THREADS=1` for small models such as stories15M/42M.
- Build with `make NATIVE=1` on the machine you will run on.
- Under WSL, keep the model files in the Linux file system (`~/models`)
  rather than under `/mnt/c/...`, especially inside OneDrive.
- Use `--ctx N` (or `LLMOptions.max_context`) to allocate a smaller KV cache
  when you do not need the model's full context.
- For models bigger than about 100M parameters, prefer a Q4_K_M GGUF file
  over FP32.

## Future work

- **AVX2 / NEON intrinsics** for the Q4_K and Q6_K dot products. The
  measurements above show these kernels are the bottleneck; this is the
  approach llama.cpp uses for its CPU speed.
- **Batched prompt processing**: run all prompt tokens through each layer
  together as a matrix-matrix product, so each weight is read once per
  prompt instead of once per token.
- **Automatic thread choice**: use 1 thread for small matrices and the
  physical-core count for large ones, based on the matrix size.
- **Quantized KV cache** (FP16 / Q8) to halve or quarter the cache for long
  contexts.

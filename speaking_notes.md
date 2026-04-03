# Milestone 1 Speaking Notes (15 min, 4 speakers)

## 分工总览

| Speaker | Slides | 时长 | 内容 |
|---------|--------|------|------|
| **Xiangchen Song** | 1-2 (Title + Motivation) | ~3 min | 开场、背景、问题定义 |
| **Chenxin Geng** | 3-4 (Hypotheses + Hardware) | ~4 min | 三个假说 + 硬件选择 |
| **Yuyang Feng** | 5-6 (Kernels + Model) | ~4 min | 实验设计 + 预测模型 |
| **Tian Xia** | 7-8 (Results + Next Steps) | ~4 min | 结果分析 + 后续计划 |

---

## Xiangchen Song — Slides 1-2 (~3 min)

### Slide 1: Title (30s)

Hi everyone, we are Team [X]. Our project is about predicting cp.async pipeline effectiveness across different GPUs. I'm Xiangchen, and I'll start with the motivation. Then Chenxin will cover our hypotheses and hardware, Yuyang will explain the methodology and model, and Tian will present our initial results and next steps.

### Slide 2: Problem & Motivation (2.5 min)

First, some background. Starting from NVIDIA Ampere, there's an instruction called cp.async that lets the GPU copy data from global memory to shared memory asynchronously — meaning the copy can overlap with computation. This is really powerful. Libraries like CUTLASS, FlashAttention, and Triton all use multi-stage cp.async pipelines, and the pipeline depth S is one of the most important tuning knobs.

（指向右边的 pipeline 图）

As you can see in this diagram, with a multi-stage pipeline, while one stage is doing a memory copy, another stage can be doing computation — they overlap in time.

But here's the problem: in practice, people tune S on one GPU — say an A100 — and just assume the same setting works on other GPUs. That's a bad assumption, because the effectiveness of pipelining fundamentally depends on the memory subsystem — L2 cache size, memory bandwidth, memory latency, shared memory per SM — and these vary dramatically across GPU generations.

No existing autotuner actually uses these architecture parameters to guide its search.

So our research question is: does the optimal pipeline depth and its speedup actually diverge across GPUs? And can we build a parameter-free model — meaning no fitting on kernel data — that predicts this purely from architecture specs?

---

## Chenxin Geng — Slides 3-4 (~4 min)

### Slide 3: Hypotheses (1.5 min)

Thanks Xiangchen. So to structure our investigation, we formulated three falsifiable hypotheses.

**H1** says the speedup is primarily driven by L2 cache capacity. The idea is simple: when your concurrent working set exceeds the L2 size, you start hitting DRAM, and that's when pipelining really helps. So we predict GPUs with huge L2 — like L40S with 112 MB — should see minimal benefit, while GPUs with small L2 — like A40 with only 8 MB — should benefit the most at large problem sizes.

**H2** says it's about the ridge point — the ratio of peak compute to peak memory bandwidth. GPUs with a low ridge point are more memory-bound, so they should benefit more from hiding memory latency. We predict A100-MIG, which has the lowest ridge point at 7.9, benefits the most.

**H3** captures a tension: deeper pipelines hide more latency, but they also consume more shared memory per SM, which hurts occupancy. So the optimal depth should be non-monotonic on GPUs like A40, where latency is high but shared memory is tight.

In practice, we expect all three mechanisms to contribute, and our predictive model incorporates all of them.

### Slide 4: Hardware Setup (2.5 min)

To test these hypotheses, we selected 5 GPUs spanning 4 architecture generations.

（指向表格）

Let me highlight the key contrasts. A40 versus H100 gives us the maximum memory-subsystem gap — 8x difference in L2 size and nearly 5x in bandwidth. A100-MIG has the lowest ridge point at 7.9, so H2 predicts it benefits the most. L40S has the largest L2 at 112 MB measured, so it tests the "L2 absorbs everything" extreme. And V100 has no cp.async at all, so it serves as our boundary condition — speedup should be exactly 1.0x.

One really important finding here: measured L2 does NOT equal nominal L2. We ran pointer-chasing microbenchmarks and found that H100's effective L2 is only 28 MB, not the 50 MB on the spec sheet. That's because of L2 slice locality — remote slices have near-DRAM latency. On the other hand, L40S actually measured 112 MB, higher than its nominal 96 MB. This distinction turned out to be critical for model accuracy.

---

## Yuyang Feng — Slides 5-6 (~4 min)

### Slide 5: Kernels & Methodology (2 min)

Thanks Chenxin. Now let me explain what we actually run.

We chose two kernel families that sit on opposite sides of the roofline. GEMM is compute-bound — its arithmetic intensity is well above the ridge point — but it still has per-tile memory stalls that pipelining can help with. Stencil is memory-bound, with an arithmetic intensity of about 5 FLOP/byte, which is near or below the ridge point on most of our GPUs.

The key insight is: if our predictor works for both compute-bound and memory-bound kernels, it's capturing general memory-subsystem properties, not kernel-specific behavior.

For each kernel, we have three versions: V0 is naive, V1 uses shared memory with synchronous loads, and V3 uses cp.async pipelining with configurable depth S from 2 to 4. Importantly, the same V3 source code runs on all GPUs — no GPU-specific branches.

We sweep matrix size N from 256 to 8192, tile sizes from 16 to 128, and stencil lengths from 2^16 to 2^26. Each configuration runs with 3 warmup iterations and 10 timed iterations, and we take the median. Correctness is verified against cuBLAS for GEMM and CPU reference for stencil.

### Slide 6: Predictive Model (2 min)

Now the core contribution: our parameter-free predictive model.

The idea is straightforward — we want to predict cp.async speedup using only architecture constants and microbenchmark measurements, with zero parameters fitted from kernel data.

We need three inputs per GPU, all from our pointer-chasing microbenchmark: effective L2 capacity, L2 latency, and DRAM latency.

（指向右边公式）

The model works in three steps. First, we estimate the L2 hit fraction h — what fraction of your working set fits in L2. Second, we compute the effective memory latency as a weighted average of L2 and DRAM latency. Third, we compute the per-stage compute time, with a critical 280-cycle floor.

Why 280 cycles? Because even if your raw computation per tile is tiny — like 28 cycles for stencil — each pipeline stage has overhead from __syncthreads, cp.async commit and wait instructions, and warp scheduling. This floor was essential — without it, our stencil MAPE was 146%; with it, it dropped to 3.7%.

The final speedup formula is the ratio of latency-exposed time without pipelining to latency-exposed time with S stages, multiplied by an occupancy correction term with exponent 0.15 — because occupancy loss turned out to be highly sub-linear.

Again — zero fitted parameters. Everything comes from architecture specs or microbenchmarks.

---

## Tian Xia — Slides 7-8 (~4 min)

### Slide 7: Results & Key Findings (2.5 min)

Thanks Yuyang. Let me walk through our results.

（指向左边 MAPE 表格）

Overall, the model achieves about 10% MAPE across all GPUs and both kernels, with zero fitted parameters. Breaking it down: H100 is the most accurate at about 9%, A40 is around 9-11%, and L40S is the hardest at about 13-14%. Stencil is consistently more accurate than GEMM, which makes sense because stencil is more memory-bound and the pipeline benefit is more directly tied to memory latency.

（指向 L2 表格）

The L2 characterization results here are worth emphasizing. H100's effective L2 is only 28 MB — 44% of nominal. This is because of L2 slice topology: each SM can only efficiently access its local L2 slice; accessing remote slices costs nearly as much as going to DRAM. This is not documented by NVIDIA and was a key finding.

Now for the hypothesis status. H1 is clearly supported — L2 capacity is the primary driver. L40S with its huge L2 barely benefits from pipelining; A40 with small L2 benefits the most. H2 is only partially supported — ridge point alone isn't enough; you need to combine it with L2 and latency. H3 is also supported — we saw A40 with tile=64, stage=3 actually losing performance because occupancy dropped from 3 to 1 block per SM.

Three surprising findings: the 280-cycle overhead floor was essential for accuracy; occupancy loss is way more sub-linear than expected — going from 3 to 1 block/SM only costs 16%, not 42%; and L40S's massive L2 effectively eliminates the need for pipelining on most workloads.

### Slide 8: Next Steps & Timeline (1.5 min)

（指向 timeline 表格）

We're on track. Weeks 1 through 6 are done — we've built the infrastructure, implemented all kernels, collected data on 5 GPUs, and developed the predictive model.

For the remaining 4 weeks: Chenxin and Tian will do Nsight Compute deep-dives on our top-5 MAPE outliers and build the Triton pre-filter, which is our stretch goal — a Python plugin that uses our model to prune Triton's autotune search space. We also need to complete the A100-MIG experiments, which are pending cluster queue time.

In weeks 9-10, the whole team will focus on generating pipeline benefit heatmaps, doing leave-one-out cross-validation, and writing the final report and poster.

The main risk is A100-MIG: its L2 behavior under MIG partitioning might differ from a full A100. If our pointer-chase microbenchmark doesn't work correctly on MIG, we'll fall back to nominal L2 values with a sensitivity analysis.

That's our progress so far. We're happy to take questions.

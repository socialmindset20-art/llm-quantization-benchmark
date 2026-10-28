import time
import torch

def run_inference_benchmark(model, tokenizer, prompt="Explique de forma técnica o que é MLOps em um parágrafo.", target_new_tokens=128):
    print(f"[Benchmark] Iniciando perfilamento com {target_new_tokens} tokens...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_len = inputs.input_ids.shape[1]

    # Warmup
    _ = model.generate(**inputs, max_new_tokens=10, do_sample=False)
    if device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        initial_vram = torch.cuda.memory_allocated() / (1024 ** 2)

    # TTFT
    if device == "cuda":
        start_event_ttft = torch.cuda.Event(enable_timing=True)
        end_event_ttft = torch.cuda.Event(enable_timing=True)
        start_event_ttft.record()
        _ = model.generate(**inputs, max_new_tokens=1, min_new_tokens=1, do_sample=False)
        end_event_ttft.record()
        torch.cuda.synchronize()
        ttft_ms = start_event_ttft.elapsed_time(end_event_ttft)
    else:
        t0 = time.time()
        _ = model.generate(**inputs, max_new_tokens=1, min_new_tokens=1, do_sample=False)
        ttft_ms = (time.time() - t0) * 1000.0

    # Generation
    if device == "cuda":
        start_event_gen = torch.cuda.Event(enable_timing=True)
        end_event_gen = torch.cuda.Event(enable_timing=True)
        start_event_gen.record()
        outputs = model.generate(**inputs, max_new_tokens=target_new_tokens, min_new_tokens=target_new_tokens, do_sample=False)
        end_event_gen.record()
        torch.cuda.synchronize()
        generation_time_ms = start_event_gen.elapsed_time(end_event_gen)
    else:
        t0 = time.time()
        outputs = model.generate(**inputs, max_new_tokens=target_new_tokens, min_new_tokens=target_new_tokens, do_sample=False)
        generation_time_ms = (time.time() - t0) * 1000.0

    generation_time_s = generation_time_ms / 1000.0
    new_tokens_generated = outputs.shape[1] - input_len
    throughput_tps = new_tokens_generated / generation_time_s if generation_time_s > 0 else 0.0
    latency_total_ms = ttft_ms + generation_time_ms

    peak_vram_mb = 0.0
    generation_overhead_vram = 0.0
    if device == "cuda":
        peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
        generation_overhead_vram = peak_vram_mb - initial_vram

    return {
        "ttft_ms": ttft_ms,
        "throughput_tps": throughput_tps,
        "latency_total_ms": latency_total_ms,
        "peak_vram_mb": peak_vram_mb,
        "generation_overhead_vram": generation_overhead_vram
    }

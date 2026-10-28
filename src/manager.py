import gc
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

class ModelManager:
    @staticmethod
    def cleanup():
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            print("[MLOps Cleanup] Cache de VRAM CUDA liberado com sucesso.")
        else:
            print("[MLOps Cleanup] Garbage Collector executado (Sem CUDA).")

    @staticmethod
    def load_model(precision_mode, model_id="Qwen/Qwen2.5-1.5B-Instruct"):
        ModelManager.cleanup()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        initial_vram = 0.0
        if device == "cuda":
            initial_vram = torch.cuda.memory_allocated() / (1024 ** 2)

        print(f"\n[Loading] Carregando modelo em modo {precision_mode}...")
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

        if precision_mode == "FP16":
            target_dtype = torch.float16 if device == "cuda" else torch.float32
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=target_dtype,
                device_map="auto" if device == "cuda" else None,
                trust_remote_code=True
            )
        elif precision_mode == "INT8":
            if device != "cuda":
                raise RuntimeError("Quantização de 8-bits requer GPU NVIDIA!")
            quant_config = BitsAndBytesConfig(load_in_8bit=True)
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                quantization_config=quant_config,
                device_map="cuda",
                trust_remote_code=True
            )
        elif precision_mode == "INT4":
            if device != "cuda":
                raise RuntimeError("Quantização de 4-bits requer GPU NVIDIA!")
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.float16
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                quantization_config=quant_config,
                device_map="cuda",
                trust_remote_code=True
            )
        else:
            raise ValueError(f"Modo não suportado: {precision_mode}")

        weights_vram = 0.0
        if device == "cuda":
            torch.cuda.synchronize()
            final_vram = torch.cuda.memory_allocated() / (1024 ** 2)
            weights_vram = final_vram - initial_vram

        return model, tokenizer, weights_vram

# Benchmark de Quantização e Trade-offs de Desempenho (FP16 vs INT8 vs INT4)

Relatório de engenharia de MLOps automatizado, focado em mensurar throughput, latência de tempo ao primeiro token (TTFT), ocupação de VRAM e perplexidade semântica utilizando o modelo base `Qwen/Qwen2.5-1.5B-Instruct` sobre uma GPU NVIDIA T4 (16GB de VRAM).

## 📊 Resultados do Benchmark

| Precisão   |   VRAM Pesos (MB) |   Pico VRAM total (MB) |   Overhead Ativação (MB) |   TTFT (ms) |   Throughput (tokens/s) |   Latência Total (ms) |   Perplexidade (PPL) |
|:-----------|------------------:|-----------------------:|-------------------------:|------------:|------------------------:|----------------------:|---------------------:|
| FP16       |           3086.12 |                4124.55 |                  1038.43 |       42.15 |                   48.22 |               2698.41 |               5.1245 |
| INT8       |           1642.3  |                2588.12 |                   945.82 |       78.44 |                   32.15 |               4059.18 |               5.1298 |
| INT4       |            954.12 |                1822.45 |                   868.33 |       92.81 |                   29.84 |               4381.12 |               5.1612 |

## 🧠 Insights de Engenharia & Conclusões

1. **Redução Drástica do Footprint de VRAM**:
   - A quantização para INT4 e INT8 reduz significativamente o espaço físico consumido pelos pesos na GPU, permitindo que o modelo seja executado em ambientes com menos recursos de memória, ideal para ambientes edge ou instâncias serverless econômicas.

2. **O Dilema da Latência de Quantização em GPUs Grátis**:
   - Em GPUs T4, a quantização INT4/INT8 pode não acelerar o throughput total em relação ao FP16. Isso ocorre porque o hardware da T4 não possui núcleos Tensor dedicados de alta performance otimizados para operações inteiras de baixa precisão sem overhead de desquantização dinâmica (dequantization overhead) em tempo de execução de kernels.

3. **Estabilidade Semântica (Perplexidade)**:
   - A perda de qualidade semântica mensurada pela perplexidade de linguagem no formato INT4 (NF4) é mantida extremamente estável em comparação com a precisão total em FP16, comprovando o sucesso de técnicas como a dupla quantização (Double Quantization) para manter a fluência linguística intacta.

---
*Gerado automaticamente pelo script de Monitoramento de MLOps de Inferência no Google Colab.*

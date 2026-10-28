import matplotlib.pyplot as plt
import seaborn as sns

def create_tradeoff_plots(df, output_path="results/llm_inference_benchmark_tradeoffs.png"):
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle('Análise de Trade-off de Inferência de LLMs (FP16 x INT8 x INT4)', fontsize=16, fontweight='bold', y=1.02)

    # VRAM
    sns.barplot(data=df, x="Precisão", y="Pico VRAM total (MB)", hue="Precisão", ax=axes[0], palette="Blues_d", legend=False)
    axes[0].set_title("Alocação Máxima de VRAM (Menor é Melhor)", fontsize=12)
    for container in axes[0].containers:
        axes[0].bar_label(container, fmt='%.1f MB')

    # Throughput
    sns.barplot(data=df, x="Precisão", y="Throughput (tokens/s)", hue="Precisão", ax=axes[1], palette="Greens_d", legend=False)
    axes[1].set_title("Velocidade de Geração (Maior é Melhor)", fontsize=12)
    for container in axes[1].containers:
        axes[1].bar_label(container, fmt='%.2f tps')

    # Perplexity
    sns.lineplot(data=df, x="Precisão", y="Perplexidade (PPL)", marker="o", ax=axes[2], color="red", linewidth=2.5, markersize=10)
    axes[2].set_title("Perplexidade Semântica (Menor é Melhor)", fontsize=12)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"[Visualização] Gráfico de Trade-offs salvo com sucesso em: '{output_path}'")

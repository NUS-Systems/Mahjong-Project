import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib
import csv
# 更新 matplotlib 参数
matplotlib.rcParams.update({'font.size': 20})

# 生成数据
def generate_data(file_path):
    models = ["LLaMA-1B", "LLaMA-3B", "LLaMA-8B", "Qwen-7B", "Ministral-8B"]
    schemes = ["'Linearized'", "Bit", "Compression"]
    redundancies = [6, 8, 10]
    error_rates = [0.01, 0.015, 0.02]
    coverages = [5, 10, 15]
    
    data = []
    for model in models:
        for scheme in schemes:
            for redundancy in redundancies:
                for error_rate in error_rates:
                    for coverage in coverages:
                        data.append({
                            "Model": model,
                            "Scheme": scheme,
                            "Redundancy": redundancy,
                            "Error Rate": error_rate,
                            "Coverage": coverage,
                            "Distance": abs(hash(f"{model}{scheme}{redundancy}{error_rate}{coverage}")) % 100 / 10,
                            "MSE": abs(hash(f"{model}{scheme}{redundancy}{error_rate}{coverage}")) % 50 / 10
                        })
    data = []
    with open(file_path, mode='r') as file:
        reader = csv.DictReader(file, delimiter=',')
        for row in reader:
            # 将 "*" 替换为 0.0，并确保数值字段转换为 float
            data.append({
                "Model": row["Model"],
                "Scheme": row["Scheme"],
                "Redundancy": int(row["Redundancy"]),
                "Error Rate": float(row["Error Rate"]),
                "Coverage": int(row["Coverage"]),
                "Distance": float(row["Distance"]) if row["Distance"] != "*" else 0.0,
                "MSE": float(row["MSE"]) if row["MSE"] != "*" else 0.0,
            })
    
    return pd.DataFrame(data)

def shorten_model_name(model_name):
    """根据规则对模型名称进行缩写"""
    if model_name.startswith("Llama"):
        return f"L{model_name.split('-')[2][0]}"  # 提取 '1', '3', '8'
    elif model_name.startswith("Qwen"):
        return f"Q{model_name.split('-')[1][0]}"  # 提取 '7'
    elif model_name.startswith("Ministral"):
        return f"M{model_name.split('-')[1][0]}"  # 提取 '8'
    else:
        return model_name  # 不在规则中的保持原样
    
# 绘制分组柱状图
def plot_grouped_bar_chart(df, x, metric, group, xlabel, ylabel, save_path, shorten_labels=False):
    # 提取分类和组
    categories = df[x].unique()
    groups = df[group].unique()
    # 如果需要缩短标签
    if shorten_labels:
        categories_display = [shorten_model_name(cat) for cat in categories]
    else:
        categories_display = categories
    # 设置分组柱状图的参数
    x_indexes = np.arange(len(categories))*0.4
    bar_width = 0.1  # 柱状图的宽度
    
    # 创建绘图
    fig, ax = plt.subplots(figsize=(6, 4.5))
    
    for i, g in enumerate(groups):
        group_df = df[df[group] == g]
        ax.bar(
            x_indexes + i * bar_width, 
            group_df[metric], 
            width=bar_width, 
            label=g,
            alpha=0.8
        )
    
    # 添加标签和格式
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x_indexes + bar_width * (len(groups) - 1) / 2)
    ax.set_xticklabels(categories_display, fontsize=18)
    ax.legend(
        frameon=False, 
        loc='upper center', 
        bbox_to_anchor=(0.5, 1.28), 
        ncol=3, 
        fontsize=20, 
        columnspacing=0.5,  # 减少列之间的距离
        handletextpad=0.2,  # 减少标记与文字的距离
        handlelength=0.8,   # 缩小标记的长度
        handleheight=0.5,   # 缩小标记的高度
        borderpad=0.2,      # 减少图例外边距
        labelspacing=0.2    # 减少图例条目之间的垂直间距
    )

    plt.tight_layout()
    plt.grid()
    # 保存图片
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

# 绘制模型对比图（第一张图，分开保存）
def plot_model_comparison(df, redundancy, error_rate, coverage, save_path):
    filtered_df = df[(df["Redundancy"] == redundancy) & 
                     (df["Error Rate"] == error_rate) & 
                     (df["Coverage"] == coverage)]
    
    for metric in ["Distance", "MSE"]:
        metric_path = os.path.join(save_path, f"model_comparison_{metric.lower()}_r{redundancy}_e{error_rate}_c{coverage}.pdf")
        plot_grouped_bar_chart(
            filtered_df,
            x="Model",
            metric=metric,
            group="Scheme",
            xlabel="Models",
            ylabel=metric,
            save_path=metric_path,
            shorten_labels=True
        )

# 绘制冗余对比图（第二张图，分开保存）
def plot_redundancy_comparison(df, model, error_rate, coverage, save_path):
    filtered_df = df[(df["Model"] == model) & 
                     (df["Error Rate"] == error_rate) & 
                     (df["Coverage"] == coverage)]
    
    for metric in ["Distance", "MSE"]:
        metric_path = os.path.join(save_path, f"redundancy_comparison_{metric.lower()}_m{model}_e{error_rate}_c{coverage}.pdf")
        plot_grouped_bar_chart(
            filtered_df,
            x="Redundancy",
            metric=metric,
            group="Scheme",
            xlabel="Redundancy",
            ylabel=metric,
            save_path=metric_path
        )

# 绘制错误率对比图（第三张图，分开保存）
def plot_error_rate_comparison(df, model, redundancy, coverage, save_path):
    filtered_df = df[(df["Model"] == model) & 
                     (df["Redundancy"] == redundancy) & 
                     (df["Coverage"] == coverage)]
    
    for metric in ["Distance", "MSE"]:
        metric_path = os.path.join(save_path, f"error_rate_comparison_{metric.lower()}_m{model}_r{redundancy}_c{coverage}.pdf")
        plot_grouped_bar_chart(
            filtered_df,
            x="Error Rate",
            metric=metric,
            group="Scheme",
            xlabel="Error Rate",
            ylabel=metric,
            save_path=metric_path
        )

# 绘制覆盖率对比图（第四张图，分开保存）
def plot_coverage_comparison(df, model, redundancy, error_rate, save_path):
    filtered_df = df[(df["Model"] == model) & 
                     (df["Redundancy"] == redundancy) & 
                     (df["Error Rate"] == error_rate)]
    
    for metric in ["Distance", "MSE"]:
        metric_path = os.path.join(save_path, f"coverage_comparison_{metric.lower()}_m{model}_r{redundancy}_e{error_rate}.pdf")
        plot_grouped_bar_chart(
            filtered_df,
            x="Coverage",
            metric=metric,
            group="Scheme",
            xlabel="Coverage",
            ylabel=metric,
            save_path=metric_path
        )

# 主函数
def main():
    save_path = "/home/gaobin/DNAStorageToolkit/Mahjong/paper-plot/figures/final"
    os.makedirs(save_path, exist_ok=True)
    
    df = generate_data("/home/gaobin/Mahjong-Project/Final-Auto/comparison_results.csv")
    
    # 第一张图
    plot_model_comparison(df, redundancy=10, error_rate=0.01, coverage=5, save_path=save_path)
    
    # 第二张图
    plot_redundancy_comparison(df, model="Llama-3.1-8B", error_rate=0.01, coverage=5, save_path=save_path)
    
    # 第三张图
    plot_error_rate_comparison(df, model="Llama-3.1-8B", redundancy=10, coverage=5, save_path=save_path)
    
    # 第四张图
    plot_coverage_comparison(df, model="Llama-3.1-8B", redundancy=10, error_rate=0.01, save_path=save_path)

if __name__ == "__main__":
    main()

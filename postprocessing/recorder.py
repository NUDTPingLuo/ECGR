import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import json
from json import JSONEncoder
import pickle
import re
from matplotlib import rcParams
import seaborn as sns
import os
from collections import defaultdict
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.patches as patches
import csv

json_types = (list, dict, str, int, float, bool, type(None))


class PythonObjectEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, json_types):
            return super().default(self, obj)
        return {'_python_object': pickle.dumps(obj).decode('latin-1')}


def as_python_object(dct):
    if '_python_object' in dct:
        return pickle.loads(dct['_python_object'].encode('latin-1'))
    return dct


class Recorder(object):
    def __init__(self):
        self.res_list = []
        self.res = {'server': {'iid_accuracy': [], 'train_loss': []},
                    'clients': {'iid_accuracy': [], 'train_loss': []}}

    def load(self, filename, label):
        """
        Load the result files
        :param filename: Name of the result file
        :param label: Label for the result file
        """
        with open(filename) as json_file:
            res = json.load(json_file, object_hook=as_python_object)
        self.res_list.append((res, label))

    def plot(self, figsize=(6, 5)):
        """
        Plot testing accuracy (mean ± range) across different seeds with custom figure size and Nature-style color scheme.
        """

        # ===== 设置字体嵌入 =====
        global Dirichlet
        rcParams['pdf.fonttype'] = 42
        rcParams['ps.fonttype'] = 42
        plt.rcParams['font.family'] = 'DejaVu Sans'

        # 设置 seaborn 风格和调色板
        sns.set_style("white")
        color_palette = sns.color_palette("colorblind")

        # 使用 color_palette 定义颜色映射
        base_color_map = {
            'FedAvg': color_palette[3],  # 红色
            'Scaffold': color_palette[0],  # 蓝色
            'FedProx': color_palette[9],  # 青色
            'FedNova': color_palette[1],  # 橙色
            'FedAvgECGR': color_palette[2],  # 绿色
            'ScaffoldECGR': color_palette[2],
            'FedNovaECGR': color_palette[2],
            'FedProxECGR': color_palette[2],
        }

        # 定义线型映射
        line_style_map = {
            'FedAvg': '-',
            'Scaffold': '--',
            'FedProx': '-.',
            'FedNova': ':',
            'FedAvgECGR': '-',
            'ScaffoldECGR': '-',
            'FedNovaECGR': '-',
            'FedProxECGR': '-',
        }

        # ===== 按算法 + 数据集 + Dirichlet 聚合同类实验（不同 seed） =====
        grouped_results = defaultdict(list)
        for res, label in self.res_list:
            matches = re.findall(r"'([^']+)'", label)
            Algorithm = matches[0]
            Dataset = matches[1]
            Dirichlet = matches[2]
            lr = matches[3]
            key = (Algorithm, Dataset, Dirichlet)
            grouped_results[key].append(np.array(res['server']['iid_accuracy']))

        # ===== 开始绘图 =====
        fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)

        for (Algorithm, Dataset, Dirichlet), acc_list in grouped_results.items():
            # 对齐长度（防止不同实验长度略有差异）
            min_len = min(len(a) for a in acc_list)
            acc_array = np.array([a[:min_len] for a in acc_list])

            # 计算均值、最大值、最小值
            mean_acc = acc_array.mean(axis=0)
            max_acc = acc_array.max(axis=0)
            min_acc = acc_array.min(axis=0)

            # 绘图属性
            color = base_color_map.get(Algorithm, None)
            linestyle = line_style_map.get(Algorithm, '-')
            alpha_value = 0.4 if "ECGR" in Algorithm else 0.2
            linewidth_value = 3.0 if "ECGR" in Algorithm else 2.0

            # 绘制平均线
            ax.plot(
                mean_acc,
                label=Algorithm,
                alpha=1,
                linewidth=linewidth_value,
                color=color,
                linestyle=linestyle
            )

            # 绘制阴影（最大值与最小值之间区域）
            ax.fill_between(
                range(min_len),
                min_acc,
                max_acc,
                color=color,
                alpha=alpha_value
            )

        # 创建 inset axes（放大图）
        inset_ax = inset_axes(
            ax,
            width="30%",  # inset 宽度
            height="30%",  # inset 高度
            bbox_to_anchor=(-0.15, -0.55, 1, 1),  # 右侧中间
            bbox_transform=ax.transAxes,
            borderpad=0
        )

        start_i, end_i = 90, 100

        for (Algorithm, Dataset, Dirichlet), acc_list in grouped_results.items():
            # 对齐长度
            min_len = min(len(a) for a in acc_list)
            acc_array = np.array([a[:min_len] for a in acc_list])

            # 计算均值（只取 90-100 范围）
            mean_acc = acc_array.mean(axis=0)[start_i:end_i]

            color = base_color_map.get(Algorithm, None)
            linestyle = line_style_map.get(Algorithm, '-')
            linewidth_value = 3.0 if "ECGR" in Algorithm else 2.0

            x_range = np.arange(start_i, end_i)

            # ⭐ 只画曲线，不画阴影
            inset_ax.plot(
                x_range,
                mean_acc,
                color=color,
                linestyle=linestyle,
                linewidth=linewidth_value,
            )

        # inset 图细节
        inset_ax.tick_params(labelsize=8)
        inset_ax.grid(alpha=0.3)

        # ===== 在主图 ↔ inset 之间添加连接线（Nature风格）=====
        try:
            ax.indicate_inset_zoom(inset_ax, edgecolor="black", linewidth=1.2)
        except Exception as e:
            print("⚠️ inset zoom connection unavailable:", e)

        # ===== 图像细节设置 =====
        ax.set_xlabel('Epochs', size=12)
        ax.set_ylabel('Testing Accuracy', size=12)
        # ax.legend(prop={'size': 14}, loc='lower right')
        ax.tick_params(axis='both', labelsize=12)
        ax.grid(alpha=0.3)

        # plt.tight_layout(rect=[0, 0, 1, 0.95])
        # plt.suptitle(Dataset, size=16, fontweight='bold')

        # ===== 保存结果 =====
        save_dir = os.path.join('..', 'plot', Dataset, Dirichlet, lr)
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"{Dataset}_{Dirichlet}_{lr}_{Algorithm}.pdf")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    def plot_mean(self, figsize=(6, 5)):
        """
        Plot only the mean testing accuracy for all methods on the same figure.
        ECGR: solid line with baseline color
        Baseline: dashed line
        """

        # ===== 字体嵌入 =====
        rcParams['pdf.fonttype'] = 42
        rcParams['ps.fonttype'] = 42
        plt.rcParams['font.family'] = 'DejaVu Sans'

        sns.set_style("white")
        color_palette = sns.color_palette("colorblind")

        # 基线颜色映射
        base_color_map = {
            'FedAvg': color_palette[2],
            'Scaffold': color_palette[0],
            'FedProx': color_palette[9],
            'FedNova': color_palette[1],
        }

        # 将 ECGR 方法映射到对应 baseline 的颜色
        ecgr_color_map = {
            'FedAvgECGR': base_color_map['FedAvg'],
            'ScaffoldECGR': base_color_map['Scaffold'],
            'FedNovaECGR': base_color_map['FedNova'],
            'FedProxECGR': base_color_map['FedProx'],
        }

        # 聚合同类实验（不同 seed）
        grouped_results = defaultdict(list)
        for res, label in self.res_list:
            matches = re.findall(r"'([^']+)'", label)
            Algorithm = matches[0]
            Dataset = matches[1]
            Dirichlet = matches[2]
            lr = matches[3]
            beta = matches[4]
            key = (Algorithm, Dataset, Dirichlet)
            grouped_results[key].append(np.array(res['server']['iid_accuracy']))

        # ===== 绘图 =====
        fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)

        for (Algorithm, Dataset, Dirichlet), acc_list in grouped_results.items():
            min_len = min(len(a) for a in acc_list)
            acc_array = np.array([a[:min_len] for a in acc_list])
            mean_acc = acc_array.mean(axis=0)

            # 区分 ECGR 与 baseline
            if 'ECGR' in Algorithm:
                color = ecgr_color_map.get(Algorithm, 'black')
                linestyle = '-'
            else:
                color = base_color_map.get(Algorithm, 'black')
                linestyle = '--'

            ax.plot(
                mean_acc,
                label=Algorithm,
                color=color,
                linestyle=linestyle,
                linewidth=2.5
            )

        # ===== 图像细节 =====
        ax.set_xlabel('Epochs', size=12)
        ax.set_ylabel('Testing Accuracy', size=12)
        ax.tick_params(axis='both', labelsize=12)
        ax.grid(alpha=0.3)
        ax.legend(prop={'size': 12}, loc='lower right')
        plt.suptitle(Dataset, size=16, fontweight='bold')

        # plt.tight_layout()
        # ===== 保存结果 =====
        save_dir = os.path.join('..', 'plot', Dataset, Dirichlet, lr)
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"{Dataset}_{Dirichlet}_{lr}_{beta}.pdf")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    def plot_ecgr_beta_compare(self, figsize=(6, 5)):
        """
        Plot ECGR methods with different beta values.
        For each algorithm:
            - beta=0   : dashed line
            - beta=0.2 : solid line
        Each line is averaged over 5 seeds.
        """

        # ===== 字体嵌入 =====
        rcParams['pdf.fonttype'] = 42
        rcParams['ps.fonttype'] = 42
        plt.rcParams['font.family'] = 'DejaVu Sans'

        sns.set_style("white")
        color_palette = sns.color_palette("colorblind")

        # 只画这四个算法
        algorithms = ['FedAvgECGR', 'FedProxECGR', 'FedNovaECGR', 'ScaffoldECGR']

        # 给每个算法分配固定颜色
        alg_color_map = {
            'FedAvgECGR': color_palette[2],
            'FedProxECGR': color_palette[9],
            'FedNovaECGR': color_palette[1],
            'ScaffoldECGR': color_palette[0],
        }

        # ===== 聚合同类实验 (alg, beta) =====
        grouped_results = defaultdict(list)
        meta_info = {}

        for res, label in self.res_list:
            # 解析如 ['FedAvgECGR','MNIST','alpha0.01','lr0.001','beta0','seed0']
            matches = re.findall(r"'([^']+)'", label)

            Algorithm = matches[0]
            Dataset = matches[1]
            Dirichlet = matches[2]
            lr = matches[3]
            beta = matches[4]      # e.g. 'beta0' or 'beta0.2'
            seed = matches[5]

            # 只保留指定算法
            if Algorithm not in algorithms:
                continue

            key = (Algorithm, beta)
            grouped_results[key].append(np.array(res['server']['iid_accuracy']))

            # 保存公共元信息
            meta_info['Dataset'] = Dataset
            meta_info['Dirichlet'] = Dirichlet
            meta_info['lr'] = lr

        # ===== 开始绘图 =====
        fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)

        for (Algorithm, beta), acc_list in grouped_results.items():
            if len(acc_list) == 0:
                continue

            # 对齐长度
            min_len = min(len(a) for a in acc_list)
            acc_array = np.array([a[:min_len] for a in acc_list])
            mean_acc = acc_array.mean(axis=0)

            # 颜色
            color = alg_color_map.get(Algorithm, 'black')

            # 线型规则
            if beta == 'beta0.2':
                linestyle = '-'      # 实线
            else:
                linestyle = '--'     # 虚线

            beta_val = beta.replace('beta', '')
            label = rf"{Algorithm} ($\beta={beta_val}$)"

            ax.plot(
                mean_acc,
                label=label,
                color=color,
                linestyle=linestyle,
                linewidth=2.5
            )

        # ===== 图像细节 =====
        ax.set_xlabel('Epochs', size=12)
        ax.set_ylabel('Testing Accuracy', size=12)
        ax.tick_params(axis='both', labelsize=12)
        ax.grid(alpha=0.3)
        ax.legend(prop={'size': 12}, loc='lower right')

        Dataset = meta_info.get('Dataset', '')
        Dirichlet = meta_info.get('Dirichlet', '')
        lr = meta_info.get('lr', '')

        plt.suptitle(Dataset, size=16, fontweight='bold')

        # ===== 保存结果 =====
        save_dir = os.path.join('..', 'plot', Dataset, Dirichlet, lr)
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"{Dataset}_ECGR_beta_compare.pdf")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    def save_final_acc_to_csv(self, save_name='final_acc_summary.csv'):
        """
        保存8种算法的最终精度统计到CSV：
        - final_mean_acc: 最后一轮的平均acc（5个seed均值）
        - best_acc: 所有round + 所有seed中的最大acc
        """

        target_algs = [
            'FedAvg', 'FedProx', 'FedNova', 'Scaffold',
            'FedAvgECGR', 'FedProxECGR', 'FedNovaECGR', 'ScaffoldECGR'
        ]

        # 按 (Algorithm, Dataset, Dirichlet) 分组
        grouped_results = defaultdict(list)

        for res, label in self.res_list:
            matches = re.findall(r"'([^']+)'", label)
            Algorithm = matches[0]
            Dataset = matches[1]
            Dirichlet = matches[2]

            if Algorithm in target_algs:
                key = (Algorithm, Dataset, Dirichlet)
                grouped_results[key].append(np.array(res['server']['iid_accuracy']))

        # ===== 写入CSV =====
        save_path = os.path.join('..', 'plot', f'{Dataset}_acc_summary.csv')

        with open(save_path, 'w', newline='') as f:
            writer = csv.writer(f)
            # 表头
            writer.writerow([
                'Algorithm', 'Dataset', 'Dirichlet',
                'Final_Mean_Acc', 'Best_Acc'
            ])

            # 计算统计量
            for (Algorithm, Dataset, Dirichlet), acc_list in grouped_results.items():
                # 截断到一致长度
                min_len = min(len(a) for a in acc_list)
                acc_array = np.array([a[:min_len] for a in acc_list])  # shape: (num_seed, rounds)

                # 最后一轮的平均acc
                final_mean_acc = acc_array[:, -1].mean()

                # 所有seed + 所有round中的最大值
                best_acc = acc_array.max()

                writer.writerow([
                    Algorithm,
                    Dataset,
                    Dirichlet,
                    f"{final_mean_acc:.4f}",
                    f"{best_acc:.4f}"
                ])

        print(f"Saved accuracy summary to: {save_path}")



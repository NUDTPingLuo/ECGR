import copy
import torch
from fed_baselines.client_base import FedClient
from utils.models import *
from torch.utils.data import DataLoader
import torch.nn.functional as F


class ECGRClient(FedClient):
    def __init__(self, name, epoch, dataset_id, model_name, lr, batch_size, momentum, beta):
        super().__init__(name, epoch, dataset_id, model_name, lr, batch_size, momentum)
        self.select_ratio = 0.5  # 可调参数
        self.beta = beta

    def flatten_grad(self, grad_dict, device):
        """把梯度字典展平为一个大向量，排除BN统计量"""
        flat_list = []
        for name, v in grad_dict.items():
            if ("mean" not in name.lower()
                    and "var" not in name.lower()
                    and "num_batches_tracked" not in name):
                flat_list.append(v.flatten().to(device))
        return torch.cat(flat_list) if flat_list else torch.tensor([], device=device)

    def select_grads(self, local_grad_list, avg_grad, select_ratio):
        """
        贪心选择与当前总梯度 S 最近的梯度
        :param local_grad_list: list，每一步的梯度字典
        :param avg_grad: dict，平均梯度
        :param select_ratio: float，选择比例
        :return: selected_indices (list)
        """
        device = next(iter(local_grad_list[0].values())).device
        if avg_grad is None:
            avg_grad = {name: torch.zeros_like(param, device=device) for name, param in local_grad_list[0].items()}

        # 1. 展平梯度
        flat_avg = self.flatten_grad(avg_grad, device)
        flat_grads = [self.flatten_grad(g, device) for g in local_grad_list]

        # 2. 归一化梯度
        normalized = [(g - flat_avg).to(device) for g in flat_grads]

        # 3. 初始化 S 为全零
        S = torch.zeros_like(flat_avg, device=device)

        selected_indices = []
        remaining_indices = list(range(len(normalized)))
        num_select = max(1, int(len(normalized) * select_ratio))

        for _ in range(num_select):
            # 批量计算所有候选的距离（GPU 并行）
            candidates = torch.stack([S + normalized[i] for i in remaining_indices])
            distances = torch.norm(candidates, dim=1)

            # 选最小的
            best_rel_idx = torch.argmin(distances).item()
            best_idx = list(remaining_indices)[best_rel_idx]

            # 更新
            selected_indices.append(best_idx)
            S = S + normalized[best_idx]
            remaining_indices.remove(best_idx)

        return selected_indices

    def select_grads_topk(self, grad_list, avg_grad, select_ratio):
        """
        用 TopK 最近邻替代贪心选择梯度
        :param grad_list: list，每步的梯度字典
        :param avg_grad: dict，平均梯度
        :param select_ratio: float，选择比例
        :return: selected_indices (list)
        """
        device = next(iter(grad_list[0].values())).device
        if avg_grad is None:
            avg_grad = {name: torch.zeros_like(param, device=device) for name, param in grad_list[0].items()}
        flat_avg = self.flatten_grad(avg_grad, device)

        # 展平梯度并计算与平均梯度的差
        flat_grads = torch.stack([self.flatten_grad(g, device) for g in grad_list])
        diffs = flat_grads - flat_avg

        # 计算 L2 距离
        distances = torch.norm(diffs, dim=1)

        # 取距离最小的前 num_select 个
        num_select = max(1, int(len(grad_list) * select_ratio))
        _, selected_indices = torch.topk(-distances, num_select)  # 负号表示取最小值

        return selected_indices.tolist()

    def select_grads_topk_cos(self, grad_list, avg_grad, select_ratio):
        """
        用 TopK 最近邻替代贪心选择梯度（改为夹角最小，即余弦相似度最大）
        :param grad_list: list，每步的梯度字典
        :param avg_grad: dict，平均梯度
        :param select_ratio: float，选择比例
        :return: selected_indices (list)
        """
        device = next(iter(avg_grad.values())).device
        flat_avg = self.flatten_grad(avg_grad, device)

        # 展平梯度
        flat_grads = torch.stack([self.flatten_grad(g, device) for g in grad_list])

        # 计算余弦相似度
        sims = [F.cosine_similarity(g, flat_avg, dim=0) for g in flat_grads]
        selected_indices = torch.topk(torch.stack(sims), k=int(len(sims) * select_ratio)).indices.tolist()

        return selected_indices

    # # 距离函数
    # def l2_distance(self, g, ref_grad):
    #     return sum(
    #         torch.norm(g[k] + ref_grad[k])
    #         for k in g
    #         if "mean" not in k.lower()  # 排除任何名字里含 mean 的层
    #         and "var" not in k.lower()  # 排除任何名字里含 var 的层
    #         and "num_batches_tracked" not in k
    #     )

    def collect_grads(self, update_model):
        train_loader = DataLoader(self.trainset, batch_size=self._batch_size, shuffle=True, num_workers=8, pin_memory=True, persistent_workers=False)

        self.model.to(self._device)
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self._lr, momentum=self._momentum)
        loss_func = torch.nn.CrossEntropyLoss()
        grad_list = []

        # 记录初始模型参数（用于step=0时的prev_params）
        prev_params = {name: param.detach().clone() for name, param in self.model.named_parameters()}

        for epoch in range(self._epoch):
            for step, (x, y) in enumerate(train_loader):
                b_x, b_y = x.to(self._device), y.to(self._device)

                self.model.train()
                optimizer.zero_grad()
                output = self.model(b_x)
                loss = loss_func(output, b_y.long())
                loss.backward()

                if update_model:
                    optimizer.step()

                # 记录当前模型参数
                current_params = {name: param.detach().clone() for name, param in self.model.named_parameters()}

                # 计算梯度为上一参数减去当前参数
                step_grad = {}
                for name in current_params:
                    step_grad[name] = prev_params[name] - current_params[name]

                # BN的running_mean和running_var直接复制当前buffer
                for name, buf in self.model.named_buffers():
                    if "running_mean" in name or "running_var" in name or "num_batches_tracked" in name:
                        step_grad[name] = buf.detach().clone()

                grad_list.append(step_grad)

                # 更新prev_params为当前参数，用于下一step计算梯度
                prev_params = current_params

        return grad_list, loss

    def train(self, global_avg_grad=None):
        """
        MAGS client local training with global gradient guidance (including BN mean/var):
        - 记录每一步的梯度和BN缓冲
        - 计算本地平均梯度 avg_grad
        - 根据本地 avg_grad 和 global_avg_grad 排序
        - 选出比例前select_ratio的梯度（取并集）
        - 对选中梯度加和
        :return: final_grad, 样本数, loss, avg_grad
        """

        grad_list, loss = self.collect_grads(update_model=True)

        # 1. 计算本地平均梯度
        avg_grad = {name: sum([g[name] for g in grad_list]) / len(grad_list)
                    for name in grad_list[0]}

        # 2. 排序
        local_indices_1 = self.select_grads(grad_list, None, self.select_ratio)

        # 打印被选中的梯度的norm
        # device = next(iter(avg_grad.values())).device
        # flat_avg = self.flatten_grad(avg_grad, device)
        # for idx in local_indices:
        #     g = self.flatten_grad(grad_list[idx],device)
        #     norm = (g - flat_avg).to(device)
        #     print(f"Index {idx}, Norm = {torch.norm(norm).item():.6f}")
        # print("next client")

        # local_indices_2 =  self.select_grads_topk(grad_list, avg_grad, self.select_ratio)

        # 3. 聚合 local_indices 和 global_indices 对应的梯度（包括 BN mean/var）
        selected_indices = list(local_indices_1)
        # start_index = round(len(grad_list) * self.select_ratio)
        # selected_indices = range(start_index, len(grad_list))
        selected_grads = [grad_list[i] for i in selected_indices]

        # ---------- 新增：未被选中的梯度集合 ----------
        unselected_indices = [i for i in range(len(grad_list)) if i not in selected_indices]
        unselected_grads = [grad_list[i] for i in unselected_indices]

        indices_division = [selected_indices, unselected_indices]

        # ---------- Step 1: 计算 selected_grads 的和 ----------
        selected_sum = {name: sum([g[name] for g in selected_grads])
                        for name in selected_grads[0]}

        # ---------- Step 2: 计算 unselected_grads 的加权和 ----------
        if len(unselected_grads) > 0:
            unselected_sum = {name: self.beta * sum([g[name] for g in unselected_grads])
                              for name in unselected_grads[0]}
        else:
            # 若全部被选中，则未选集合为0
            unselected_sum = {name: torch.zeros_like(selected_sum[name])
                              for name in selected_sum}

        # ---------- Step 3: 聚合两者 ----------
        combined_sum = {name: selected_sum[name] + unselected_sum[name]
                        for name in selected_sum}

        # ---------- Step 4: 计算总梯度（用于归一化） ----------
        total_sum = {name: sum([g[name] for g in grad_list])
                     for name in grad_list[0]}

        flat_combined_sum = self.flatten_grad(combined_sum, self._device)
        flat_total_sum = self.flatten_grad(total_sum, self._device)

        # ---------- Step 5: 计算归一化比例 ----------
        ratio = torch.norm(flat_combined_sum) / torch.norm(flat_total_sum)

        # ---------- Step 6: 最终聚合 ----------
        final_grad = {}
        state_dict = self.model.state_dict()
        for name in combined_sum:
            if "running_mean" in name or "running_var" in name or "num_batches_tracked" in name:
                final_grad[name] = state_dict[name]  # BN统计量直接取当前状态
            else:
                final_grad[name] = combined_sum[name] / ratio

        return final_grad, self.n_data, loss.item(), indices_division


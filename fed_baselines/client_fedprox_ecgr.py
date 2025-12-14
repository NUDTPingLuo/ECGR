from fed_baselines.client_base import FedClient
import copy
from utils.models import *

from torch.utils.data import DataLoader


class FedProxECGRClient(FedClient):
    def __init__(self, name, epoch, dataset_id, model_name, lr, batch_size, momentum, beta):
        super().__init__(name, epoch, dataset_id, model_name, lr, batch_size, momentum)
        self.mu = 0.1
        self.select_ratio = 0.5
        self.beta = beta

    def flatten_grad(self, grad_dict):
        """把梯度字典展平为一个大向量，排除BN统计量"""
        flat_list = []
        for name, v in grad_dict.items():
            if ("mean" not in name.lower()
                    and "var" not in name.lower()
                    and "num_batches_tracked" not in name):
                flat_list.append(v.flatten().to(self._device))
        return torch.cat(flat_list) if flat_list else torch.tensor([], device=self._device)

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
        flat_avg = self.flatten_grad(avg_grad)

        # 展平梯度并计算与平均梯度的差
        flat_grads = torch.stack([self.flatten_grad(g) for g in grad_list])
        diffs = flat_grads - flat_avg

        # 计算 L2 距离
        distances = torch.norm(diffs, dim=1)

        # 取距离最小的前 num_select 个
        num_select = max(1, int(len(grad_list) * select_ratio))
        _, selected_indices = torch.topk(-distances, num_select)  # 负号表示取最小值

        return selected_indices.tolist()
    def select_grads(self, local_grad_list, avg_grad, select_ratio):
        """
        贪心选择与当前总梯度 S 最近的梯度
        :param local_grad_list: list，每一步的梯度字典
        :param avg_grad: dict，平均梯度
        :param select_ratio: float，选择比例
        :return: selected_indices (list)
        """
        if avg_grad is None:
            avg_grad = {name: torch.zeros_like(param, device=self._device) for name, param in local_grad_list[0].items()}

        # 1. 展平梯度
        flat_avg = self.flatten_grad(avg_grad)
        flat_grads = [self.flatten_grad(g) for g in local_grad_list]

        # 2. 归一化梯度
        normalized = [(g - flat_avg) for g in flat_grads]

        # 3. 初始化 S 为全零
        S = torch.zeros_like(flat_avg, device=self._device)

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

    def train(self):
        """
        Client trains the model on local dataset using FedProx
        :return: Local updated model, number of local data points, training loss
        """
        train_loader = DataLoader(self.trainset, batch_size=self._batch_size, shuffle=True, num_workers=8, pin_memory=True, persistent_workers=False)

        self.model.to(self._device)
        # global_weights = copy.deepcopy(list(self.model.parameters()))
        global_weights = {k: v.detach().clone().to(self._device) for k, v in self.model.state_dict().items()}

        optimizer = torch.optim.SGD(self.model.parameters(), lr=self._lr, momentum=self._momentum)
        # optimizer = torch.optim.Adam(self.model.parameters(), lr=self._lr, weight_decay=1e-4)
        loss_func = nn.CrossEntropyLoss()

        epoch_loss_collector = []
        grad_list = []  # 用于存储每轮梯度

        local_state_dict_end = copy.deepcopy(self.model.state_dict())
        global_state_dict = copy.deepcopy(self.model.state_dict())

        # pbar = tqdm(range(self._epoch))
        for epoch in range(self._epoch):
            for step, (x, y) in enumerate(train_loader):
                # 记录前一步的参数
                params_before = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
                with torch.no_grad():
                    b_x = x.to(self._device)  # Tensor on GPU
                    b_y = y.to(self._device)  # Tensor on GPU

                with torch.enable_grad():
                    self.model.train()
                    output = self.model(b_x)
                    loss = loss_func(output, b_y.long())
                    optimizer.zero_grad()

                    #fedprox
                    prox_term = 0.0
                    for name, param in self.model.named_parameters():
                        diff = param - global_weights[name]
                        prox_term += (self.mu / 2) * torch.sum(diff * diff)
                    loss += prox_term

                    loss.backward()
                    optimizer.step()

                    # 记录本轮训练结束后的参数
                    params_after = {k: v.detach().clone() for k, v in self.model.state_dict().items()}

                    # 计算本轮梯度（前参数 - 后参数）
                    step_grad = {}
                    for key in params_before:
                        step_grad[key] = params_before[key] - params_after[key]
                    grad_list.append(step_grad)

        avg_grad = {}
        state_dict = self.model.state_dict()
        for name in grad_list[0]:
            if "running_mean" in name or "running_var" in name or "num_batches_tracked" in name:
                # BN buffer 直接取 state_dict 中的值
                avg_grad[name] = state_dict[name].clone()
            else:
                # 可训练参数做平均梯度
                avg_grad[name] = sum([g[name] for g in grad_list]) / len(grad_list)

        # ---------- Step 2: MAGS 选择 ----------
        local_indices_1 = self.select_grads(grad_list, None, self.select_ratio)
        # local_indices_2 = self.select_grads(grad_list, avg_grad, self.select_ratio)

        selected_indices = list(local_indices_1)
        # start_index = round(len(grad_list) * self.select_ratio)
        # selected_indices = range(start_index, len(grad_list))
        selected_grads = [grad_list[i] for i in selected_indices]

        # ---------- 新增：未被选中的梯度集合 ----------
        unselected_indices = [i for i in range(len(grad_list)) if i not in selected_indices]
        unselected_grads = [grad_list[i] for i in unselected_indices]

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

        flat_combined_sum = self.flatten_grad(combined_sum)
        flat_total_sum = self.flatten_grad(total_sum)

        # ---------- Step 5: 计算归一化比例 ----------
        ratio = torch.norm(flat_combined_sum) / torch.norm(flat_total_sum)

        # ---------- Step 3: 聚合 ----------
        final_grad = {}
        for name in state_dict:
            if "running_mean" in name or "running_var" in name or "num_batches_tracked" in name:
                # BN buffer 直接取 state_dict 中的值
                final_grad[name] = state_dict[name]
                local_state_dict_end[name] = state_dict[name]
            else:
                final_grad[name] = combined_sum[name] / ratio
                local_state_dict_end[name] = global_state_dict[name] - final_grad[name]

        return local_state_dict_end, self.n_data, loss.data.cpu().numpy()
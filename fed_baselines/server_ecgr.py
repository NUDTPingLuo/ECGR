from fed_baselines.server_base import FedServer
import copy
import torch
import os
import csv


class ECGRServer(FedServer):
    def __init__(self, client_list, dataset_id, model_name, batch_size, select_ratio=0.5):
        super().__init__(client_list, dataset_id, model_name, batch_size)
        # self.select_ratio = select_ratio  # MAGS采样比例
        self.global_avg_grad = None       # 存储聚合后的全局平均梯度
        self.dataset_id = dataset_id

    def agg(self):
        """
        聚合客户端上传的 final_grad，同时计算全局平均梯度。
        使用 global_weights + scale * grad_sum 更新模型。
        """
        client_num = len(self.selected_clients)
        if client_num == 0 or self.n_data == 0:
            return self.model.state_dict(), 0, 0, self.global_avg_grad

        self.model.to(self._device)
        global_model = self.model.state_dict()
        new_model = copy.deepcopy(global_model)
        avg_loss = 0

        # 初始化 grad_sum 和 avg_grad_sum
        grad_sum = {}
        avg_grad_sum = {}

        # 假设当前文件在项目根目录下执行
        save_root = os.path.join(os.getcwd(), "results", "Gradient_Division", self.dataset_id)
        os.makedirs(save_root, exist_ok=True)

        # 聚合客户端上传的梯度
        for i, name in enumerate(self.selected_clients):
            client_grad = self.client_state[name]["final_grad"]
            client_indices_division = self.client_state[name]["indices_division"]
            weight = self.client_n_data[name] / self.n_data  # 加权

            # csv_path = os.path.join(save_root, f"{name}_division.csv")
            #
            # # 判断文件是否存在，决定是否写入表头
            # write_header = not os.path.exists(csv_path)
            #
            # # 追加写入
            # with open(csv_path, "a", newline="") as f:
            #     writer = csv.writer(f)
            #
            #     # 第一次创建文件才写表头
            #     if write_header:
            #         writer.writerow(["Round", "Selected", "Unselected"])
            #
            #     writer.writerow([self.round,
            #                      client_indices_division[0],
            #                      client_indices_division[1]])

            for key in client_grad:
                if i == 0:
                    grad_sum[key] = weight * client_grad[key].to(self._device)
                else:
                    grad_sum[key] += weight * client_grad[key].to(self._device)

            avg_loss += self.client_loss[name] * self.client_n_data[name] / self.n_data

        for key in global_model:
            if "running_mean" in key or "running_var" in key or "num_batches_tracked" in key:
                new_model[key] = grad_sum[key]  # 直接复制，不更新
            else:
                new_model[key] = global_model[key] - grad_sum[key]

        self.model.load_state_dict(new_model)
        self.round += 1

        return new_model, avg_loss, self.n_data

    def rec(self, name, final_grad, n_data, loss, indices_division):
        """
        接收客户端上传的 final_grad 和 avg_grad。
        """
        self.n_data += n_data
        self.client_state[name] = {"final_grad": final_grad, "indices_division": indices_division}
        self.client_n_data[name] = n_data
        self.client_loss[name] = loss

    def flush(self):
        """
        每轮通信前清空缓存。
        """
        self.n_data = 0
        self.client_state = {}
        self.client_n_data = {}
        self.client_loss = {}

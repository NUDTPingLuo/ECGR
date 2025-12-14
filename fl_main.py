#!/usr/bin/env python
import os
import random
import json
import pickle
import argparse
import yaml
from json import JSONEncoder
from tqdm import tqdm

from fed_baselines.client_base import FedClient
from fed_baselines.client_fedprox import FedProxClient
from fed_baselines.client_scaffold import ScaffoldClient
from fed_baselines.client_fednova import FedNovaClient
from fed_baselines.client_ecgr import ECGRClient
from fed_baselines.client_scaffold_ecgr import ScaffoldECGRClient
from fed_baselines.client_fednova_ecgr import FedNovaECGRClient
from fed_baselines.client_fedprox_ecgr import FedProxECGRClient
from fed_baselines.server_base import FedServer
from fed_baselines.server_scaffold import ScaffoldServer
from fed_baselines.server_fednova import FedNovaServer
from fed_baselines.server_ecgr import ECGRServer

from postprocessing.recorder import Recorder
from preprocessing.baselines_dataloader import divide_data, divide_data_dirichlet
from utils.models import *

json_types = (list, dict, str, int, float, bool, type(None))

print("CUDA是否可用:", torch.cuda.is_available())
print("可用的GPU数量:", torch.cuda.device_count())
print("当前默认GPU设备:", torch.cuda.current_device())
print("当前默认设备名称:",
      torch.cuda.get_device_name(torch.cuda.current_device()) if torch.cuda.is_available() else "无GPU")


class PythonObjectEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, json_types):
            return super().default(self, obj)
        return {'_python_object': pickle.dumps(obj).decode('latin-1')}


def as_python_object(dct):
    if '_python_object' in dct:
        return pickle.loads(dct['_python_object'].encode('latin-1'))
    return dct


def fed_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Yaml file for configuration')
    args = parser.parse_args()
    return args


def fed_run():
    args = fed_args()
    with open(args.config, "r") as yaml_file:
        try:
            config = yaml.safe_load(yaml_file)
        except yaml.YAMLError as exc:
            print(exc)

    # ✅ 支持的算法
    algo_list = ["FedAvg", "Scaffold", "FedProx", "FedNova", "FedAvgECGR", "ScaffoldECGR", "FedProxECGR", "FedNovaECGR"]
    assert config["client"]["fed_algo"] in algo_list, "The federated learning algorithm is not supported"

    dataset_list = ['MNIST', 'CIFAR10', 'FashionMNIST', 'SVHN', 'CIFAR100', 'IMAGENET', 'LC25000']
    assert config["system"]["dataset"] in dataset_list, "The dataset is not supported"

    model_list = ["LeNet",'CNN_MNIST', 'AlexCifarNet', "ResNet18", 'VGG11', "CNN", 'ResNet18_LC25000']
    assert config["system"]["model"] in model_list, "The model is not supported"

    np.random.seed(config["system"]["i_seed"])
    torch.manual_seed(config["system"]["i_seed"])
    random.seed(config["system"]["i_seed"])

    client_dict = {}
    recorder = Recorder()

    trainset_config, testset = divide_data_dirichlet(num_client=config["system"]["num_client"],
                                                     alpha=config["system"]["dirichlet_alpha"],
                                                     dataset_name=config["system"]["dataset"],
                                                     i_seed=config["system"]["i_seed"])
    max_acc = 0

    # ✅ 初始化 Client
    for client_id in trainset_config['users']:
        if config["client"]["fed_algo"] == 'FedAvg':
            client_dict[client_id] = FedClient(client_id, dataset_id=config["system"]["dataset"],
                                               epoch=config["client"]["num_local_epoch"],
                                               model_name=config["system"]["model"],
                                               lr=config["client"]["lr"],
                                               batch_size=config["client"]["batch_size"],
                                               momentum=config["client"]["momentum"]
                                               )
        elif config["client"]["fed_algo"] == 'FedAvgECGR':
            client_dict[client_id] = ECGRClient(client_id, dataset_id=config["system"]["dataset"],
                                                 epoch=config["client"]["num_local_epoch"],
                                                 model_name=config["system"]["model"],
                                                lr=config["client"]["lr"],
                                                batch_size=config["client"]["batch_size"],
                                                momentum=config["client"]["momentum"],
                                                beta=config["system"]["extraction_beta"]
                                                )
        elif config["client"]["fed_algo"] == 'Scaffold':
            client_dict[client_id] = ScaffoldClient(client_id, dataset_id=config["system"]["dataset"],
                                                    epoch=config["client"]["num_local_epoch"],
                                                    model_name=config["system"]["model"],
                                                    lr=config["client"]["lr"],
                                                    batch_size=config["client"]["batch_size"],
                                                    momentum=config["client"]["momentum"]
                                                    )
        elif config["client"]["fed_algo"] == 'ScaffoldECGR':
            client_dict[client_id] = ScaffoldECGRClient(client_id, dataset_id=config["system"]["dataset"],
                                                         epoch=config["client"]["num_local_epoch"],
                                                         model_name=config["system"]["model"],
                                                        lr=config["client"]["lr"],
                                                        batch_size=config["client"]["batch_size"],
                                                        momentum=config["client"]["momentum"],
                                                        beta=config["system"]["extraction_beta"]
                                                        )
        elif config["client"]["fed_algo"] == 'FedProx':
            client_dict[client_id] = FedProxClient(client_id, dataset_id=config["system"]["dataset"],
                                                   epoch=config["client"]["num_local_epoch"],
                                                   model_name=config["system"]["model"],
                                                   lr=config["client"]["lr"],
                                                   batch_size=config["client"]["batch_size"],
                                                   momentum=config["client"]["momentum"]
                                                   )
        elif config["client"]["fed_algo"] == 'FedProxECGR':
            client_dict[client_id] = FedProxECGRClient(client_id, dataset_id=config["system"]["dataset"],
                                                   epoch=config["client"]["num_local_epoch"],
                                                   model_name=config["system"]["model"],
                                                   lr=config["client"]["lr"],
                                                   batch_size=config["client"]["batch_size"],
                                                   momentum=config["client"]["momentum"],
                                                    beta=config["system"]["extraction_beta"]
                                                   )
        elif config["client"]["fed_algo"] == 'FedNova':
            client_dict[client_id] = FedNovaClient(client_id, dataset_id=config["system"]["dataset"],
                                                   epoch=config["client"]["num_local_epoch"],
                                                   model_name=config["system"]["model"],
                                                   lr=config["client"]["lr"],
                                                   batch_size=config["client"]["batch_size"],
                                                   momentum=config["client"]["momentum"]
                                                   )
        elif config["client"]["fed_algo"] == 'FedNovaECGR':
            client_dict[client_id] = FedNovaECGRClient(client_id, dataset_id=config["system"]["dataset"],
                                                   epoch=config["client"]["num_local_epoch"],
                                                   model_name=config["system"]["model"],
                                                   lr=config["client"]["lr"],
                                                   batch_size=config["client"]["batch_size"],
                                                   momentum=config["client"]["momentum"],
                                                   beta=config["system"]["extraction_beta"]
                                                   )
        client_dict[client_id].load_trainset(trainset_config['user_data'][client_id])

    # ✅ 初始化 Server
    if config["client"]["fed_algo"] == 'FedAvg':
        fed_server = FedServer(trainset_config['users'], dataset_id=config["system"]["dataset"],
                               model_name=config["system"]["model"],
                               batch_size=config["client"]["batch_size"])
    elif config["client"]["fed_algo"] == 'FedAvgECGR':
        fed_server = ECGRServer(trainset_config['users'], dataset_id=config["system"]["dataset"],
                                 model_name=config["system"]["model"],
                                batch_size=config["client"]["batch_size"])
    elif config["client"]["fed_algo"] == 'Scaffold' or config["client"]["fed_algo"] == 'ScaffoldECGR':
        fed_server = ScaffoldServer(trainset_config['users'], dataset_id=config["system"]["dataset"],
                                    model_name=config["system"]["model"],
                                    batch_size=config["client"]["batch_size"])
        scv_state = fed_server.scv.state_dict()
    elif config["client"]["fed_algo"] == 'FedProx' or config["client"]["fed_algo"] == 'FedProxECGR':
        fed_server = FedServer(trainset_config['users'], dataset_id=config["system"]["dataset"],
                               model_name=config["system"]["model"],
                               batch_size=config["client"]["batch_size"])
    elif config["client"]["fed_algo"] == 'FedNova' or config["client"]["fed_algo"] == 'FedNovaECGR':
        fed_server = FedNovaServer(trainset_config['users'], dataset_id=config["system"]["dataset"],
                                   model_name=config["system"]["model"],
                                   batch_size=config["client"]["batch_size"])



    fed_server.load_testset(testset)
    global_state_dict = fed_server.state_dict()

    # ✅ 主训练循环
    pbar = tqdm(range(config["system"]["num_round"]))
    for global_round in pbar:
        for client_id in trainset_config['users']:
            if config["client"]["fed_algo"] == 'FedAvg':
                client_dict[client_id].update(global_round, global_state_dict)
                state_dict, n_data, loss = client_dict[client_id].train()
                fed_server.rec(client_dict[client_id].name, state_dict, n_data, loss)

            elif config["client"]["fed_algo"] == 'FedAvgECGR':
                client_dict[client_id].update(global_round, global_state_dict)
                global_avg_grad = getattr(fed_server, "global_avg_grad", None)
                final_grad, n_data, loss, indices_division = client_dict[client_id].train(global_avg_grad=global_avg_grad)
                fed_server.rec(client_dict[client_id].name, final_grad, n_data, loss, indices_division)

            elif config["client"]["fed_algo"] == 'Scaffold' or config["client"]["fed_algo"] == 'ScaffoldECGR':
                client_dict[client_id].update(global_round, global_state_dict, scv_state)
                state_dict, n_data, loss, delta_ccv_state = client_dict[client_id].train()
                fed_server.rec(client_dict[client_id].name, state_dict, n_data, loss, delta_ccv_state)

            elif config["client"]["fed_algo"] == 'FedProx' or config["client"]["fed_algo"] == 'FedProxECGR':
                client_dict[client_id].update(global_round, global_state_dict)
                state_dict, n_data, loss = client_dict[client_id].train()
                fed_server.rec(client_dict[client_id].name, state_dict, n_data, loss)

            elif config["client"]["fed_algo"] == 'FedNova' or config["client"]["fed_algo"] == 'FedNovaECGR':
                client_dict[client_id].update(global_round, global_state_dict)
                state_dict, n_data, loss, coeff, norm_grad = client_dict[client_id].train()
                fed_server.rec(client_dict[client_id].name, state_dict, n_data, loss, coeff, norm_grad)

        # ✅ 聚合
        fed_server.select_clients()
        if config["client"]["fed_algo"] == 'FedAvg':
            global_state_dict, avg_loss, _ = fed_server.agg()
        elif config["client"]["fed_algo"] == 'FedAvgECGR':
            global_state_dict, avg_loss, _ = fed_server.agg()
        elif config["client"]["fed_algo"] == 'Scaffold' or config["client"]["fed_algo"] == 'ScaffoldECGR':
            global_state_dict, avg_loss, _, scv_state = fed_server.agg()
        elif config["client"]["fed_algo"] == 'FedProx' or config["client"]["fed_algo"] == 'FedProxECGR':
            global_state_dict, avg_loss, _ = fed_server.agg()
        elif config["client"]["fed_algo"] == 'FedNova' or config["client"]["fed_algo"] == 'FedNovaECGR':
            global_state_dict, avg_loss, _ = fed_server.agg()

        # ✅ 测试 & 清空
        accuracy = fed_server.test()
        fed_server.flush()

        recorder.res['server']['iid_accuracy'].append(accuracy)
        recorder.res['server']['train_loss'].append(avg_loss)

        if max_acc < accuracy:
            max_acc = accuracy
        pbar.set_description(
            f'Global Round: {global_round} | Train loss: {avg_loss:.4f} | Accuracy: {accuracy:.4f} | Max Acc: {max_acc:.4f}')

        # ✅ 保存结果
        if not os.path.exists(config["system"]["res_root"]):
            os.makedirs(config["system"]["res_root"])

        fed_algo = config["client"]["fed_algo"]
        dataset = config["system"]["dataset"]
        lr = config["client"]["lr"]
        alpha = config["system"]["dirichlet_alpha"]
        beta = config["system"]["extraction_beta"]
        random_seed = config["system"]["i_seed"]

        # 标准字符串拼接方式：每个字段都加单引号并用逗号隔开
        file_name = f"['{fed_algo}','{dataset}','alpha{alpha}','lr{lr}','beta{beta}','seed{random_seed}']"

        # 拼接完整路径
        save_dir = os.path.join(config["system"]["res_root"],
                                 config["system"]["dataset"],
                                "alpha" + str(config["system"]["dirichlet_alpha"]),
                                "lr" + str(config["client"]["lr"]),
                                "beta" + str(config["system"]["extraction_beta"]),
                                 config["client"]["fed_algo"].replace("ECGR", "")
                                 )
        # save_dir = os.path.join(config["system"]["res_root"])

        os.makedirs(save_dir, exist_ok=True)
        file_path = os.path.join(save_dir, file_name)
        # 打开文件并写入
        with open(file_path, "w") as jsfile:
            json.dump(recorder.res, jsfile, cls=PythonObjectEncoder)


if __name__ == "__main__":
    fed_run()

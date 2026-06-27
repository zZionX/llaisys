import os
from glob import glob
import torch
from torch import nn
from safetensors import safe_open

def default_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor):
    param.data.copy_(loaded_weight)

def load_model(model, checkpoint_path):
    packed_modules_mapping = getattr(model, 'packed_modules_mapping', {})

    for file in glob(os.path.join(checkpoint_path, '*.safetensors')):
        with safe_open(file, framework="pt", device="cpu") as f: # f是个句柄对象，数据还在硬盘上，safe_open 按需读取，读一个张量加载一个，极大地节省了内存
            for weight_name in f.keys(): # f.keys()是一个迭代器，包含该文件中所有张量的名字，eg. ['model.layers.0.q_proj.weight', 'model.layers.0.k_proj.weight', 'model.embed_tokens.weight', ...]
                for k in packed_modules_mapping: # dict的key
                    if k in weight_name:
                        # 获取目标参数名和分片ID
                        # 例如 k="q_proj" -> v="qkv_proj", shard_id="q"
                        v, shard_id = packed_modules_mapping[k]
                        # 把名字替换一下：model.layer.0.q_proj.weight -> model.layer.0.qkv_proj.weight
                        param_name = weight_name.replace(k, v)
                        # 从模型中拿到那个拼合后的大参数对象
                        param = model.get_parameter(param_name)
                        # 获取该参数绑定的自定义加载函数
                        weight_loader = getattr(param, 'weight_loader')
                        # 调用自定义加载函数，把当前读出的小张量，作为分片加载进大张量中
                        weight_loader(param,f.get_tensor(weight_name), shard_id)
                        break
                else:
                    param = model.get_parameter(weight_name)
                    weight_loader = getattr(param, 'weight_loader', default_weight_loader)
                    weight_loader(param, f.get_tensor(weight_name))

import json
import os
from ctypes import byref, c_int, c_int64, c_size_t
from pathlib import Path
from typing import Sequence


import safetensors

from ..libllaisys import (
    LIB_LLAISYS,
    DeviceType,
    Qwen2MetaCStruct,
    Qwen2WeightsCStruct,
    Qwen2WeightsNaming,
)


class Qwen2:
    def __init__(self, model_path, device: DeviceType = DeviceType.CPU):
        model_path = Path(model_path)

        # load config
        print("Qwen2: loading configs...", flush=True)
        with open(os.path.join(model_path, "config.json"), "r") as f:
            config = json.load(f)
            self.config = config

        # load weights
        print("Qwen2: loading weights...", flush=True)
        for file in sorted(model_path.glob("*.safetensors")):
            state_dict = {}
            data_ = safetensors.safe_open(file, framework="pytorch", device="cpu")
            for name_ in data_.keys():
                state_dict[name_] = data_.get_tensor(name_)

        # create model
        naming = Qwen2WeightsNaming()
        if naming.match(state_dict):
            print("Qwen2: creating model...", flush=True)
            ndev = 1
            dev_ids = (c_int * ndev)(*[i for i in range(ndev)])
            self.meta = Qwen2MetaCStruct(config)
            self.weights = Qwen2WeightsCStruct(self.meta, state_dict, naming, ndev)
            self.model = LIB_LLAISYS.llaisysQwen2ModelCreate(
                byref(self.meta), byref(self.weights), device, ndev, dev_ids
            )
            self.weights.release()
            print("Qwen2: create model ok!!!", flush=True)
        else:
            raise ValueError("state_dict fail weights name compare")

    def generate(
        self,
        inputs: Sequence[int],
        max_new_tokens: int = None,
        top_k: int = 1,
        top_p: float = 0.8,
        temperature: float = 0.8,
    ):
        tokens = list(inputs)
        max_len = len(tokens) + max_new_tokens
        kvcache = LIB_LLAISYS.llaisysQwen2KVCacheCreate(self.model, max_len)

        # prefill
        print("Qwen2: prefilling...", flush=True)
        ntoken = len(tokens)
        token_ids = (c_int64 * ntoken)(*tokens)
        past_len = c_size_t(0)
        next_token = LIB_LLAISYS.llaisysQwen2ModelInfer(
            self.model, token_ids, c_size_t(ntoken), kvcache, past_len
        )
        tokens.append(next_token)
        print("current tokens: ", tokens, flush=True)

        # decode
        print("Qwen2: decoding...", flush=True)
        for _ in range(max_new_tokens):
            if next_token == self.meta.end_token:
                break
            ntoken = 1
            token_ids = (c_int64 * 1)(next_token)
            past_len = c_size_t(len(tokens) - 1)
            next_token = LIB_LLAISYS.llaisysQwen2ModelInfer(
                self.model, token_ids, ntoken, kvcache, past_len
            )
            tokens.append(next_token)
            print("current tokens: ", tokens, flush=True)

        nlayer = self.meta.nlayer
        LIB_LLAISYS.llaisysQwen2KVCacheDestroy(kvcache, nlayer)

        return tokens

    def __del__(self):
        LIB_LLAISYS.llaisysQwen2ModelDestroy(self.model)

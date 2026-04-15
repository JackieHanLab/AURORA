import torch
from typing import List, Mapping, Optional,Any
from anndata import AnnData

import torch.distributions as D
import torch.nn.functional as F
import collections
import numpy as np
import pathlib
from torch.optim.lr_scheduler import ReduceLROnPlateau
from itertools import chain

import pandas as pd
import copy
from math import ceil
import dill
from abc import abstractmethod
import os

from .trainer import Trainer,DataLoader
from .utils import config, get_chained_attr, autodevice
from .dataset import AnnDataset,ArrayDataset

EPS = 1e-7
AUTO = -1

class FeatureEncoder(torch.nn.Module):
    def __init__(
            self, vnum: int, out_features: int
    ) -> None:
        super().__init__()
        self.num_heads = 4
        self.vrepr = torch.nn.Parameter(torch.zeros(vnum, out_features))
        self.qkv = torch.nn.Linear(out_features, out_features * 3, bias=True)
        self.mha = torch.nn.MultiheadAttention(out_features,self.num_heads,dropout=0.2)
        self.loc = torch.nn.Linear(out_features, out_features)
        self.std_lin = torch.nn.Linear(out_features, out_features)
        self.device = self.vrepr.device
    
    def attention(self, vrepr, get_att = False):
        N, C = vrepr.shape
        qkv = self.qkv(vrepr).reshape(N, 3, C).permute(1, 0, 2)
        q, k, v = qkv[0], qkv[1], qkv[2]
        ptr,att = self.mha(q,k,v,need_weights=get_att)
        loc = self.loc(ptr)
        std = F.softplus(self.std_lin(ptr)) + EPS
        if get_att:
            return D.Normal(loc, std),att
        else:
            return D.Normal(loc, std)
    
    def forward(self,get_att = False) -> D.Normal:
        vrepr = self.vrepr
        return self.attention(vrepr, get_att)

class DataEncoder(torch.nn.Module):
    def __init__(
            self, in_features: int, out_features: int,
            h_depth: int = 2, h_dim: int = 256,
            dropout: float = 0.2
    ) -> None:
        super().__init__()
        self.h_depth = h_depth
        ptr_dim = in_features
        for layer in range(self.h_depth):
            setattr(self, f"linear_{layer}", torch.nn.Linear(ptr_dim, h_dim))
            setattr(self, f"act_{layer}", torch.nn.LeakyReLU(negative_slope=0.2))
            setattr(self, f"bn_{layer}", torch.nn.BatchNorm1d(h_dim))
            setattr(self, f"dropout_{layer}", torch.nn.Dropout(p=dropout))
            ptr_dim = h_dim
        self.loc = torch.nn.Linear(ptr_dim, out_features)
        self.std_lin = torch.nn.Linear(ptr_dim, out_features)

    def forward(  # pylint: disable=arguments-differ
            self, x: torch.Tensor, xalt: torch.Tensor
    ) -> D.Normal:
        if xalt.numel():
            ptr = xalt
        else:
            ptr = x
        for layer in range(self.h_depth):
            ptr = getattr(self, f"linear_{layer}")(ptr)
            ptr = getattr(self, f"act_{layer}")(ptr)
            ptr = getattr(self, f"bn_{layer}")(ptr)
            ptr = getattr(self, f"dropout_{layer}")(ptr)
        loc = self.loc(ptr)
        std = F.softplus(self.std_lin(ptr)) + EPS
        return D.Normal(loc, std)


class DataDecoder(torch.nn.Module):
    def __init__(self, out_features: int) -> None:
        super().__init__()
        self.scale_lin = torch.nn.Parameter(torch.zeros(out_features))
        self.bias = torch.nn.Parameter(torch.zeros(out_features))
        self.std_lin = torch.nn.Parameter(torch.zeros(out_features))

    def forward(
            self, u: torch.Tensor, v: torch.Tensor,
    ) -> D.Normal:
        scale = F.softplus(self.scale_lin)
        loc = scale * (u @ v.t()) + self.bias
        std = F.softplus(self.std_lin) + EPS
        return D.Normal(loc, std)

class Discriminator(torch.nn.Sequential, torch.nn.Module):
    def __init__(
            self, 
            in_features: int, 
            out_features: int,
            h_depth: int = 2, 
            h_dim: Optional[int] = 256,
            dropout: float = 0.2
    ) -> None:
        od = collections.OrderedDict()
        ptr_dim = in_features
        for layer in range(h_depth):
            od[f"linear_{layer}"] = torch.nn.Linear(ptr_dim, h_dim)
            od[f"act_{layer}"] = torch.nn.LeakyReLU(negative_slope=0.2)
            od[f"dropout_{layer}"] = torch.nn.Dropout(p=dropout)
            ptr_dim = h_dim
        od["pred"] = torch.nn.Linear(ptr_dim, out_features)
        super().__init__(od)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x)
    
class Prior(torch.nn.Module):
    def __init__(
            self, loc: float = 0.0, std: float = 1.0
    ) -> None:
        super().__init__()
        loc = torch.as_tensor(loc, dtype=torch.get_default_dtype())
        std = torch.as_tensor(std, dtype=torch.get_default_dtype())
        self.register_buffer("loc", loc)
        self.register_buffer("std", std)

    def forward(self) -> D.Normal:
        return D.Normal(self.loc, self.std)
    
class BaseModel(torch.nn.Module):
    def __init__(
            self, 
            f2u,
            x2z,
            z2x,
            idx,
            du,
            prior
    ) -> None:
        super().__init__()
        
        self.keys = list(idx.keys())  # Keeps a specific order
        self.idx = idx
        self.f2u = f2u
        self.x2z = torch.nn.ModuleDict(x2z)
        self.z2x = torch.nn.ModuleDict(z2x)
        for k, v in idx.items():  # Since there is no BufferList
            self.register_buffer(f"{k}_idx", v)
        self.du = du
        self.prior = prior
        self.device = autodevice()

    @property
    def device(self) -> torch.device:
        return self._device

    @device.setter
    def device(self, device: torch.device) -> None:
        self._device = device
        self.to(self._device)
    
    def forward(self) -> None:
        return None

class AuroraModel(torch.nn.Module):
    def __init__(self, 
                 adatas: Mapping[str, AnnData],
                 vertices: List[str], 
                 latent_dim: int = 48,
                 h_depth: int = 2, 
                 h_dim: int = 256,
                 dropout: float = 0.2, 
                 random_seed: int = 0
                 ) -> None:
        super().__init__()
        self.vertices = pd.Index(vertices)
        self.random_seed = random_seed
        torch.manual_seed(self.random_seed)
        f2u = FeatureEncoder(self.vertices.size, latent_dim)
        self.domains, idx, x2z, z2x = {}, {}, {}, {}
        all_labels = set()
        for k, adata in adatas.items():
            data_config = copy.deepcopy(adata.uns[config.ANNDATA_KEY])
            idx[k] = self.vertices.get_indexer(data_config["features"]).astype(np.int64)
            idx[k] = torch.as_tensor(idx[k])
            x2z[k] = DataEncoder(
                in_features=data_config["rep_dim"] or len(data_config["features"]), 
                out_features=latent_dim,
                h_depth=h_depth, 
                h_dim=h_dim, 
                dropout=dropout
            )
            data_config["batches"] = pd.Index([])
            z2x[k] = DataDecoder(
                out_features= len(data_config["features"])
            )
            all_labels = all_labels.union(
                set() if data_config["labels"] is None
                else data_config["labels"]
            )
            self.domains[k] = data_config
        all_labels = pd.Index(all_labels).sort_values()
        for domain in self.domains.values():
            domain["labels"] = all_labels
        du = Discriminator(
            in_features=latent_dim, 
            out_features=len(self.domains),
            h_depth=h_depth, h_dim=h_dim, dropout=dropout
        )
        prior = Prior()
        self._net = BaseModel(f2u,x2z, z2x, idx, du, prior)
        self.device = autodevice()
    @property
    def device(self) -> torch.device:
        return self._device

    @device.setter
    def device(self, device: torch.device) -> None:
        self._device = device
        self.to(self._device)
    
    @property
    def net(self) -> torch.nn.Module:
        return self._net
    
    def adopt_pretrained_model(
            self, source: "AuroraModel", submodule: Optional[str] = None
    ) -> None:
        source, target = source.net, self.net
        if submodule:
            source = get_chained_attr(source, submodule)
            target = get_chained_attr(target, submodule)
        for k, t in chain(target.named_parameters(), target.named_buffers()):
            try:
                s = get_chained_attr(source, k)
            except AttributeError:
                continue
            if isinstance(t, torch.nn.Parameter):
                t = t.data
            if isinstance(s, torch.nn.Parameter):
                s = s.data
            if s.shape != t.shape:
                continue
            s = s.to(device=t.device, dtype=t.dtype)
            t.copy_(s)

    @property
    def trainer(self) -> Trainer:
        if self._trainer is None:
            raise RuntimeError(
                "No trainer has been registered! "
                "Please call `.compile()` first."
            )
        return self._trainer

    def compile(self,lam_data: float = 1.0,
            lam_kl: float = 1.0,
            lam_feature: float = 0.02,
            lr: float = 2e-3, **kwargs) -> None:
        self._trainer = Trainer(self.net, 
                                lam_data=lam_data, 
                                lam_kl=lam_kl,
                                lam_feature = lam_feature,
            optim="RMSprop", lr=lr,**kwargs)

    def fit(self, adatas: Mapping,
            val_split: float = 0.1,
            data_batch_size: int = 128,
            align_burnin: int = AUTO, 
            safe_burnin: bool = True,
            max_epochs: int = AUTO, 
            patience: Optional[int] = AUTO,
            reduce_lr_patience: Optional[int] = AUTO,
            wait_n_lrs: int = 1, directory: Optional[os.PathLike] = None
    ) -> None:
        data = AnnDataset(
            [adatas[key] for key in self.net.keys],
            [self.domains[key] for key in self.net.keys],
            mode="train"
        )
        batch_per_epoch = data.size * (1 - val_split) / data_batch_size
        if align_burnin == AUTO:
            align_burnin = max(
                        ceil(4000 / batch_per_epoch),
                        8
            )
        max_epochs = max(
                ceil(4000 / batch_per_epoch),
                48
        )
        patience = max(
                ceil(2000 / batch_per_epoch),
                4
        )
        reduce_lr_patience = max(
                ceil(1000 / batch_per_epoch),
                2
        )
        self.trainer.fit(data, val_split=val_split,
            data_batch_size=data_batch_size,
            align_burnin=align_burnin, safe_burnin=safe_burnin,
            max_epochs=max_epochs, patience=patience,
            reduce_lr_patience=reduce_lr_patience, wait_n_lrs=wait_n_lrs,
            random_seed=self.random_seed,
            directory=directory)
    
    @torch.no_grad()
    def encode_feature(
            self,
            get_att = False
    ) -> np.ndarray:
        self.net.eval()
        if get_att:
            v,att = self.net.f2u(get_att = get_att)
            return v.mean.detach().cpu().numpy(), att.detach().cpu().numpy()
        else:
            v = self.net.f2u(get_att = get_att)
            return v.mean.detach().cpu().numpy()

    @torch.no_grad()
    def encode_data(
            self, key: str, adata, batch_size: int = 128
    ) -> np.ndarray:
        self.net.eval()
        encoder = self.net.x2z[key]
        data = AnnDataset(
            [adata], [self.domains[key]],
            mode="eval", getitem_size=batch_size
        )
        data_loader = DataLoader(
            data, batch_size=1, shuffle=False,
            num_workers=0,
            pin_memory=config.DATALOADER_PIN_MEMORY and not config.CPU_ONLY, drop_last=False,
            persistent_workers=False
        )
        result = []
        for x, xalt, *_ in data_loader:
            u = encoder(
                x.to(self.net.device, non_blocking=True),
                xalt.to(self.net.device, non_blocking=True)
            )
            result.append(u.mean.detach().cpu())
        return torch.cat(result).numpy()

    @torch.no_grad()
    def decode_data(
            self, target_key: str,
            adata: Any, 
            source_key: str = None,
            latent_key: str = None,
            batch_size: int = 128
    ) -> np.ndarray:
        net = self.net
        device = net.device
        net.eval()
        if source_key is None:
            if latent_key is None:
                if isinstance(adata, (np.ndarray, torch.Tensor)):
                    z = torch.Tensor(adata)
                else:
                    raise ValueError("Either source_key or latent_key must be provided, or adata must be a matrix or tensor.")
            elif latent_key in adata.obsm.keys():
                z = torch.Tensor(adata.obsm[latent_key])
            else:
                raise ValueError(f"Provided latent_key '{latent_key}' not found in adata.obsm.")
        else:
            z = self.encode_data(source_key, adata, batch_size=batch_size)
        u = self.encode_feature()
        u = torch.as_tensor(u, device=device)
        u = u[getattr(net, f"{target_key}_idx")]
        data = ArrayDataset(z,getitem_size=128)
        data_loader = DataLoader(
            data, batch_size=1, shuffle=False,
            num_workers=0,
            pin_memory=config.DATALOADER_PIN_MEMORY and not config.CPU_ONLY, drop_last=False,
            persistent_workers=False
        )
        decoder = net.z2x[target_key]

        result = []
        for z_ in data_loader:
            z_ = z_[0].to(device, non_blocking=True)
            result.append(decoder(z_, u).mean.detach().cpu())
        return torch.cat(result).numpy()

    def save(self, fname: os.PathLike) -> None:
        fname = pathlib.Path(fname)
        trainer_backup, self._trainer = self._trainer, None
        device_backup, self.net.device = self.net.device, torch.device("cpu")
        with fname.open("wb") as f:
            dill.dump(self, f, protocol=4, byref=False, recurse=True)
        self.net.device = device_backup
        self._trainer = trainer_backup

    @staticmethod
    def load(fname: os.PathLike) -> "AuroraModel":
        fname = pathlib.Path(fname)
        with fname.open("rb") as f:
            model = dill.load(f)
        model.net.device = autodevice()
        return model

def load_model(fname: str) -> AuroraModel:
    return AuroraModel.load(fname)
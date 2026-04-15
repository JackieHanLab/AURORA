from anndata import AnnData
from typing import Mapping, Optional,Any, List
from collections import defaultdict
import multiprocessing
from multiprocessing import Process
import signal
import os
import queue

import torch
import numpy as np
import pandas as pd
from math import ceil
import h5py
import copy
import scipy.sparse


from .utils import config,get_rs,get_default_numpy_dtype

processes: Mapping[int, Mapping[int, Process]] = defaultdict(dict)

class Dataset(torch.utils.data.Dataset):
    def __init__(self, getitem_size: int = 1) -> None:
        super().__init__()
        self.getitem_size = getitem_size
        self.shuffle_seed: Optional[int] = None
        self.seed_queue: Optional[multiprocessing.Queue] = None
        self.propose_queue: Optional[multiprocessing.Queue] = None
        self.propose_cache: Mapping[int, Any] = {}

    @property
    def has_workers(self) -> bool:
        self_processes = processes[id(self)]
        pl = bool(self_processes)
        sq = self.seed_queue is not None
        pq = self.propose_queue is not None
        if not pl == sq == pq:
            raise RuntimeError("Background shuffling seems broken!")
        return pl and sq and pq

    def prepare_shuffle(self, num_workers: int = 1, random_seed: int = 0) -> None:
        if self.has_workers:
            self.clean()
        self_processes = processes[id(self)]
        self.shuffle_seed = random_seed
        if num_workers:
            self.seed_queue = multiprocessing.Queue()
            self.propose_queue = multiprocessing.Queue()
            for i in range(num_workers):
                p = multiprocessing.Process(target=self.shuffle_worker)
                p.start()
                self_processes[p.pid] = p
                self.seed_queue.put(self.shuffle_seed + i)

    def shuffle(self) -> None:
        r"""
        Custom shuffling
        """
        if self.has_workers:
            self_processes = processes[id(self)]
            self.seed_queue.put(self.shuffle_seed + len(self_processes))  # Look ahead
            while self.shuffle_seed not in self.propose_cache:
                shuffle_seed, shuffled = self.propose_queue.get()
                self.propose_cache[shuffle_seed] = shuffled
            self.accept_shuffle(self.propose_cache.pop(self.shuffle_seed))
        else:
            self.accept_shuffle(self.propose_shuffle(self.shuffle_seed))
        self.shuffle_seed += 1

    def shuffle_worker(self) -> None:
        r"""
        Background shuffle worker
        """
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        while True:
            seed = self.seed_queue.get()
            if seed is None:
                self.propose_queue.put((None, os.getpid()))
                break
            self.propose_queue.put((seed, self.propose_shuffle(seed)))

    def propose_shuffle(self, seed: int) -> Any:
        r"""
        Propose shuffling using a given random seed

        Parameters
        ----------
        seed
            Random seed

        Returns
        -------
        shuffled
            Shuffled result
        """
        raise NotImplementedError  # pragma: no cover

    def accept_shuffle(self, shuffled: Any) -> None:
        r"""
        Accept shuffling result

        Parameters
        ----------
        shuffled
            Shuffled result
        """
        raise NotImplementedError  # pragma: no cover

    def clean(self) -> None:
        r"""
        Clean up multi-process resources used in custom shuffling
        """
        self_processes = processes[id(self)]
        if not self.has_workers:
            return
        for _ in self_processes:
            self.seed_queue.put(None)
        self.propose_cache.clear()
        while self_processes:
            try:
                first, second = self.propose_queue.get(
                    timeout=60
                )
            except queue.Empty:
                break
            if first is not None:
                continue
            pid = second
            self_processes[pid].join()
            del self_processes[pid]
        for pid in list(self_processes.keys()):  # If some background processes failed to exit gracefully
            self_processes[pid].terminate()
            self_processes[pid].join()
            del self_processes[pid]
        self.propose_queue = None
        self.seed_queue = None

    def __del__(self) -> None:
        self.clean()

class ArrayDataset(Dataset):
    def __init__(self, *arrays, getitem_size: int = 1) -> None:
        super().__init__(getitem_size=getitem_size)
        self.sizes = None
        self.size = None
        self.view_idx = None
        self.shuffle_idx = None
        self.arrays = arrays

    @property
    def arrays(self):
        return self._arrays

    @arrays.setter
    def arrays(self, arrays) -> None:
        self.sizes = [array.shape[0] for array in arrays]
        if min(self.sizes) == 0:
            raise ValueError("Empty array is not allowed!")
        self.size = max(self.sizes)
        self.view_idx = [np.arange(s) for s in self.sizes]
        self.shuffle_idx = self.view_idx
        self._arrays = arrays

    def __len__(self) -> int:
        return ceil(self.size / self.getitem_size)

    def __getitem__(self, index: int) -> List[torch.Tensor]:
        index = np.arange(index * self.getitem_size,min((index + 1) * self.getitem_size, self.size))
        return [
            torch.as_tensor(a[self.shuffle_idx[i][np.mod(index, self.sizes[i])]].toarray())
            if scipy.sparse.issparse(a)
            else torch.as_tensor(a[self.shuffle_idx[i][np.mod(index, self.sizes[i])]])
            for i, a in enumerate(self.arrays)
        ]

    def propose_shuffle(self, seed: int) -> List[np.ndarray]:
        rs = get_rs(seed)
        return [rs.permutation(view_idx) for view_idx in self.view_idx]

    def accept_shuffle(self, shuffled: List[np.ndarray]) -> None:
        self.shuffle_idx = shuffled

    def random_split(
            self, fractions: List[float], random_state = None
    ) -> List["ArrayDataset"]:
        if min(fractions) <= 0:
            raise ValueError("Fractions should be greater than 0!")
        if sum(fractions) != 1:
            raise ValueError("Fractions do not sum to 1!")
        rs = get_rs(random_state)
        cum_frac = np.cumsum(fractions)
        subdatasets = [
            ArrayDataset(
                *self.arrays, getitem_size=self.getitem_size
            ) for _ in fractions
        ]
        for j, view_idx in enumerate(self.view_idx):
            view_idx = rs.permutation(view_idx)
            split_pos = np.round(cum_frac * view_idx.size).astype(int)
            split_idx = np.split(view_idx, split_pos[:-1])  # Last pos produces an extra empty split
            for i, idx in enumerate(split_idx):
                subdatasets[i].sizes[j] = len(idx)
                subdatasets[i].view_idx[j] = idx
                subdatasets[i].shuffle_idx[j] = idx
        return subdatasets

class AnnDataset(Dataset):
    def __init__(
            self, adatas: List[AnnData], data_configs: List[Mapping[str, Any]],
            mode: str = "train", getitem_size: int = 1
    ) -> None:
        super().__init__(getitem_size=getitem_size)
        self.mode = mode
        self.adatas = adatas
        self.data_configs = data_configs

    @property
    def adatas(self) -> List[AnnData]:
        return self._adatas

    @property
    def data_configs(self) -> Any:
        return self._data_configs

    @adatas.setter
    def adatas(self, adatas) -> None:
        self.sizes = [adata.shape[0] for adata in adatas]
        self._adatas = adatas

    @data_configs.setter
    def data_configs(self, data_configs) -> None:
        self.data_idx, self.extracted_data = self._extract_data(data_configs)
        self.view_idx = pd.concat(
            [data_idx.to_series() for data_idx in self.data_idx]
        ).drop_duplicates().to_numpy()
        self.size = self.view_idx.size
        self.shuffle_idx, self.shuffle_pmsk = self._get_idx_pmsk(self.view_idx)
        self._data_configs = data_configs

    def _get_idx_pmsk(
            self, view_idx: np.ndarray, random_fill: bool = False,
            random_state= None
    ):
        rs = get_rs(random_state) if random_fill else None
        shuffle_idx, shuffle_pmsk = [], []
        for data_idx in self.data_idx:
            idx = data_idx.get_indexer(view_idx)
            pmsk = idx >= 0
            n_true = pmsk.sum()
            n_false = pmsk.size - n_true
            idx[~pmsk] = rs.choice(idx[pmsk], n_false, replace=True) \
                if random_fill else idx[pmsk][np.mod(np.arange(n_false), n_true)]
            shuffle_idx.append(idx)
            shuffle_pmsk.append(pmsk)
        return np.stack(shuffle_idx, axis=1), np.stack(shuffle_pmsk, axis=1)

    def __len__(self) -> int:
        return ceil(self.size / self.getitem_size)

    def __getitem__(self, index: int) -> List[torch.Tensor]:
        s = slice(
            index * self.getitem_size,
            min((index + 1) * self.getitem_size, self.size)
        )
        shuffle_idx = self.shuffle_idx[s].T
        shuffle_pmsk = self.shuffle_pmsk[s]
        items = [
            torch.as_tensor(self._index_array(data, idx))
            for extracted_data in self.extracted_data
            for idx, data in zip(shuffle_idx, extracted_data)
        ]
        items.append(torch.as_tensor(shuffle_pmsk))
        return items

    @staticmethod
    def _index_array(arr, idx: np.ndarray) -> np.ndarray:
        if isinstance(arr, (h5py.Dataset)):
            rank = scipy.stats.rankdata(idx, method="dense") - 1
            sorted_idx = np.empty(rank.max() + 1, dtype=int)
            sorted_idx[rank] = idx
            arr = arr[sorted_idx][rank] 
        else:
            arr = arr[idx]
        return arr.toarray() if scipy.sparse.issparse(arr) else arr

    def _extract_data(self, data_configs):
        if self.mode == "eval":
            return self._extract_data_eval(data_configs)
        return self._extract_data_train(data_configs)  # self.mode == "train"

    def _extract_data_train(self, data_configs):
        xuid = [
            self._extract_xuid(adata, data_config)
            for adata, data_config in zip(self.adatas, data_configs)
        ]
        x = [
            self._extract_x(adata, data_config)
            for adata, data_config in zip(self.adatas, data_configs)
        ]
        xalt = [
            self._extract_xalt(adata, data_config)
            for adata, data_config in zip(self.adatas, data_configs)
        ]
        xlbl = [
            self._extract_xlbl(adata, data_config)
            for adata, data_config in zip(self.adatas, data_configs)
        ]

        return xuid, (x, xalt, xlbl)

    def _extract_data_eval(self, data_configs):
        default_dtype = get_default_numpy_dtype()
        xuid = [
            self._extract_xuid(adata, data_config)
            for adata, data_config in zip(self.adatas, data_configs)
        ]
        xalt = [
            self._extract_xalt(adata, data_config)
            for adata, data_config in zip(self.adatas, data_configs)
        ]
        x = [
            np.empty((adata.shape[0], 0), dtype=default_dtype)
            if xalt_.size else self._extract_x(adata, data_config)
            for adata, data_config, xalt_ in zip(self.adatas, data_configs, xalt)
        ]
        xlbl = [
            np.empty((adata.shape[0], 0), dtype=int)
            for adata in self.adatas
        ]
        return xuid, (x, xalt, xlbl)

    def _extract_x(self, adata: AnnData, data_config):
        default_dtype = get_default_numpy_dtype()
        features = data_config["features"]
        use_layer = data_config["use_layer"]
        if not np.array_equal(adata.var_names, features):
            adata = adata[:, features]  # This will load all data to memory if backed
        if use_layer:
            if use_layer not in adata.layers:
                raise ValueError(
                    f'{use_layer} cannot be found in input data.layers!'
                )
            x = adata.layers[use_layer]
        else:
            x = adata.X
        if x.dtype.type is not default_dtype:
            x = x.astype(default_dtype)
        if scipy.sparse.issparse(x):
            x = x.tocsr()
        return x

    def _extract_xalt(self, adata, data_config):
        default_dtype = get_default_numpy_dtype()
        use_rep = data_config["use_rep"]
        rep_dim = data_config["rep_dim"]
        if use_rep:
            if use_rep not in adata.obsm:
                raise ValueError(
                    f"'{use_rep}' cannot be found in input data.obsm!"
                )
            xalt = adata.obsm[use_rep].astype(default_dtype)
            if xalt.shape[1] != rep_dim:
                raise ValueError(
                    f"'{use_rep}' has {xalt.shape[1]} dimensions, but the configured dimensionality is {rep_dim}!"
                )
            return xalt
        return np.empty((adata.shape[0], 0), dtype=default_dtype)

    def _extract_xlbl(self, adata, data_config):
        use_labels = data_config["use_labels"]
        labels = data_config["labels"]
        if use_labels:
            if use_labels not in adata.obs:
                raise ValueError(
                    f"Configured labels '{use_labels}' "
                    f"cannot be found in input data!"
                )
            if labels is not None:
                return labels.get_indexer(adata.obs[use_labels])
            else:
                return np.array(adata.obs[use_labels].tolist())
        return -np.ones(adata.shape[0], dtype=int)

    def _extract_xuid(self, adata, data_config) -> pd.Index:
        use_uid = data_config["use_uid"]
        xuid = adata.obs[use_uid].to_numpy()
        if len(set(xuid)) != xuid.size:
            raise ValueError("Non-unique sample ID!")
        return pd.Index(xuid)

    def propose_shuffle(self, seed: int):
        rs = get_rs(seed)
        view_idx = rs.permutation(self.view_idx)
        return self._get_idx_pmsk(view_idx, random_fill=True, random_state=rs)

    def accept_shuffle(self, shuffled) -> None:
        self.shuffle_idx, self.shuffle_pmsk = shuffled

    def random_split(
            self, fractions: List[float], random_state = None
    ) -> List["AnnDataset"]:
        if min(fractions) <= 0:
            raise ValueError("Fractions should be greater than 0!")
        if sum(fractions) != 1:
            raise ValueError("Fractions do not sum to 1!")
        rs = get_rs(random_state)
        cum_frac = np.cumsum(fractions)
        view_idx = rs.permutation(self.view_idx)
        split_pos = np.round(cum_frac * view_idx.size).astype(int)
        split_idx = np.split(view_idx, split_pos[:-1])  # Last pos produces an extra empty split
        subdatasets = []
        for idx in split_idx:
            sub = copy.copy(self)
            sub.view_idx = idx
            sub.size = idx.size
            sub.shuffle_idx, sub.shuffle_pmsk = sub._get_idx_pmsk(idx)  # pylint: disable=protected-access
            subdatasets.append(sub)
        return subdatasets

def configure_dataset(
        adata: AnnData,
        use_uid: str,
        use_layer: Optional[str] = None,
        use_rep: Optional[str] = None,
        use_labels: Optional[str] = None,
        use_dsc_weight: Optional[str] = None,
        labels: Optional[str] = None
) -> None:
    data_config = {}
    data_config["features"] = adata.var_names.to_numpy().tolist()
    if use_layer:
        if use_layer not in adata.layers:
            raise ValueError("Invalid `use_layer`!")
        data_config["use_layer"] = use_layer
    else:
        data_config["use_layer"] = None
    if use_rep:
        if use_rep not in adata.obsm:
            raise ValueError("Invalid `use_rep`!")
        data_config["use_rep"] = use_rep
        data_config["rep_dim"] = adata.obsm[use_rep].shape[1]
    else:
        data_config["use_rep"] = None
        data_config["rep_dim"] = None
    if use_labels:
        if use_labels not in adata.obs:
            raise ValueError(f"{use_labels} not in adata.obs!")
        if not np.issubdtype(adata.obs[use_labels].dtype, np.number):
            raise ValueError(f"{use_labels} is not numeric!")
        data_config["use_labels"] = use_labels
        if labels is not None:
            data_config["labels"] = pd.Index(
                labels
            ).dropna().drop_duplicates().sort_values().to_numpy()
        else:
            data_config["labels"] = pd.Index(
                adata.obs[use_labels]
            ).dropna().drop_duplicates().sort_values().to_numpy()

    else:
        data_config["use_labels"] = None
        data_config["labels"] = None

    if use_uid:
        if use_uid not in adata.obs:
            raise ValueError(f"{use_uid} not in adata.obs!")
        xuid = adata.obs[use_uid].to_numpy()
        if len(set(xuid)) != xuid.size:
            raise ValueError("Non-unique sample ID!")
        data_config["use_uid"] = use_uid
    else:
        raise ValueError("`use_uid` is necessary!")
    
    adata.uns[config.ANNDATA_KEY] = data_config
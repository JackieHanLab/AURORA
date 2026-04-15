from typing import List,Mapping,Optional
from anndata import AnnData

import os
import numpy as np
import pandas as pd
import scanpy as sc
from sparse import COO

import scipy.sparse
from sklearn.preprocessing import normalize

from .utils import config, logged,prod
from .model import AuroraModel, load_model

@logged
def fit_model(
        adatas: Mapping[str, AnnData],
        features: List[str], 
        model: type = AuroraModel,
        project_name: str = "my_project"
) -> AuroraModel:
    fit_kws = {"directory":project_name}
    fit_model.logger.info("Prepare AURORA model...")
    fit_kws = fit_kws.copy()
    fit_kws.update({"align_burnin": np.inf, "safe_burnin": False})
    tmp_model = model(adatas, sorted(features))
    tmp_model.compile()
    fit_model.logger.info("Training AURORA model...")
    tmp_model.fit(adatas, **fit_kws)
    tmp_model.save(os.path.join(fit_kws["directory"], "model.dill"))
    return tmp_model
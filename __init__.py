
import numpy as np

import _C_gdt


def prob_dt(p, sigma):
    """
    p contains maps of probabilities of boundaries. Order is p[y,x,theta],
    so p[:,:,theta] contains the edgemap for direction theta.
    """
    # we are about to do log(0) = -inf, which is perfectly fine for the
    # generalized distance transform
    err_settings = np.geterr()
    np.seterr(divide="ignore", invalid="ignore")
    f = -2 * sigma * sigma
    logp = f * np.log(p)
    np.seterr(**err_settings)
    logp = np.rollaxis(logp, 2, 0)
    logp = np.require(logp, dtype=np.float32, requirements=['C', 'W', 'O'])
    _C_gdt.gdt(logp)
    logp = np.rollaxis(logp, 0, 3)
    return np.exp(logp / f)

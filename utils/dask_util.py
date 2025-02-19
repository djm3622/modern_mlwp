import dask # type: ignore
from dask.distributed import Client # type: ignore
import torch


# Initialize Dask settings for per process
def init(cfg):
    threads_per_dask_worker = max(1, torch.get_num_threads() // 2 // cfg['training']['dataloader_workers'])
    
    dask.config.set(scheduler='threads')
    client = Client(threads_per_worker=threads_per_dask_worker, n_workers=1)
    
    torch.set_num_threads(threads_per_dask_worker)

    return client


# Reinitialize Dask settings for per worker
def worker_init_fn(worker_id):
    threads_per_worker = max(1, torch.get_num_threads())

    dask.config.set(scheduler='threads')
    Client(threads_per_worker=threads_per_worker, n_workers=1)
    
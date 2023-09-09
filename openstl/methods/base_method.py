from typing import Dict, List, Union
import numpy as np

import torch
from torch.nn.parallel import DistributedDataParallel as NativeDDP
from contextlib import suppress
from timm.utils import NativeScaler
from timm.utils.agc import adaptive_clip_grad

from openstl.core import metric
from openstl.core.optim_scheduler import get_optim_scheduler
from openstl.utils import gather_tensors_batch, get_dist_info, ProgressBar

has_native_amp = False
try:
    if getattr(torch.cuda.amp, 'autocast') is not None:
        has_native_amp = True
except AttributeError:
    pass


class Base_method(object):
    """Base Method.

    This class defines the basic functions of a video prediction (VP)
    method training and testing. Any VP method that inherits this class
    should at least define its own `train_one_epoch`, `vali_one_epoch`,
    and `test_one_epoch` function.

    """

    def __init__(self, args, device, steps_per_epoch):
        super(Base_method, self).__init__()
        self.args = args
        self.dist = args.dist
        self.device = device
        self.config = args.__dict__
        self.criterion = None
        self.model_optim = None
        self.scheduler = None
        if self.dist:
            self.rank, self.world_size = get_dist_info()
            assert self.rank == int(device.split(':')[-1])
        else:
            self.rank, self.world_size = 0, 1
        self.clip_value = self.args.clip_grad
        self.clip_mode = self.args.clip_mode if self.clip_value is not None else None
        # setup automatic mixed-precision (AMP) loss scaling and op casting
        self.amp_autocast = suppress  # do nothing
        self.loss_scaler = None
        # setup metrics
        if ('weather' in self.args.dataname) or ('sst' in self.args.dataname):
            self.metric_list, self.spatial_norm = ['mse', 'rmse', 'mae', 'crps'], True
        else:
            self.metric_list, self.spatial_norm = ['mse', 'mae'], False

    def _build_model(self, **kwargs):
        raise NotImplementedError

    def _init_optimizer(self, steps_per_epoch):
        additional_params = []
        if self.args.loss == "dynmix":
            additional_params += list(self.criterion.covariance.parameters())

        return get_optim_scheduler(
            self.args, self.args.epoch, self.model, steps_per_epoch, additional_params)

    def _init_distributed(self):
        """Initialize DDP training"""
        if self.args.fp16 and has_native_amp:
            self.amp_autocast = torch.cuda.amp.autocast
            self.loss_scaler = NativeScaler()
            if self.rank == 0:
                print('Using native PyTorch AMP. Training in mixed precision (fp16).')
        else:
            print('AMP not enabled. Training in float32.')
        self.model = NativeDDP(self.model, device_ids=[self.rank],
                               broadcast_buffers=self.args.broadcast_buffers,
                               find_unused_parameters=self.args.find_unused_parameters)

    def train_one_epoch(self, runner, train_loader, **kwargs):
        """Train the model with train_loader.

        Args:
            runner: the trainer of methods.
            train_loader: dataloader of train.
        """
        raise NotImplementedError

    def _predict(self, batch_x, batch_y, **kwargs):
        """Forward the model.

        Args:
            batch_x, batch_y: testing samples and groung truth.
        """
        raise NotImplementedError

    def _dist_forward_collect(self, data_loader, length=None, gather_data=False):
        """Forward and collect predictios in a distributed manner.

        Args:
            data_loader: dataloader of evaluation.
            length (int): Expected length of output arrays.
            gather_data (bool): Whether to gather raw predictions and inputs.

        Returns:
            results_all (dict(np.ndarray)): The concatenated outputs.
        """
        # preparation
        results = []
        length = len(data_loader.dataset) if length is None else length
        if self.rank == 0:
            prog_bar = ProgressBar(len(data_loader))

        # loop
        for idx, (batch_x, batch_y) in enumerate(data_loader):
            if idx == 0:
                part_size = batch_x.shape[0]
            with torch.no_grad():
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                pred_y = self._predict(batch_x, batch_y)

            if gather_data:  # return raw datas
                results.append(dict(zip(['inputs', 'preds', 'trues'],
                                        [batch_x.cpu().numpy(), pred_y.cpu().numpy(), batch_y.cpu().numpy()])))
            else:  # return metrics
                eval_res, _ = metric(pred_y.cpu().numpy(), batch_y.cpu().numpy(),
                                     data_loader.dataset.mean, data_loader.dataset.std,
                                     metrics=self.metric_list, spatial_norm=self.spatial_norm, return_log=False)
                eval_res['loss'] = self.criterion(pred_y, batch_y).cpu().numpy()
                for k in eval_res.keys():
                    eval_res[k] = eval_res[k].reshape(1)
                results.append(eval_res)

            if self.args.empty_cache:
                torch.cuda.empty_cache()
            if self.rank == 0:
                prog_bar.update()

        # post gather tensors
        results_all = {}
        for k in results[0].keys():
            results_cat = np.concatenate([batch[k] for batch in results], axis=0)
            # gether tensors by GPU (it's no need to empty cache)
            results_gathered = gather_tensors_batch(results_cat, part_size=min(part_size*8, 16))
            results_strip = np.concatenate(results_gathered, axis=0)[:length]
            results_all[k] = results_strip
        return results_all

    def _nondist_forward_collect(self, data_loader, length=None, gather_data=False):
        """Forward and collect predictios.

        Args:
            data_loader: dataloader of evaluation.
            length (int): Expected length of output arrays.
            gather_data (bool): Whether to gather raw predictions and inputs.

        Returns:
            results_all (dict(np.ndarray)): The concatenated outputs.
        """
        # preparation
        results = []
        prog_bar = ProgressBar(len(data_loader))
        length = len(data_loader.dataset) if length is None else length

        # loop
        for idx, (batch_x, batch_y) in enumerate(data_loader):
            with torch.no_grad():
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                pred_y = self._predict(batch_x, batch_y)
                # if self.args.loss == "dynmix":
                pred_y_mix, logw, sigma = pred_y
                pred_y = (pred_y_mix.permute(0, 2, 1, 3, 4, 5) * logw.exp().unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)).sum(1)

                if not self.args.loss == "dynmix":
                    sigma = None
                    logw = None
                    pred_sigma = None
                else:
                    L_list = self.criterion.covariance.get_L()
                    b, t, k, x, y = pred_y.shape

                    L_list = [l.transpose(-1, -2).unsqueeze(0).repeat(b, 1, 1, 1) for l in L_list]
                    sigma0 = torch.nn.functional.elu(sigma) + 1 + 1e-6
                    sigma0 = sigma0.permute(0,3,1,2).reshape(b,k,x*y)
                    L_list[2] = L_list[2] * sigma0.unsqueeze(-1)

                    # L_list[0] is [ b,k,t,t]
                    # L_list[2] is [ b,k,xy,xy]
                    # get diagonal of L_list[0] and L_list[2]
                    L_list_0_diag = torch.diagonal(L_list[0], dim1=-2, dim2=-1)
                    L_list_2_diag = torch.diagonal(L_list[2], dim1=-2, dim2=-1)
                    # generate [b , k , t , xy] tensor by doing element-wise multiplication between L_list[0] and L_list[2]
                    L_list_0_diag = L_list_0_diag.unsqueeze(-1).repeat(1,1,1,x*y)
                    L_list_2_diag = L_list_2_diag.unsqueeze(-2).repeat(1,1,t,1)
                    pred_sigma = L_list_0_diag * L_list_2_diag
                    pred_sigma = pred_sigma.reshape(b,t,k,x,y)

            if gather_data:  # return raw datas
                results.append(dict(zip(['inputs', 'preds', 'trues'],
                                        [batch_x.cpu().numpy(), pred_y.cpu().numpy(), batch_y.cpu().numpy()])))
            else:  # return metrics
                eval_res, _ = metric(pred_y.cpu().numpy(), batch_y.cpu().numpy(), pred_sigma.cpu().numpy(),
                                     data_loader.dataset.mean, data_loader.dataset.std,
                                     metrics=self.metric_list, spatial_norm=self.spatial_norm, return_log=False)

                if self.args.loss == "dynmix":
                    eval_res['loss'] = self.criterion(pred_y_mix, batch_y, logw, sigma).cpu().numpy()
                else:
                    eval_res['loss'] = self.criterion(pred_y, batch_y).cpu().numpy()

                for k in eval_res.keys():
                    eval_res[k] = eval_res[k].reshape(1)

                eval_res["inputs"] = batch_x.cpu().numpy()
                eval_res["preds"] = pred_y.cpu().numpy()
                eval_res["trues"] = batch_y.cpu().numpy()

                results.append(eval_res)

            prog_bar.update()
            if self.args.empty_cache:
                torch.cuda.empty_cache()

        # post gather tensors
        results_all = {}
        for k in results[0].keys():
            results_all[k] = np.concatenate([batch[k] for batch in results], axis=0)
        return results_all

    def vali_one_epoch(self, runner, vali_loader, **kwargs):
        """Evaluate the model with val_loader.

        Args:
            runner: the trainer of methods.
            val_loader: dataloader of validation.

        Returns:
            list(tensor, ...): The list of predictions and losses.
            eval_log(str): The string of metrics.
        """
        self.model.eval()
        if self.dist and self.world_size > 1:
            results = self._dist_forward_collect(vali_loader, len(vali_loader.dataset), gather_data=False)
        else:
            results = self._nondist_forward_collect(vali_loader, len(vali_loader.dataset), gather_data=False)

        eval_log = ""
        for k, v in results.items():
            v = v.mean()
            if k != "loss":
                eval_str = f"{k}:{v.mean()}" if len(eval_log) == 0 else f", {k}:{v.mean()}"
                eval_log += eval_str

        residual = results["preds"] - results["trues"]
        b,t,_,h,w = residual.shape

        residual_t = residual.transpose(1,0,2,3,4).reshape(t,-1)
        residual_s = residual.reshape(b,t,h*w).transpose(2,0,1).reshape(h*w,b*t)
        residual_x = residual.transpose(3,0,1,2,4).reshape(h,-1)
        residual_y = residual.transpose(4,0,1,2,3).reshape(w,-1)

        cov_t = residual_t @ residual_t.T / (b*t*h*w)
        cov_s = residual_s @ residual_s.T / (b*t*h*w)
        cov_x = residual_x @ residual_x.T / (b*t*h*w)
        cov_y = residual_y @ residual_y.T / (b*t*h*w)

        import seaborn as sns
        import matplotlib.pyplot as plt
        import wandb
        sns.set_theme(style="white")

        # draw cov_t, cov_s, cov_x, cov_y and upload on wandb
        fig, ax = plt.subplots(figsize=(10, 10))
        sns.heatmap(cov_t, ax=ax)
        wandb.log({"cov_t": wandb.Image(fig)})
        plt.close()

        fig, ax = plt.subplots(figsize=(10, 10))
        sns.heatmap(cov_s, ax=ax)
        wandb.log({"cov_s": wandb.Image(fig)})
        plt.close()

        fig, ax = plt.subplots(figsize=(10, 10))
        sns.heatmap(np.log(cov_x), ax=ax)
        wandb.log({"cov_x": wandb.Image(fig)})
        plt.close()

        fig, ax = plt.subplots(figsize=(10, 10))
        sns.heatmap(np.log(cov_y), ax=ax)
        wandb.log({"cov_y": wandb.Image(fig)})
        plt.close()


        for i in range(t):
            fig, ax = plt.subplots(figsize=(10, 10))
            data = residual[10,:,0]
            sns.heatmap(data[i], ax = ax, vmin = np.min(data), vmax = np.max(data))
            wandb.log({f"residual/{str(i).zfill(2)}": wandb.Image(fig)})
            plt.close()

        return results, eval_log

    def test_one_epoch(self, runner, test_loader, **kwargs):
        """Evaluate the model with test_loader.

        Args:
            runner: the trainer of methods.
            test_loader: dataloader of testing.

        Returns:
            list(tensor, ...): The list of inputs and predictions.
        """
        self.model.eval()
        if self.dist and self.world_size > 1:
            results = self._dist_forward_collect(test_loader, gather_data=True)
        else:
            results = self._nondist_forward_collect(test_loader, gather_data=True)

        return results

    def current_lr(self) -> Union[List[float], Dict[str, List[float]]]:
        """Get current learning rates.

        Returns:
            list[float] | dict[str, list[float]]: Current learning rates of all
            param groups. If the runner has a dict of optimizers, this method
            will return a dict.
        """
        lr: Union[List[float], Dict[str, List[float]]]
        if isinstance(self.model_optim, torch.optim.Optimizer):
            lr = [group['lr'] for group in self.model_optim.param_groups]
        elif isinstance(self.model_optim, dict):
            lr = dict()
            for name, optim in self.model_optim.items():
                lr[name] = [group['lr'] for group in optim.param_groups]
        else:
            raise RuntimeError(
                'lr is not applicable because optimizer does not exist.')
        return lr

    def clip_grads(self, params, norm_type: float = 2.0):
        """ Dispatch to gradient clipping method

        Args:
            parameters (Iterable): model parameters to clip
            value (float): clipping value/factor/norm, mode dependant
            mode (str): clipping mode, one of 'norm', 'value', 'agc'
            norm_type (float): p-norm, default 2.0
        """
        if self.clip_mode is None:
            return
        if self.clip_mode == 'norm':
            torch.nn.utils.clip_grad_norm_(params, self.clip_value, norm_type=norm_type)
        elif self.clip_mode == 'value':
            torch.nn.utils.clip_grad_value_(params, self.clip_value)
        elif self.clip_mode == 'agc':
            adaptive_clip_grad(params, self.clip_value, norm_type=norm_type)
        else:
            assert False, f"Unknown clip mode ({self.clip_mode})."

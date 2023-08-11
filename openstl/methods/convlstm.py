import torch.nn as nn

from openstl.models import ConvLSTM_Model
from .predrnn import PredRNN
from .dynmix import better_loss, covariance


class ConvLSTM(PredRNN):
    r"""ConvLSTM

    Implementation of `Convolutional LSTM Network: A Machine Learning Approach
    for Precipitation Nowcasting <https://arxiv.org/abs/1506.04214>`_.

    Notice: ConvLSTM requires `find_unused_parameters=True` for DDP training.
    """

    def __init__(self, args, device, steps_per_epoch):
        PredRNN.__init__(self, args, device,  steps_per_epoch)
        self.model = self._build_model(self.args)
        self.model_optim, self.scheduler, self.by_epoch = self._init_optimizer(steps_per_epoch)
        # self.criterion = nn.MSELoss()

    def _build_model(self, args):
        num_hidden = [int(x) for x in self.args.num_hidden.split(',')]
        num_layers = len(num_hidden)
        return ConvLSTM_Model(num_layers, num_hidden, args, criterion=self.criterion).to(self.device)

    def _set_criterion(self, args):
        if args.loss == "mse":
            self.criterion = nn.MSELoss()
        elif args.loss == "mae":
            self.criterion = nn.L1Loss()
        elif args.loss == "dynmix":
            self.criterion = better_loss(xdim=args.input_dim_x,
                                         ydim=args.input_dim_y,
                                         pred_len=args.pred_len,
                                         train_L_x=args.train_L_x,
                                         train_L_y=args.train_L_y,
                                         train_L_t=args.train_L_t)
        else:
            raise NotImplementedError

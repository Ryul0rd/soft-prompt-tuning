from torch.utils.tensorboard.summary import hparams
from torch.utils.tensorboard import SummaryWriter
import torch
import os

class FixedSummaryWriter(SummaryWriter):
    def add_hparams(self, hparam_dict, metric_dict, hparam_domain_discrete=None, global_step=0, run_name=None):
        torch._C._log_api_usage_once("tensorboard.logging.add_hparams")
        if type(hparam_dict) is not dict or type(metric_dict) is not dict:
            raise TypeError('hparam_dict and metric_dict should be dictionary.')
        exp, ssi, sei = hparams(hparam_dict, metric_dict, hparam_domain_discrete)

        if not run_name:
            logdir = self.log_dir
        else:
            logdir = os.path.join(self._get_file_writer().get_logdir(), run_name)
        with SummaryWriter(log_dir=logdir) as w_hp:
            w_hp.file_writer.add_summary(exp)
            w_hp.file_writer.add_summary(ssi)
            w_hp.file_writer.add_summary(sei)
            for k, v in metric_dict.items():
                w_hp.add_scalar(k, v, global_step)

# Example usage:
# exp_name = 'testing'
# WRITER = FixedSummaryWriter(f'logs/{exp_name}')
# param_dict = {'a': 1, 'b': 2}
# for i in range(5):
#     loss = 1 / (i ** 2 + 1)
#     WRITER.add_hparams(
#           hparam_dict=param_dict,
#           metric_dict={'metric/loss':loss},
#           global_step=i)
# WRITER.flush()
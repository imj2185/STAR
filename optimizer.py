import torch
import torch.optim as opt


class NoamOpt(object):
    def __init__(self, model_size, factor, warmup_steps, optimizer):
        """
        Args:
            model_size: hidden size
            factor: coefficient
            warmup_steps: warm up steps(step ** (-0.5) == step * warmup_steps ** (-1.5) holds when warmup_steps equals step)
            optimizer: opt.optimizer
        """
        self.optimizer = optimizer
        self._step = 0
        self.warmup_steps = warmup_steps
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    def rate(self, step=None):
        if step is None:
            step = self._step
        return self.factor * (self.model_size ** (-0.5) * min(step ** (-0.5), step * self.warmup_steps ** (-1.5)))

    def step(self):
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()


def get_std_opt(model, args):
    channels = [int(n) for n in args.encoder_channels.split(',')]
    return NoamOpt(channels[2],  # TODO num_nodes is not fixed
                   args.opt_train_factor,
                   args.warmup_steps,
                   opt.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98), eps=1e-9,
                            weight_decay=args.weight_decay))
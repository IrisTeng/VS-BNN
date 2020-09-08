from torch.distributions import constraints
from torch.distributions.transforms import SigmoidTransform
from torch.distributions.normal import Normal
from torch.distributions.transformed_distribution import TransformedDistribution
from torch.distributions.kl import kl_divergence, register_kl

class LogitNormal(TransformedDistribution):
    r"""
    Creates a logit-normal distribution parameterized by
    :attr:`loc` and :attr:`scale` where::

        X ~ Normal(loc, scale)
        Y = Logistic(X) ~ LogitNormal(loc, scale)

    Example::

        >>> m = LogitNormal(torch.tensor([0.0]), torch.tensor([1.0]))
        >>> m.sample()  # logit-normal distributed with mean=0 and stddev=1
        tensor([ 0.6798])

    Args:
        loc (float or Tensor): mean of logit of distribution
        scale (float or Tensor): standard deviation of logit of the distribution
    """
    arg_constraints = {'loc': constraints.real, 'scale': constraints.positive}
    support = constraints.positive
    has_rsample = True

    def __init__(self, loc, scale, validate_args=None):
        base_dist = Normal(loc, scale)
        #super(LogitNormal, self).__init__(base_dist, SigmoidTransform(), validate_args=validate_args) # causes an error if using importlib.reload
        super().__init__(base_dist, SigmoidTransform(), validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(LogitNormal, _instance)
        return super(LogitNormal, self).expand(batch_shape, _instance=new)

    @property
    def loc(self):
        return self.base_dist.loc

    @property
    def scale(self):
        return self.base_dist.scale

    @property
    def mean(self):
        raise NotImplementedError # No analytical solution

    @property
    def variance(self):
        raise NotImplementedError # No analytical solution

    def entropy(self):
        return self.base_dist.entropy() + self.loc # BC: is this correct?

@register_kl(LogitNormal, LogitNormal)
def _kl_logitnormal_logitnormal(p, q):
    return kl_divergence(p.base_dist, q.base_dist)



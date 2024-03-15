import torch

# [UNEFFICIENT!!! USE KFRANK VERSION BELOW]
def log1m_softmax(x : torch.Tensor, mask : torch.Tensor = None, dim : int = -1, default = torch.nan) -> torch.Tensor:
    """[UNEFFICIENT!!! USE KFRANK VERSION BELOW] This function computes the log(1-softmax(x)) on a portion of x specified by the given mask

    Args:
        x (torch.Tensor): The input tensor (e.g. logits)
        mask (torch.Tensor, optional): A mask (bool tensor). If mask==None, then the mask is full ones (compute everything). Defaults to None.
        dim (int, optional): The dimension on which to compute log(1-softmax(x)). Defaults to -1.
        default (_type_, optional): _description_. Defaults to torch.nan.

    Returns:
        torch.Tensor: log(1-softmax(x)) except for mask = 0
    """
    # default is interesting for someone who is trying to compute the loss (default = 0 then .sum(dim))
    n_dims = x.dim()
    if dim < 0:
        # Handle negative dimension
        dim = n_dims + dim

    if mask is None:
        # If mask not given, compute log(1-softmax) for all the tensor (WARNING: Time and memory complexity scales quadratically with x.shape[dim])
        mask = torch.ones_like(x)
    
    res = torch.full_like(x, default)

    where = list(torch.where(mask))

    if len(where) == 0:
        # Mask is empty ==> Return tensor full of 'default'
        return res
    n_nonzero = len(where[0])

    # Substract maximum to ensure x < 0 
    # This part improves greatly the precision of the function when max(x) → -inf
    # This is odd, I thought substracting the max is automatically done in torch.logsumexp (https://discuss.pytorch.org/t/source-code-of-torch-logsumexp/54848)
    x = x - torch.max(x, dim=dim, keepdim=True).values

    lse1 = x.logsumexp(dim)

    dim_pos_interest = where[dim]
    where[dim] = slice(None,None,None)
    if n_dims > 1:
        x_of_interest = x[where]
    else:
        x_of_interest = x.repeat(len(dim_pos_interest), 1)

    if n_dims > 1 and dim == 0:
        # Edge case
        x_of_interest = x_of_interest.T

    where.pop(dim)
    lse1_of_interest = lse1[where] if n_dims > 1 else lse1 # Edge case
    where.insert(dim, dim_pos_interest)

    mask_of_interest = torch.zeros_like(x_of_interest, dtype=torch.bool)
    mask_of_interest[range(n_nonzero), dim_pos_interest] = 1
    x_of_interest[mask_of_interest] = -torch.inf

    lse2_of_interest = x_of_interest.logsumexp(-1)

    log1msoft = lse2_of_interest - lse1_of_interest

    res[where] = log1msoft

    return res


def log1m_softmax_kfrank(X : torch.Tensor, dim : int):
    """This function computes log(1-softmax(x))

    Args:
        x (torch.Tensor): The input tensor (e.g. logits)
        dim (int): The dimension on which to compute log(1-softmax(x)).

    Returns:
        torch.Tensor: log(1-softmax(x)) except for mask = 0
    """
    xm, im = X.max (dim, keepdim=True)                               # largest value in X is the potential problem
    X_bar = X - xm                                                   # uniform shift doesn't affect softmax (except numerically)
    lse = X_bar.logsumexp (dim, keepdim=True)                        # denominator for final result
    sumexp = X_bar.exp().sum(dim, keepdim=True) - X_bar.exp()        # sumexp[im] potentially zero
    sumexp.scatter_(dim, im, 1.0)                                    # protect against log (0)
    log1msm = sumexp.log()                                           # good for all but i = im
    X_bar = X_bar.clone()                                            # to support backward pass
    X_bar.scatter_(dim, im, -float ('inf'))                          # "zero out" xm in log space
    log1msm.scatter_(dim, im, X_bar.logsumexp (dim).view(im.shape))  # replace bad xm term
    log1msm -= lse                                                   # final result
    return log1msm

if __name__ == '__main__':
    # a = torch.tensor([[-100,-100,-100,-100,-100], [1,1,1,1,1], [2,2,2,2,2]]).bfloat16().requires_grad_()

    # a = torch.tensor([[-100,-100,-100,-100,-100]]).bfloat16().requires_grad_()

    # a = torch.randn(20,50,50000).float()
    # mask = a.ge(4)

    a = torch.tensor([-100,-100,-100,-100,1000]).repeat(2,1).bfloat16().requires_grad_()
    l = log1m_softmax_kfrank(a, -1)
    l2 = log1m_softmax(a, dim=-1)
    l3 = (1- a.softmax(0)).log()
    print(l,l2,l3)

    a = torch.randn((10,2,500))
    res1 = log1m_softmax(a, dim=1)
    res2 = torch.stack([log1m_softmax(a[i,:,j], dim=-1) for i in range(a.shape[0]) for j in range(a.shape[2])]).view(10,500,2).swapaxes(1,2)

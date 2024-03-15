# Numerically stable log(1-softmax(x))

This repository contains the implementation of a numerically stable log(1-softmax(x)) function in PyTorch.
- The implementation is located in *log1msoftmax.py* under the name *log1m_softmax_kfrank*
- The evaluation of this function compared to the naive unstable implementation can be found in the notebook *test.ipynb*
- The unit tests of this implementation can be found in the folder *tests*

This is the PyTorch-quality-of-life version of the numerically stable log(1-softmax) function proposed by *KFrank* in a [PyTorch forum discussion](https://discuss.pytorch.org/t/how-to-calculate-log-1-softmax-x-numerically-stably/169007/11?u=hichem) 
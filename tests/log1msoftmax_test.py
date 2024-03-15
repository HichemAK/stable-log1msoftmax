import unittest
from log1msoftmax import log1m_softmax
import torch
from mpmath import *

mp.prec=1000

class Testlog1m_softmax(unittest.TestCase):

    def test_general(self):
        # Test if tensor output shape is correct using a naive implementation of log(1-softmax) is pytorch
        # Trying different number of dimensions (1 to 5) with dim going from 0 to n_dim
        # We also test that the positions of nan values in the output correspond to the mask given in arguments
        # We also 
        shape = [10,20,30,40,50]
        for n_dim in range(5):
            a = torch.randn(shape[:n_dim+1]).double()
            mask = a.ge(4)
            for dim in range(n_dim+1):
                l = log1m_softmax(a, mask, dim, default=torch.nan)
                l2 = (1- a.softmax(dim)).log()
                self.assertTrue(l2.shape == l.shape)
                self.assertTrue((l.isnan() == ~mask).all())


    def test_stability(self):
        # Test the numerical stability of log1m_softmax. This is a fail case for the naive implementation of the log(1-softmax)
        # We test the stability on float64, float32, float16, and bfloat16
        for d in (torch.float64, torch.float32, torch.float16, torch.bfloat16):
            a = torch.tensor([0,0,2000], dtype=d).requires_grad_()
            mask = torch.tensor([1,1,1]).bool()
            not_nan = log1m_softmax(a, mask, 0)
            self.assertTrue(not_nan[0] == 0)
            self.assertTrue(not_nan[1] == 0)
            self.assertTrue(-1999 >= not_nan[2] >= -2000)

    def test_parameter(self):
        # Test that parameters work as intended
        # Test mask=None which means compute log1m_softmax for all the input tensor
        # Test default parameter which controls what to put in (mask == False), by default, it is equal to torch.nan 

        a = torch.zeros((20,50))
        l = log1m_softmax(a, default=torch.nan)
        self.assertFalse(l.isnan().any())
        
        mask = torch.zeros_like(a).bool()
        l = log1m_softmax(a, mask, default=1024)
        self.assertTrue((l == 1024).all())

        
        mask[0,0] = True
        l = log1m_softmax(a, mask, default=1024)
        self.assertTrue(((l != 1024) == mask).all())

    def test_precision(self):
        # It was shown in test.ipynb that log1m_softmax was very precise in for 1D vectors. Let's try to see if it is the case with higher dimensions
        a = torch.randn((10,500))
        res1 = log1m_softmax(a)
        res2 = torch.stack([log1m_softmax(a[i]) for i in range(a.shape[0])])
        self.assertTrue((res1 == res2).all())

        a = torch.randn((10,2,500))
        res1 = log1m_softmax(a)
        res2 = torch.stack([log1m_softmax(a[i,j]) for i in range(a.shape[0]) for j in range(a.shape[1])]).view(10,2,-1)
        self.assertTrue((res1 == res2).all())

        a = torch.randn((10,2,500))
        res1 = log1m_softmax(a, dim=1)
        res2 = torch.stack([log1m_softmax(a[i,:,j], dim=-1) for i in range(a.shape[0]) for j in range(a.shape[2])]).view(10,500,2).swapaxes(1,2)
        self.assertTrue((res1 == res2).all())

if __name__ == '__main__':
    unittest.main()
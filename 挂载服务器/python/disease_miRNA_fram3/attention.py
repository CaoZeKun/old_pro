import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable

class Attention(nn.Module):
    def __init__(self,input_size,output_size):
        super(Attention,self).__init__()

        self.linear = nn.Linear(input_size , output_size,bias=True) #input_size * 2 + zs_size
        nn.init.xavier_uniform(self.linear.weight)

        self.h_n_parameters = nn.Parameter(torch.randn(output_size,1))
        nn.init.xavier_uniform(self.h_n_parameters)

    def forward(self, Z_st,h_n_state):
        #for i in range(h_n_state.size(0)):  #(batch, time_step, hidden_size * num_directions)
        #for i in range(h_n_state.size(0)):
        a = []
        temp= []
        for j in range(h_n_state[0].size(0)): # length in path should same
            temp.append(Z_st)
        a.append(temp)
        a = torch.Tensor(a)
        print(a)
        h_combine = torch.cat((h_n_state,a),2)
        h_combine = Variable(h_combine)
        #print(h_combine)
        k = self.linear(h_combine)
        h_combine = F.tanh(k)
        #print(h_combine)
        h_combine = torch.squeeze(h_combine)
        print(h_combine)

        score_n = torch.mm(h_combine,self.h_n_parameters)
        print(score_n)
        score_n = torch.t(score_n)
        print(score_n)

        alpha = F.softmax(score_n)
        d = torch.squeeze(h_n_state)
        print(d)
        h_n_state_c = Variable(torch.squeeze(h_n_state))
        C = torch.mm(alpha,h_n_state)  # get vectors of different node

        return C



if __name__ == '__main__':
    #z_st = torch.Tensor([1,2,3]) # torch.Size([3])
    z_st =[1, 2, 3]

    #print(z_st)    1  2  3  [torch.FloatTensor of size 1x3]
    h_n = torch.Tensor([[[2,3,4,7],[2,5,6,7]]])#torch.Size([1, 2, 4])

    #print(h_n)
    attention = Attention(7,2)
    c = attention(z_st,h_n)
    print(c)


    """ ----cat---
    h_n = torch.Tensor([[[2, 3, 4, 7]],
                        [[1, 2, 2, 2]]])  # torch.Size([2, 3, 4])
    z = [1, 2, 3]
    a = []
    for i in range(h_n.size(0)):
        temp = []
        for j in range(h_n[0].size(0)):  # length in path should same
            temp.append(z)
        a.append(temp)
    a = torch.Tensor(a)
    c = torch.cat((a,h_n),2)
    print(c)
    """

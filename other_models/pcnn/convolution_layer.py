import torch
import torch.nn as nn
import sys
# sys.path.append('../')
from utils.kcnet_utils import debugPrint

class ConvLayer(nn.Module):
    def __init__(self, input_channels, out_channels, num_of_translations, scope, is_interpolation=False, 
                use_xavier=True,l2_regularizer=1.0e-3, bn=False, activation_fn="relu"):
        super(ConvLayer, self).__init__()
        self.input_channels = input_channels
        # self.block_elemnts = block_elemnts
        self.num_of_translations=num_of_translations
        self.out_channels = out_channels
        self.scope = scope
        self.use_xavier = use_xavier
        
        self.weight_decay = 0.0
        self.l2_regularizer = l2_regularizer
        self.is_interpolation = is_interpolation
        # no weight decay
        self.k_tensor = nn.Parameter(torch.Tensor(self.out_channels, self.input_channels,self.num_of_translations)) # out=xA^T+b
        # self.biases = nn.Parameter(torch.Tensor(self.out_channels))
        self.bn = nn.BatchNorm1d(self.out_channels)
        if activation_fn=='relu':
            self.act=nn.functional.relu
        elif activation_fn=='elu':
            self.act=nn.functional.elu
        self.reset_parameters()
        self.cuda()
    def reset_parameters(self):
        with torch.no_grad():
            if self.use_xavier:
                nn.init.xavier_normal_(self.k_tensor,gain=1.0)
                # nn.init.xavier_uniform_(self.k_tensor,gain=1.0)
                # self.weight.uniform_(-a, a)
            else:
                nn.init.normal_(self.k_tensor,mean=0,std=0.1)
            # nn.init.constant_(self.biases,0.0)

    def get_convlution_operator(self,block_elemnts,functions_pl):
        translations = block_elemnts.kernel_translations # (b,27,3)
        distances = block_elemnts.get_distance_matrix() # (b,n,n)
        # debugPrint(translations.size())
        # debugPrint(block_elemnts.points_pl.size())
        points_translations_dot = torch.matmul(block_elemnts.points_pl, translations.transpose(1,2)) # (b,n,num_trans)
        # debugPrint(points_translations_dot.size())
        translations_square = torch.sum(translations * translations, dim=2) # (b,27)

        # Find weights w of the extension operator. result dimensions are BATCH_SIZE x NUM_OF_POINTS x  POINT_CLOUD_FUNCTION_DIM
        if (self.is_interpolation):
            # SOLVE AX=B
            w_tensor = torch.solve(B=functions_pl,
                                   A=block_elemnts.get_interpolation_matrix())
        else:
            w_tensor = torch.mul(
                        torch.reciprocal(torch.sum(block_elemnts.get_interpolation_matrix(),dim=2,keepdim=True)),
                        functions_pl)  # (b,n,4)
        # print("w_tensor before shape={}".format(w_tensor.size()))
        # Calculate the product of w_tensor and the kernel weights. Result dimensions are TRANSLATIONS x BATCH_SIZE x NUM_OF_POINTS x OUT_CHANNELS
        b_tensor = torch.matmul(w_tensor.unsqueeze(1).repeat(1,self.out_channels,1,1), # (b,out,n,in)
                    self.k_tensor.unsqueeze(0).repeat(block_elemnts.batch_size, 1, 1, 1)).transpose(0,1).transpose(0,3) # (b,out,in,num_trans)-->.. -->(num_trans,b,n,out)
        # print("b_tensor shape={}".format(b_tensor.size()))
        # Calculate the pcnn convolution operator. Result dimensions are BATCH_SIZE x NUM_OF_POINTS x OUT_CHANNELS
        
        ### convopeator_per_translation
        def convopeator_per_translation(b_per_translation, translation_index):
            # b_per_translation, translation_index = inputs
            dot = -2 * points_translations_dot[:,:,translation_index].unsqueeze(2).repeat(1,1,block_elemnts.num_of_points)  # -->(b,n,1) -->(b,n,n)
            # print("dot shape={}".format(dot.size()))
            # print("distances shape={}".format(distances.size()))
            q_tensor = torch.exp(-(distances+dot-dot.transpose(1,2)+
                                translations_square[:,translation_index].view(-1,1,1))
                              / (2 * block_elemnts.combined_sigma ** 2))
            return torch.matmul(q_tensor, b_per_translation)  #--> (b,n,n)*(b,n,out)=(b,n,out)
        # dot = -2 * points_translations_dot[:,:,translation_index].unsqueeze(2).repeat(1,1,block_elemnts.num_of_points)  # -->(b,n,1) -->(b,n,n)
        # q_tensor = torch.exp(-(distances+dot-dot.transpose(1,2)+
        #                         translations_square[:,translation_index].view(-1,1,1))
        #                       / (2 * block_elemnts.combined_sigma ** 2))
        # convopeator_per_translation=torch.matmul(q_tensor, b_per_translation)

        return torch.sum(torch.stack([convopeator_per_translation(b_tensor[i],i)
                for i in range(block_elemnts.num_of_translations)], dim=0),
                dim=0)

    # @staticmethod
    # def convopeator_per_translation(b_per_translation, translation_index, 
    #                 points_translations_dot, num_of_points, combined_sigma):
    #         # b_per_translation, translation_index = inputs
    #         dot = -2 * points_translations_dot[:,:,translation_index].unsqueeze(2).repeat(1,1,num_of_points)  # -->(b,n,1) -->(b,n,n)
    #         # print("dot shape={}".format(dot.size()))
    #         # print("distances shape={}".format(distances.size()))
    #         q_tensor = torch.exp(-(distances+dot-dot.transpose(1,2)+
    #                             translations_square[:,translation_index].view(-1,1,1))
    #                           / (2 * combined_sigma ** 2))
    #         return torch.matmul(q_tensor, b_per_translation)  #--> (b,n,n)*(b,n,out)=(b,n,out)

    
    def forward(self,block_elemnts,functions_pl):
        outputs = self.get_convlution_operator(block_elemnts,functions_pl)
        # print("outputs shape={}".format(outputs.size()))
        # convlution_operation: BATCH_SIZE x NUM_OF_POINTS x OUT_CHANNELS
        outputs = self.bn(outputs.transpose(1,2))  # because bn follows, no need to add biases
        outputs = self.act(outputs)
        return outputs.transpose(1,2)  # --->(b,n,c)
import torch
from utils.kcnet_utils import debugPrint


def gather_nd(inpus_tensor, mask, num_channels, batch_size, dim=0):
    """
    mimc tf.gather_nd
    """
    # mask=mask.cuda()

    if mask.size(0)==batch_size:  # bboxes_xyz is a batched version
        b = inpus_tensor.size(0)
        b_ = mask.size(0)
        # assert(b == b_)
        # assert(b == batch_size)
        return torch.stack([torch.index_select(inpus_tensor[k],0,mask[k]) for k in range(batch_size)]) # keep dim as xyz
    else:
        return torch.index_select(inpus_tensor,0,mask)
    # return torch.index_select(inpus_tensor.view(-1,num_channels),dim,mask).view(batch_size,-1,num_channels)



"""
def gather_nd(bboxes_xyz, votes_assignment,
                first_two_dim_index=False, batch=False):
    '''
    implement tf.gather_nd like op by pytorch ops
    :param bboxes_xyz: float tensor, shape=(b,num_bb,...) dim=2 or 3
    :param votes_assignment:long tensor, dim=2, shape=(num_p,2) or (b_,m)
    :param first_two_dim_index: bool. votes_assignment's dim=2, it pick the index (i,votes_assignment[i][j])
        of bboxes_xyz when first_two_dim_index=True; votes_assignment's dim=1, it pick the index (votes_assignment[i])
        of bboxes_xyz when first_two_dim_index=False.
        default=False
    :param batched: bool, batched version gather_nd. If it is true, bboxes_xyz is a batched version,
        vice versa. default=False
    :return selected_bboxes_xyz, float tensor. 
    '''
    if first_two_dim_index:
        b = bboxes_xyz.size(0)
        num_bb = bboxes_xyz.size(1)

        bboxes_xyz = bboxes_xyz.view(b * num_bb, -1) if bboxes_xyz.dim() == 3 else bboxes_xyz.view(-1)
        one_dim_index = votes_assignment[:, 0] * num_bb + votes_assignment[:, 1]
        return torch.index_select(bboxes_xyz, 0, one_dim_index.long()) # shape=(num_p,...)
    if batched:  # bboxes_xyz is a batched version
        b = bboxes_xyz.size(0)
        b_ = votes_assignment.size(0)
        assert(b == b_)
        return torch.stack([torch.index_select(bboxes_xyz[k],0,votes_assignment[k]) for k in range(b)]) # keep dim as xyz
    else:
        assert(bboxes_xyz.dim()==2)
        num_p, _ = bboxes_xyz.size()
        b_ = votes_assignment.size(0)
        new_bbox_xyz = torch.stack([torch.index_select(bboxes_xyz,0,votes_assignment[k]) for k in range(b_)]) # keep dim as bboxes_xyz
    return new_bbox_xyz
"""

class PoolingLayer:
    def __init__(self, block_elm, input_channels, out_channels, k):
        self.out_channels = out_channels
        self.input_channels = input_channels
        self.k = k
        self.block_elm = block_elm

    def fps(self,d,startidx = None):
        batch_size = self.block_elm.batch_size
        num_of_points = self.block_elm.num_of_points

        if (startidx is None): # this
            idx = torch.randint(low=0,high=num_of_points, size=(1,), device='cuda')
        else:
            idx = torch.as_tensor([startidx],dtype=torch.int32, device='cuda')

        idx = idx.repeat(batch_size)  # --> (b,)

        if (self.k == 1):
            idx = idx.unsqueeze(1)
        else:

            gather = gather_nd(d,idx.view(batch_size,-1),num_of_points,batch_size).squeeze(1) #--(b,num_of_points)
            # debugPrint(gather.size())
            idx = torch.stack([idx, torch.argmax(gather, 1)], dim=1) # --> (b,2), first sample point idx and the furthest unsampled point with it
            # debugPrint(idx.size())
            for step in range(2, max(self.k, 2)):
                gather = gather_nd(d, idx, num_of_points, batch_size) # (b,step,num_points)
                idx = torch.cat([idx, torch.argmax(torch.min(gather,dim=1)[0],dim=1,keepdim=True)], dim=1) #(b,step+1)

        return idx  # (b,k)

    def get_layer(self,network,use_fps,startidx = None,is_subsampling = False):
        batch_size = self.block_elm.batch_size
        num_of_points = self.block_elm.num_of_points
        _,_,num_channels = network.size()
        distances = self.block_elm.get_distance_matrix()
        with torch.no_grad():
            if use_fps:  # True
                idx=self.fps(distances,startidx)
            else:
                idx=torch.arange(self.k,device='cuda').view(1,-1).repeat(batch_size,1) # shape (b,k)
        
        pooled_points_pl = gather_nd(self.block_elm.points_pl,idx,3,batch_size)

        if is_subsampling: # False
            return pooled_points_pl, gather_nd(network,idx,num_channels,batch_size)
        else:
            center_indexes = torch.argmin(gather_nd(distances,idx,num_of_points,batch_size),dim=1) # (b,num_points)
            tensor1=gather_nd(idx,center_indexes,1,batch_size).view(batch_size,1,-1)
            # debugPrint(tensor1.size()) # (b,1,n)
            # print("idx.unsqueeze(2) shape={} ,type={}".format(idx.unsqueeze(2).size(),idx.type())) # (b,k,1)
            tensor2=torch.eq(tensor1,idx.unsqueeze(2)).float().unsqueeze(3)
            # debugPrint(tensor2.size()) #(b,k,n,1)
            tensor3=torch.mul(tensor2, network.unsqueeze(1))
            pooled_network = torch.max(tensor3, dim=2)[0] # zhuyijie @2019.10.2
            # debugPrint(pooled_network.size())
            
            return pooled_points_pl, pooled_network
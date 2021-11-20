import torch
import torch.nn as nn
import numpy as np

class ConvElements:
    def __init__(self, points_pl, sigma, spacing, kernel_sigma_factor):

        self.points_pl = points_pl
        self.batch_size, self.num_of_points, _ = self.points_pl.size()
        self.sigma_f = sigma
        self.sigma_k =  kernel_sigma_factor * self.sigma_f
        self.spacing = spacing # 2.0
        self.combined_sigma = np.sqrt(self.sigma_f * self.sigma_f + self.sigma_k * self.sigma_k)
        self.num_of_centers = 3
        self.num_of_translations = self.num_of_centers ** 3  #3^{3}=27
        self.kernel_translations = self.get_kernel_translations(spacing)

    def get_kernel_translations(self,spacing):
        """
        return kernel_translations, with shape (b, num_of_translations, 3)
        """
        grid = torch.linspace(-spacing * self.sigma_k, spacing * self.sigma_k, steps=3, device='cuda')
        return torch.stack(torch.meshgrid(grid, grid, grid)).view(3,-1).transpose(0,1).view(-1,3).repeat(self.batch_size,1,1)

    def get_conv_matrix(self):
        c1 = tf.tile(tf.reshape(tf.square(tf.norm(self.points_pl, axis=2)), [self.batch_size,self.num_of_points, 1, 1]),
                     [1, 1, self.num_of_points, self.num_of_translations])
        c2 = tf.tile(tf.reshape(tf.square(tf.norm(self.points_pl, axis=2)), [self.batch_size,1, self.num_of_points, 1]),
                     [1, self.num_of_points, 1, self.num_of_translations])
        c3 = tf.tile(tf.reshape(tf.square(tf.norm(self.kernel_translations, axis=2)), [self.batch_size,1, 1, self.num_of_translations]),
                     [1, self.num_of_points, self.num_of_points, 1])

        c4 = tf.tile(-2.0 * tf.expand_dims(tf.matmul(self.points_pl, tf.transpose(self.points_pl,[0, 2, 1])),
                                       dim=3), [1,1, 1, self.num_of_translations])
        c5 = tf.tile(-2.0 * tf.expand_dims(tf.matmul(self.points_pl, tf.transpose(self.kernel_translations,[0,2,1])),
                                       dim=2), [1, 1, self.num_of_points, 1])
        c6 = tf.tile(2.0 * tf.expand_dims(tf.matmul(self.points_pl, tf.transpose(self.kernel_translations,[0, 2, 1])),
                                          dim=1), [1, self.num_of_points, 1, 1])

        C_add = c1 + c2 + c3 + c4 + c5 + c6


        C = tf.exp(-C_add / (2.0 * self.combined_sigma * self.combined_sigma))

        return C

    def get_interpolation_matrix(self):
        return torch.exp(-self.get_distance_matrix() / (2.0 * self.sigma_f * self.sigma_f))

    def get_distance_matrix(self):
        """
        return: D, with shape (b,n,n), D_ij is the squred distance between point i and j.
        """
        r = torch.sum(self.points_pl * self.points_pl, dim=2, keepdim=True)
        D = r - 2 * torch.matmul(self.points_pl, torch.transpose(self.points_pl, 1,2)) + r.transpose(1,2)
        return D


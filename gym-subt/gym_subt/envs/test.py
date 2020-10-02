import numpy as np
import torch

# data1 = np.full([10],1)
# data_all = np.tile(data1,(4,1))
# print(data_all)
# for i in range(5):
#     data1 = data1+1
#     data_all[:-1] = data_all[1:]
#     data_all[-1] = data1
#     data_r = data_all.reshape(-1)
#     data_o = torch.Tensor([data_r])
#     data_o = data_o.reshape(1, 4, -1)
#     # data_o = data_o.transpose(1,2)
#     print(data_o)
#     print(data_o.size())

# pos1 = np.array([[1,2,3]])
# pos1 = np.vstack((pos1,[4,5,6]))
# print(pos1)
# pos1 = np.vstack((pos1,[7,8,9]))
# print(pos1)
# pos1 = pos1[1:]
# print(pos1)

# poses = np.array([[64.01147457, 100.11660621, 0.13227352],
#                   [64.01147474, 100.11660647, 0.13227352],
#                   [64.01147496, 100.11660683, 0.13227352],
#                   [64.01147509, 100.11660703, 0.13227352],
#                   [64.01147528, 100.11660733, 0.13227352],
#                   [64.01147544, 100.11660759, 0.13227352],
#                   [64.01147561, 100.11660785, 0.13227352],
#                   [64.01147577, 100.1166081, 0.13227352],
#                   [64.01147599, 100.11660846, 0.13227352],
#                   [64.01147615, 100.11660871, 0.13227352]])

# last = np.array([64,0,0])
# last_poses =  np.tile(last,(10,1))
# print(poses - last_poses)
# dis = np.linalg.norm((last_poses - poses),axis=1)
# print(dis)
https://github.com/hlei-ziyan/SPH3D-GCN/


#This is an unoffical pytorch impelement of SPH3D-GCN(at ModelNet40) by Yijie Zhu @2019.10


To run the code, please compile the cuda-based operations in pytorch_ops folder using the command
```
cd pytorch_ops
(sudo) python setup.py install
```
The compiled util package named 'sph3d_ops_cuda'. You also need another compiled util package named ['pointnet2_cuda'](https://github.com/erikwijmans/Pointnet2_PyTorch).

Then,
```
cd $ROOT_DIR
python train_sph3d.py
```



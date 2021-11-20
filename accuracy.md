adaptive 200 epoch record    on 1024 points    Accuracy of the network on the test images: 92 %

origin knn 200 epoch record on 2048 points Accuracy of the network on the test images: 90.31 %

origin knn 200 epoch record on 1024 points    Accuracy of the network on the test images: 92.62 %

adaptive 200 epoch record    on 1024 points  Accuracy of the network on the test images: 92.29 %

origin knn 200 epoch record on 1024 points(modelnet40)     Accuracy: 88.82 %

adaptive knn 200 epoch record on 1024 points(modelnet40)     Accuracy: 88.98 %

adaptive knn 150 epoch record on 2048 points(shapenet)  Accuracy: 93.47%

origin knn 150 epoch record on 2048 points(shapenet) Accuracy: 93.55%

mesh knn 150 epoch record on 1024 points(modelnet40) Accuracy test images: 88.41 %



|knn type|acc|#points|dataset|
|----|----|----|----|
|adaptive manifold|92.29 %|1024|xx|
|adaptive manifold|88.98 %|1024|modelnet40|
|mesh knn|88.41 %|1024|modelnet40|
|origin|88.82 %|1024|modelnet40|
|origin|90.31 %|2048| xx |
|adaptive manifold|93.47 %|2048|shapenet|
|origin manifold|93.55 %|2048|shapenet|





####GeoNet####
./runs/May06_16-33-08_zhu  origin knn_16 100 epoch record on 1024 points(modelnet40) (20th) 89.18%   (40th)88.05%    (60th)90.92%    (80th)89.79%    (100th) 90.32%

./runs/May07_11-48-48_zhu adaptive knn_16 100 epoch record on 1024 points(modelnet40) (20th)88.13%  (40th)90.80%   (60th)90.48%   (80th)90.76%  (100th)91.05%

./runs/May08_19-07-33_zhu mesh knn_16 100 epoch record on 1024 points(modelnet40) (20th)88.78%  (40th)90.44%   (60th)90.32%   (80th) 91.41%  (100th)91.09%

./runs/May10_11-50-31_zhu  parall knn_16 100 epoch record on 1024 points(modelnet40)  (10th)88.29%  (20th)88.17%  (30th)89.51%    (70th)90.03%  (80th)89.91% (90th)90.40%  (100th)89.67%
./runs/May20_23-40-49_zhu SphericalGeoNet knn_16 100 epcho record on 1024 points(modelnet40) (60th)89.06% (80th)89.87% (100th)90.32%

./runs/May21_21-50-42_zhu spherical mesh 100epcho 1024 points(modelnet40) (20th)88.86% (40th)90.03% (60th)89.14% (80th)89.70% (100th) 89.99%





## DGCNN
总参数数量和：1809576

epoch 80 loss 0.046
Accuracy of the network on the test images: 91.69 %

epoch 90 loss 0.043
Accuracy of the network on the test images: 91.57 %

epoch 100 loss 0.042
Accuracy of the network on the test images: 91.37 %

## POINTNET
a) without feature_transform

总参数数量和：1614129

epoch 80 loss 0.035
Accuracy of the network on the test images: 88.21 %

epoch 90 loss 0.031
Accuracy of the network on the test images: 88.37 %

epoch 100 loss 0.027
Accuracy of the network on the test images: 88.17 %

## POINTCNN

a) 总参数数量和：275904

epoch 70 loss 0.291
Accuracy of the network on the test images: 82.94 %

epoch 80 loss 0.285
Accuracy of the network on the test images: 86.10 %

epoch 90 loss 0.283
Accuracy of the network on the test images: 84.08 %

epoch 100 loss 0.282
Accuracy of the network on the test images: 84.76 %

b) 总参数数量和：860348

epoch 40 loss 0.206
Accuracy of the network on the test images: 84.52 %

epoch 50 loss 0.175
Accuracy of the network on the test images: 85.70 %

epoch 60 loss 0.160
Accuracy of the network on the test images: 86.79 %

epoch 70 loss 0.150
Accuracy of the network on the test images: 88.01 %

epoch 80 loss 0.137
Accuracy of the network on the test images: 87.16 %

epoch 90 loss 0.137
Accuracy of the network on the test images: 86.87 %

epoch 100 loss 0.137
Accuracy of the network on the test images: 86.87 %






## SONET

总参数数量和：1032768 + 667944

epoch 80 loss 0.163
Accuracy of the network on the test images: 90.52 %

epoch 90 loss 0.153
Accuracy of the network on the test images: 90.52 %

epoch 100 loss 0.147
Accuracy of the network on the test images: 90.44 %

## KDNET

总参数数量和：3552616

epoch 10 loss 0.134
Accuracy of the network on the test images: 78.57 %

Accuracy of the network on the test images: 79.21 %

Accuracy of the network on the test images: 79.62 %

epoch 40 loss 0.017
Accuracy of the network on the test images: 79.82 %

epoch 50 loss 0.011
Accuracy of the network on the test images: 80.31 %

epoch 60 loss 0.006
Accuracy of the network on the test images: 80.75 %

epoch 70 loss 0.005
Accuracy of the network on the test images: 80.59 %

epoch 80 loss 0.005
Accuracy of the network on the test images: 81.04 %

epoch 90 loss 0.005
Accuracy of the network on the test images: 80.47 %

epoch 100 loss 0.005
Accuracy of the network on the test images: 80.96 %
2069228.745 ms has passed

|model|acc|source|
|---|---|---|
|DGCNN|91.69 %|[offical](https://github.com/WangYueFt/dgcnn)|
|POINTNET|88.37 %|[third-party](https://github.com/fxia22/pointnet.pytorch)|
|POINTCNN|88.01 %|[third-party](https://github.com/hxdengBerkeley/PointCNN.Pytorch)|
|SONET|90.52 %|[offical](https://github.com/lijx10/SO-Net)|
|KDNET|81.04 %|[third-party](https://github.com/fxia22/kdnet.pytorch)|


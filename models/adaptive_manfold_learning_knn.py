import os
import sys
import numpy as np
import scipy
import scipy.sparse as sp
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
import multiprocessing
from multiprocessing.dummy import Pool as ThreadPool
# from multiprocessing import Pool as ThreadPool
from time import clock, sleep
import math


def timeit(func):
    def wrapper(*args, **kwargs):
        starting_time = clock()
        result = func(*args, **kwargs)
        ending_time = clock()
        print('Duration: {}'.format(ending_time - starting_time))
        return result

    return wrapper


@timeit
def hello():
    hello_list = [i for i in range(3)]
    print(hello_list)

    def process(i):
        a = math.sqrt(i * i + 1)
        result = [i]
        return result

    pool = ThreadPool(4)
    results = pool.map(process, hello_list)
    pool.close()
    pool.join()
    print(results)


@timeit
def adaptive_knn(filename=None, savename=None, d=2, k_max=16, k_min=None):
    if (filename == None):
        print("need a file name")
        return
    modelnet10 = np.load(filename, encoding='latin1', allow_pickle=True)
    modelnet10_data = modelnet10.tolist()['data']  #(3991, 1024, 3)
    # modelnet10_label = modelnet10.tolist()['label'] #(3991,)
    # modelnet10_seg =  modelnet10.tolist()['seg_label'] #(n_model, 2048, C)
    del modelnet10

    print("the dataset shape is {}".format(modelnet10_data.shape))
    n_model, n_point, _ = modelnet10_data.shape
    print("k_max={}".format(k_max))
    start = n_model // 4 * 0
    end = n_model
    print('process start={},end={}'.format(start, end))
    # modelnet10_data=modelnet10_data[start:end]
    result_knn = []
    # d=2
    # k_max=16
    if k_min is None:  # 6
        k_min = d + 4
    yita = 0.32
    print('k_max={}'.format(k_max))
    print('k_min={}'.format(k_min))
    print('yita={}'.format(yita))
    for model_i in range(start, end):
        if (model_i % 100 == 0):
            print(model_i)
        X = modelnet10_data[model_i]  # i-th model, shape=(1024,3)
        nbrs = NearestNeighbors(n_neighbors=k_max + 1,
                                algorithm='ball_tree').fit(X)
        distances, indices = nbrs.kneighbors(X)
        indx = indices[:, 1:]  # nearest 16 neighbors
        n, m = X.shape

        # STEP 1
        rho = [[] for i in range(n)]
        result_indx = [[] for i in range(n)]
        for i in range(n):
            flag = 0
            tmp = indx[i]
            X_k = np.transpose(X[tmp])  # (nfeatures, npoints)
            for j in range(k_max, k_min, -1):
                x_i = np.mean(X_k, axis=1).reshape(-1, 1)  # (nfeatures, 1)
                X_i = X_k - x_i

                # compute singular value  d=2 (8), k_min=d+4 , yita=0.32
                u, sigma, v = np.linalg.svd(X_i, full_matrices=False)
                sigma = sigma**2

                r_i = np.sqrt(np.sum(sigma[2:]) / np.sum(sigma[:2]))
                if r_i < yita:
                    result_indx[i] = indx[i][:j]
                    rho[i].append(r_i)
                    flag = 1
                    break
                rho[i].append(r_i)
                X_k = X_k[:, :-1]
            if flag == 0:
                max_k = np.argmin(rho[i])
                result_indx[i] = indx[i][:k_max - max_k]

        # STEP 2
        for i in range(n):
            X1 = X[result_indx[i]].copy()  # the neighborhood of i-th point
            x2_indx = indx[i][len(result_indx[i]):]
            X2 = X[x2_indx]  # (N_SMAPLE, N_FEATURE)
            if X2.shape[0] == 0:
                continue
            pca = PCA(n_components=2)
            pca.fit(X1)
            # pca_score = pca.explained_variance_ratio_
            V = pca.components_
            # pca_X1=pca.fit_transform(X1)
            mypca_X2 = np.dot(X2 - pca.mean_, V.T)  # (N_SAMPLE, N_FEATURE')
            recover_X2 = pca.inverse_transform(mypca_X2)

            do_select = np.linalg.norm(
                X2 - recover_X2,
                axis=1) <= yita * np.linalg.norm(mypca_X2, axis=1)
            NE = [
                x2_indx[idx] for idx, ii in enumerate(do_select) if ii == True
            ]  # Neighborhood Expansion
            if NE != []:
                result_indx[i] = np.append(result_indx[i], NE)
            # print(np.linalg.norm(X2-recover_X2,axis=1))
            # print(yita*np.linalg.norm(mypca_X2,axis=1))
            # print(do_select)
            # print(np.linalg.norm(np.dot(X1-pca.mean_,V.T)-pca_X1))
            # print(np.linalg.norm(pca.inverse_transform(pca_XX)-XX))
        result_knn.append(result_indx)

    if (len(result_knn) != end -
            start):  #n_moddel(list), n_points(list), n_neiberhood(np.array)
        raise Exception("len of result_knn!=n_model")

    # convert list  to sparse matrix
    for i in range(end - start):
        data = result_knn[i]
        row_ = []
        col_ = []
        for row, cols in enumerate(data):
            row_ += [row for _ in cols]
            col_ += list(cols)
        sp_data = sp.csr_matrix(
            (np.ones(len(row_), dtype='int32'), (row_, col_)),
            shape=(n_point, n_point))
        result_knn[i] = sp_data

    # savename='./modelnet/data/modelNet40_train_16nn_GM_adaptive_knn_sparse.npy'
    if (savename == None):
        # e.g.
        # filename = './modelnet/data/modelNet10_train_16nn_GM.npy'
        # savename = ./modelnet/data/modelNet10_train_16nn_GM_adaptive_knn.npy
        savename = "".join(
            filename.split('.npy')) + "_adaptive_knn_sparse_4.npy"
    # shapenet 50
    np.save(
        savename,
        np.array({  #'data': modelnet10_data,
            'graph': result_knn
            #'seg_label': modelnet10_seg,
            #'label': modelnet10_label
        }))  #'label_dict':test_modelnet10_label_dict,
    # np.save(savename, np.array(result_knn))
    print("saved to {}".format(savename))


def do_work(result_knn, modelnet10_data, start, stop, k_max, d):
    result_knn_ = []
    for model_i in range(start, stop):
        if (model_i % 10 == 0):
            print(model_i)
        X = modelnet10_data[model_i]  # i-th model, shape=(1024,3)
        nbrs = NearestNeighbors(n_neighbors=k_max + 1,
                                algorithm='ball_tree').fit(X)
        distances, indices = nbrs.kneighbors(X)
        indx = indices[:, 1:]  # nearest 16 nerighbors

        # d=2
        # k_max=16
        k_min = d + 4  # 6
        n, m = X.shape
        # STEP 1
        rho = [[] for i in range(n)]
        result_indx = [[] for i in range(n)]
        yita = 0.32
        for i in range(n):
            flag = 0
            tmp = indx[i]
            X_k = np.transpose(X[tmp])  #(nfeatures, npoints)
            for j in range(k_max, k_min, -1):
                x_i = np.mean(X_k, axis=1).reshape(-1, 1)  #(nfeatures, 1)
                X_i = X_k - x_i

                # compute singular value,  d=2 (8), k_min=d+4 , yita=0.32
                u, sigma, v = np.linalg.svd(X_i, full_matrices=False)
                sigma = sigma**2

                r_i = np.sqrt(np.sum(sigma[2:]) / np.sum(sigma[:2]))
                if r_i < yita:
                    result_indx[i] = indx[i][:j]
                    rho[i].append(r_i)
                    flag = 1
                    break
                rho[i].append(r_i)
                X_k = X_k[:, :-1]
            if flag == 0:
                max_k = np.argmin(rho[i])
                result_indx[i] = indx[i][:k_max - max_k]

        # STEP 2
        for i in range(n):
            X1 = X[result_indx[i]].copy()  # neighborhood of i-th point
            x2_indx = indx[i][len(result_indx[i]):]
            X2 = X[x2_indx]  #(N_SMAPLE, N_FEATURE)
            if X2.shape[0] == 0:
                continue
            pca = PCA(n_components=2)
            pca.fit(X1)
            # pca_score = pca.explained_variance_ratio_
            V = pca.components_
            # pca_X1=pca.fit_transform(X1)
            mypca_X2 = np.dot(X2 - pca.mean_, V.T)  #(N_SAMPLE, N_FEATURE')
            recover_X2 = pca.inverse_transform(mypca_X2)

            do_select = np.linalg.norm(
                X2 - recover_X2,
                axis=1) <= yita * np.linalg.norm(mypca_X2, axis=1)
            NE = [
                x2_indx[idx] for idx, ii in enumerate(do_select) if ii == True
            ]  # Neighborhood Expansion
            # print("i={}, orig ks={}, NE={}".format(i,X1.shape[0],NE))
            # result_indx[i]+=NE
            if NE != []:
                result_indx[i] = np.append(result_indx[i], NE)
        result_knn_.append(result_indx)
    result_knn[start:stop] = result_knn_


@timeit
def multi_threads_adaptive_knn(filename=None, savename=None, d=2, k_max=16):
    if (filename == None):
        print("need a file name")
        return
    modelnet10 = np.load(filename, encoding='latin1')
    modelnet10_data = modelnet10.tolist()['data']  #(3991, 1024, 3)
    modelnet10_label = modelnet10.tolist()['label']  #(3991,)
    print("the dataset shape is {}".format(modelnet10_data.shape))
    n_model, n_point, _ = modelnet10_data.shape
    print("k_max={}".format(k_max))
    # n_model=40
    result_knn = []
    for model_i in range(n_model):
        if (model_i % 100 == 0):
            print(model_i)
        X = modelnet10_data[model_i]  # model_i-th model  shape=(1024,3)
        nbrs = NearestNeighbors(n_neighbors=k_max + 1,
                                algorithm='ball_tree').fit(X)
        distances, indices = nbrs.kneighbors(X)
        indx = indices[:, 1:]  # nearset 16 neighbors

        # d=2
        # k_max=16
        k_min = d + 4  #6
        n, m = X.shape

        # STEP 1
        rho = [[] for i in range(n)]
        result_indx = [[] for i in range(n)]
        yita = 0.32
        for i in range(n):
            flag = 0
            tmp = indx[i]
            X_k = np.transpose(X[tmp])  #(nfeatures, npoints)
            for j in range(k_max, k_min, -1):
                x_i = np.mean(X_k, axis=1).reshape(-1, 1)  #(nfeatures, 1)
                X_i = X_k - x_i

                #计算奇异值  d=2 (8), k_min=d+4 , yita=0.32
                u, sigma, v = np.linalg.svd(X_i, full_matrices=False)
                sigma = sigma**2

                r_i = np.sqrt(np.sum(sigma[2:]) / np.sum(sigma[:2]))
                if r_i < yita:
                    result_indx[i] = indx[i][:j]
                    rho[i].append(r_i)
                    flag = 1
                    break
                rho[i].append(r_i)
                X_k = X_k[:, :-1]
            if flag == 0:
                max_k = np.argmin(rho[i])
                result_indx[i] = indx[i][:k_max - max_k]

        # STEP 2
        for i in range(n):
            X1 = X[result_indx[i]].copy()  #第i个点的neighborhood
            x2_indx = indx[i][len(result_indx[i]):]
            X2 = X[x2_indx]  #(N_SMAPLE, N_FEATURE)
            if X2.shape[0] == 0:
                continue
            pca = PCA(n_components=2)
            pca.fit(X1)
            # pca_score = pca.explained_variance_ratio_
            V = pca.components_
            # pca_X1=pca.fit_transform(X1)
            mypca_X2 = np.dot(X2 - pca.mean_, V.T)  #(N_SAMPLE, N_FEATURE')
            recover_X2 = pca.inverse_transform(mypca_X2)

            do_select = np.linalg.norm(
                X2 - recover_X2,
                axis=1) <= yita * np.linalg.norm(mypca_X2, axis=1)
            NE = [
                x2_indx[idx] for idx, ii in enumerate(do_select) if ii == True
            ]  # Neighborhood Expansion
            # print("i={}, orig ks={}, NE={}".format(i,X1.shape[0],NE))
            # result_indx[i]+=NE
            if NE != []:
                result_indx[i] = np.append(result_indx[i], NE)
        # return result_indx
        result_knn.append(result_indx)

    # pool = ThreadPool(4)
    # # result_knn = pool.map(process, range(n_model))
    # pool.close()
    # pool.join()
# with multiprocessing.Manager() as MG:   #重命名
# mydict=MG.dict()   #主进程与子进程共享这个字典
# mydict["array"]=np.zeros((3,3)).tolist()


#         result_knn=MG.list(result_knn)   #主进程与子进程共享这个List
# #        mylist.append([1,2])
#         modelnet10_data=MG.list(modelnet10_data)   #主进程与子进程共享这个List
#         # 多线程部分
#         #result=multiprocessing.Manager().dict()
#         #result['par']=Par
#         #result['num']=xy_arrays
#         threads=[]
#         t1 =multiprocessing.Process(target=do_work,args=(result_knn,modelnet10_data,0,n_model//4,k_max,d))
#         threads.append(t1)
#         t2 =multiprocessing.Process(target=do_work,args=(result_knn,modelnet10_data,n_model//4,n_model//4*2,k_max,d))
#         threads.append(t2)
#         t3 =multiprocessing.Process(target=do_work,args=(result_knn,modelnet10_data,n_model//4*2,n_model//4*3,k_max,d))
#         threads.append(t3)
#         t4 =multiprocessing.Process(target=do_work,args=(result_knn,modelnet10_data,n_model//4*3,n_model,k_max,d))
#         threads.append(t4)
#         [t.start() for t in threads]
#         [t.join() for t in threads]
#         print(result_knn)

    if (len(result_knn) != n_model): 
        raise Exception("len of result_knn!=n_model")
    # convert list  to sparse matrix
    for i in range(n_model):
        data = result_knn[i]
        row_ = []
        col_ = []
        for row, cols in enumerate(data):
            row_ += [row for _ in cols]
            col_ += list(cols)
        sp_data = sp.csr_matrix(
            (np.ones(len(row_), dtype='int32'), (row_, col_)),
            shape=(n_point, n_point))
        result_knn[i] = sp_data

    # savename='./modelnet/data/modelNet40_train_16nn_GM_adaptive_knn_sparse.npy'
    if (savename == None):
        # e.g.
        # filename = './modelnet/data/modelNet10_train_16nn_GM.npy'
        # savename = ./modelnet/data/modelNet10_train_16nn_GM_adaptive_knn.npy
        savename = "".join(filename.split('.npy')) + "_adaptive_knn_sparse.npy"
    # shapenet 50
    np.save(savename,
            np.array({
                'data': modelnet10_data,
                'graph': result_knn,
                'seg_label': modelnet10.tolist()['seg_label'],
                'label': modelnet10_label
            }))  #'label_dict':test_modelnet10_label_dict,
    # np.save(savename, np.array(result_knn))
    print("saved to {}".format(savename))

if __name__ == '__main__':
    filename = './modelnet/data/modelNet40_test_16nn_GM.npy'
    savename = "".join(filename.split('.npy')) + "_adaptive_32knn_sparse.npy"
    adaptive_knn(filename=filename, savename=savename, k_max=32, k_min=16)
    # abc=np.load(savename, allow_pickle=True)

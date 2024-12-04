import numpy as np
import copy
import random
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model
from sklearn.model_selection import KFold



def DC(D,mu,T0,g):
    U,S,V = np.linalg.svd(D)
    T1 = np.zeros(np.size(T0))
    for i in range(1,100):
        T1 = DCInner(S,mu,T0,g)
        err = np.sum(np.square(T1-T0))#The square of the vector 2-parameter
        if err < 1e-6:
            break
        T0 = T1
    l_1 = np.dot(U, np.diag(T1))
    l = np.dot(l_1, V)
    return l,T1

def DCInner(S,mu,T_k,gam):
    lamb = 1/mu
    grad = (1+gam)*gam/(np.square(gam+T_k))
    T_k1 = S-lamb*grad
    T_k1[T_k1<0]=0
    return T_k1

def errorsol(T_1,U_1,V_1,mu,min):
    alpha = 1 / mu
    G = T_1-U_1.dot(np.transpose(V_1))
    G1=np.sqrt(np.sum(np.square(G),1))
    G1[G1 == 0] = alpha#Ensure that the next line to do the division will not make an error due to the divisor G1==0 and will not have an effect on the final result.
    G2 = (G1 - alpha)/ G1
    G3 = (G1 > alpha) * G2
    G4 = G3[:min]
    E = np.dot(G, np.diag(G4))
    return E

def GAMA(D,l,r,m3): # D=U*VT+S
    row_num = D.shape[0]
    column_num = D.shape[1]
    min_row_column = min(row_num,column_num)
    # Variable parameters
    gamma = 0.01
    lamda = l
    tol = 1e-3
    # Iterative initialization
    U, S_1, V = np.linalg.svd(D)
    U = U[:,0:min_row_column]
    S_sqrt = np.sqrt(np.diag(S_1))
    # print(np.shape(S_sqrt))
    U = U.dot(S_sqrt)
    V = S_sqrt.dot(V)
    V = np.transpose(V)
    #Initialize U and V
    m, n = np.shape(D)
    S = np.zeros((m, n))
    Y = np.zeros((m, n))
    rho = r
    mu = m3
    for i in range(0, 500):
        T = D + Y/rho
        T1 = (T-S).dot(V)
        A, US, B = np.linalg.svd(T1)
        A = A[:, 0:min_row_column]
        U = A.dot(B)
        # U = ((D-S-Y).dot(V))
        # U = U.dot(np.transpose(V).dot(V))

        Q = (np.transpose(T-S)).dot(U)
        sig = np.zeros(min(m, n))
        V, sig = DC(copy.deepcopy(Q), rho/lamda, copy.deepcopy(sig), gamma)

        S = errorsol(copy.deepcopy(T), copy.deepcopy(U), copy.deepcopy(V), rho,min_row_column)

        Y= Y+rho*(D-U.dot(np.transpose(V))-S)
        rho = mu*rho

        sigma = np.linalg.norm(D-U.dot(np.transpose(V))-S,'fro')
        RRE = sigma/np.linalg.norm(D,'fro')

        if RRE < tol:
            break
    # print(np.linalg.matrix_rank(U.dot(np.transpose(V))))
    return U, V



# MLP
def BuildModel(train_x, train_y):
    # Get the dimensions of the input data
    l = len(train_x[1])
    inputs = Input(shape=(l,))

    # Defining the network structure
    x = Dense(128, activation='relu')(inputs)
    x = Dense(64, activation='relu')(x)
    predictions = Dense(1, activation='sigmoid')(x)

    # Creating Models
    model = Model(inputs=inputs, outputs=predictions)
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    model.fit(train_x, train_y, epochs=25)  # training model
    return model


# Calculate the positive and negative sample coordinates in the correlation matrix
def get_all_the_samples(A):
    m, n = A.shape
    pos = []
    neg = []
    for i in range(m):
        for j in range(n):
            if A[i, j] != 0:  # If the element is not 0, it is treated as a positive case
                pos.append([i, j, 1])
            else:
                neg.append([i, j, 0])
    samplesz = random.sample(pos, len(pos)) # disorder
    samplesz = np.array(samplesz)
    samplesf = random.sample(neg, len(pos))
    samplesf = np.array(samplesf)
    return samplesz,samplesf


# Predictive Functions
def predict(model, feature_m, feature_d, feature_MFm, feature_MFd, all_samples, D):
    vect_len1 = feature_m.shape[0]  # Dimensions of drug similarity characterization
    vect_len2 = feature_d.shape[0]  #  Dimensions of miRNA similarity characterization

    test_N = all_samples.shape[0]
    test_feature = np.zeros([test_N, vect_len1 + vect_len2 + 2 * D])
    test_label = np.zeros(test_N)

    for i in range(test_N):
        test_feature[i, 0:vect_len2] = feature_MFm[all_samples[i, 0], :]
        test_feature[i, vect_len2:(vect_len2 + vect_len1)] = feature_m[all_samples[i, 0], :]
        test_feature[i, (vect_len2 + vect_len1):(vect_len1 + vect_len2 + vect_len2)] = feature_MFd[all_samples[i, 1], :]
        test_feature[i, (vect_len1 + vect_len2 + vect_len2):(vect_len1 + vect_len2 + 2 * vect_len2)] = feature_d[
                                                                                                       all_samples[
                                                                                                           i, 1], :]
        test_label[i] = all_samples[i, 2]

    # executing forecast
    y_score = model.predict(test_feature)
    return y_score, test_label


def run_model():
    A = np.loadtxt(r"../dataset/guanlianmatrix.txt", dtype=int)
    SM = np.loadtxt(r'../dataset/drugsimilarity.txt', dtype=float)
    miRNA = np.loadtxt(r'../dataset/miRNAsimilarity.txt', dtype=float)
    A_2 = copy.deepcopy(A)
    SM_2 = copy.deepcopy(SM)
    miRNA_2 = copy.deepcopy(miRNA)

    # Removing noise from drug/miRNA similarity matrices
    l_1 = 0.1
    l_3 = 2
    r = 50
    m = 10
    M_1, D_1 = GAMA(SM_2, l_1, r, m)
    SM_denoise = M_1.dot(np.transpose(D_1))
    M_2, D_2 = GAMA(miRNA_2, l_1, r, m)
    m_denoise = M_2.dot(np.transpose(D_2))
    M_3, D_3 = GAMA(A_2, l_3, r, m)


    D = 140
    samplesz, samplesf = get_all_the_samples(A)

    A = []
    B = []
    C = []
    E = []
    kf = KFold(n_splits=5, shuffle=True)
    for train_index, test_index in kf.split(samplesz):
        train_samples_z = samplesz[train_index]
        test_samples_z = samplesz[test_index]
        A.append(test_samples_z)
        B.append(train_samples_z)

    for train_index, test_index in kf.split(samplesf):
        train_samples_f = samplesf[train_index]
        test_samples_f = samplesf[test_index]
        C.append(test_samples_f)
        E.append(train_samples_f)

    test_samples_all = [np.concatenate((l1, l2), axis=0) for l1, l2 in zip(A, C)]
    train_samples_all = [np.concatenate((l1, l2), axis=0) for l1, l2 in zip(B, E)]



    for train_samples, test_samples in zip(train_samples_all, test_samples_all):
        feature_m = SM_denoise  #
        feature_d = m_denoise
        feature_MFm, feature_MFd = M_3, D_3
        # emerge the miRNA feature and disease feature
        vect_len1 = feature_m.shape[0]  # 431
        vect_len2 = feature_d.shape[0]  # 140
        # print(vect_len1)
        train_n = train_samples.shape[0]  # 训练样本个数
        train_feature = np.zeros([train_n, vect_len1 + vect_len2 + 2 * D])
        train_label = np.zeros([train_n])
        # for i in range(train_n):
        #     train_feature[i, 0:vect_len1] = feature_m[train_samples[i, 0], :]
        #     train_feature[i, vect_len1:(vect_len1 + vect_len2)] = feature_d[train_samples[i, 1], :]
        #     train_feature[i, (vect_len1 + vect_len2):(vect_len1 + vect_len2 + D)] = feature_MFm[train_samples[i, 0], :]
        #     train_feature[i, (vect_len1 + vect_len2 + D):(vect_len1 + vect_len2 + 2 * D)] = feature_MFd[train_samples[i, 1], :]
        #     train_label[i] = train_samples[i, 2]
        for i in range(train_n):
            train_feature[i, 0:vect_len2] = feature_MFm[train_samples[i, 0], :]
            train_feature[i, vect_len2:(vect_len2 + vect_len1)] = feature_m[train_samples[i, 0], :]
            train_feature[i, (vect_len2 + vect_len1):(vect_len1 + vect_len2 + vect_len2)] = feature_MFd[
                                                                                            train_samples[i, 1], :]
            train_feature[i, (vect_len1 + vect_len2 + vect_len2):(vect_len1 + vect_len2 + 2 * vect_len2)] = feature_d[
                                                                                                            train_samples[
                                                                                                                i, 1],
                                                                                                            :]
            train_label[i] = train_samples[i, 2]
        # get the featrue vectors of test samples
        test_N = test_samples.shape[0]
        test_feature = np.zeros([test_N, vect_len1 + vect_len2 + 2 * D])
        test_label = np.zeros(test_N)
        # for i in range(test_N):
        #     test_feature[i, 0:vect_len1] = feature_m[test_samples[i, 0], :]
        #     test_feature[i, vect_len1:(vect_len1 + vect_len2)] = feature_d[test_samples[i, 1], :]
        #     test_feature[i, (vect_len1 + vect_len2):(vect_len1 + vect_len2 + D)] = feature_MFm[test_samples[i, 0], :]
        #     test_feature[i, (vect_len1 + vect_len2 + D):(vect_len1 + vect_len2 + 2 * D)] = feature_MFd[test_samples[i, 1], :]
        #     test_label[i] = test_samples[i, 2]
        for i in range(test_N):
            test_feature[i, 0:vect_len2] = feature_MFm[test_samples[i, 0], :]
            test_feature[i, vect_len2:(vect_len2 + vect_len1)] = feature_m[test_samples[i, 0], :]
            test_feature[i, (vect_len2 + vect_len1):(vect_len1 + vect_len2 + vect_len2)] = feature_MFd[
                                                                                           test_samples[i, 1], :]
            test_feature[i, (vect_len1 + vect_len2 + vect_len2):(vect_len1 + vect_len2 + 2 * vect_len2)] = feature_d[
                                                                                                           test_samples[
                                                                                                               i, 1], :]
            test_label[i] = test_samples[i, 2]
        # train the neural network model
        model = BuildModel(train_feature, train_label)
        y_score = np.zeros(test_N)
        y_score = model.predict(test_feature)



# running model
run_model()

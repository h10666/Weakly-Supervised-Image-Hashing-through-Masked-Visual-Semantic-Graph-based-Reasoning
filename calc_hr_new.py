import numpy as np

def calc_hammingDist(B1, B2):
    q = B2.shape[1]
    distH = 0.5 * (q - np.dot(B1, B2.transpose()))
    return distH

def calc_map(qB, rB, queryL, retrievalL, knn):
    # qB: {-1,+1}^{mxq}
    # rB: {-1,+1}^{nxq}
    # queryL: {0,1}^{mxl}
    # retrievalL: {0,1}^{nxl}
    num_query = queryL.shape[0]
    p = 0
    map = 0
    r_2 = 0.0
    #print(num_query)
    # print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')

    for iter in range(num_query):
        # gnd = (np.dot(queryL[iter, :], retrievalL.transpose()) > 0).astype(np.float32)
        gnd = (np.dot(queryL[iter, :], retrievalL.transpose()) > 0).astype(np.int64)
        tsum = np.sum(gnd)
        if tsum == 0:
            continue
        hamm = calc_hammingDist(qB[iter, :], rB)
        #print(hamm)
        true_in_r_2 = (hamm <= 2)
        #if np.where(true_in_r_2 == True)[0].shape[0] != 0:
        #print(np.sum(true_in_r_2))
        if np.sum(true_in_r_2) !=0:
            r_2_ = np.sum(true_in_r_2 * gnd) / np.sum(true_in_r_2)
        else:
            r_2_ = 0.0
        ind = np.argsort(hamm)
        gnd = gnd[ind]
        p_ = np.sum(gnd[:knn] / knn)
        count = np.linspace(1, tsum, tsum)

        tindex = np.asarray(np.where(gnd == 1)) + 1.0
        map_ = np.mean(count / (tindex))
        # print(map_)
        map = map + map_
        p = p + p_
        r_2 = r_2 + r_2_
    map = map / num_query
    p = p / num_query
    #print(r_2)
    r_2 = r_2/ num_query
    # print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')

    return map, p, r_2

def calc_topMap(qB, rB, queryL, retrievalL, topk):
    # qB: {-1,+1}^{mxq}
    # rB: {-1,+1}^{nxq}
    # queryL: {0,1}^{mxl}
    # retrievalL: {0,1}^{nxl}
    num_query = queryL.shape[0]
    topkmap = 0
    # print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    for iter in range(num_query):
        gnd = (np.dot(queryL[iter, :], retrievalL.transpose()) > 0).astype(np.float32)
        hamm = calc_hammingDist(qB[iter, :], rB)
        ind = np.argsort(hamm)
        gnd = gnd[ind]

        tgnd = gnd[0:topk]
        tsum = np.sum(tgnd)
        if tsum == 0:
            continue
        count = np.linspace(1, tsum, tsum)

        tindex = np.asarray(np.where(tgnd == 1)) + 1.0
        topkmap_ = np.mean(count / (tindex))
        # print(topkmap_)
        topkmap = topkmap + topkmap_
    topkmap = topkmap / num_query
    # print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    return topkmap

if __name__=='__main__':
    pass
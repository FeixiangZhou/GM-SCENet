import numpy as np

class PCK(object):
    """docstring for PCK"""
    def __init__(self,njoints):
        super(PCK, self).__init__()
        self.njoints = njoints

    def calc_dists(self, preds, target, normalize):
        preds = preds.astype(np.float32)
        target = target.astype(np.float32)
        dists = np.zeros((preds.shape[1], preds.shape[0])) # 4, batchsize
        for n in range(preds.shape[0]):
            for c in range(preds.shape[1]):
                if target[n, c, 0] > 0 and target[n, c, 1] > 0: #remove invlid points
                    normed_preds = preds[n, c, :] / normalize[n]
                    normed_targets = target[n, c, :] / normalize[n]
                    dists[c, n] = np.linalg.norm(normed_preds - normed_targets) #L2范数
                    # dists[c, n] = np.power(dists[c, n], 2)
                    # dists[c, n] = dists[c, n] / 10 / 10
                else:
                    # print("NaN" % [n,c])
                    dists[c, n] = -1 #nan points
        return dists

    def dist_acc(self, dists, thr=0.5):
         '''
         :param dists:
         :param thr:  Normalized Distance 0：0.01:0.5
         :return:
         '''
         ''' Return percentage below threshold while ignoring values with a -1 '''
         dist_cal = np.not_equal(dists, -1)   #return true or false
         num_dist_cal = dist_cal.sum()
         if num_dist_cal > 0:
             return np.less(dists[dist_cal], thr).sum() * 1.0 / num_dist_cal
         else:
             return -1

    def get_peak_points(self, heatmaps):
        """
        由heatmap计算keypoints坐标
        :param heatmaps: numpy array (N,15,96,96)
        :return:numpy array (N,15,2)
        """
        N, C, H, W = heatmaps.shape
        all_peak_points = []
        for i in range(N):
            peak_points = []
            for j in range(C):
                yy, xx = np.where(heatmaps[i, j] == heatmaps[i, j].max())
                y = yy[0]
                x = xx[0]
                peak_points.append([x, y])
            all_peak_points.append(peak_points)
        all_peak_points = np.array(all_peak_points)
        return all_peak_points

    def eval(self, pred, target, thr): # (N*4*2)
        '''
        Calculate accuracy according to PCK,
        but uses ground truth heatmap rather than x,y locations
        First value to be returned is average accuracy across 'idxs',
        followed by individual accuracies
        '''

        idx = list(range(self.njoints))
        norm = 1.0

        # h = 480 # size of heatmap for mouse
        # w = 640

        # h = 160  # size of heatmap for zebra
        # w = 160

        h = 192  # size of heatmap for fly
        w = 192

        # norm = np.ones((pred.shape[0], 2)) * np.array([w, h]) / 7  #归一化[2,2] 24*32，变为【32，24】，行在前面    #   zebra=7
        norm = np.ones((pred.shape[0], 2)) * np.array([w, h]) / 6  # 归一化[2,2] 24*32，变为【32，24】，行在前面    #   fly=8 norm=6

        # norm = np.ones((pred.shape[0], 2)) * np.array([w, h]) / 10  #归一化[2,2] 24*32，变为【32，24】，行在前面     #mouse
        dists = self.calc_dists(pred, target, norm)

        acc = np.zeros((len(idx) + 1))
        avg_acc = 0
        cnt = 0

        for i in range(len(idx)):

            acc[i + 1] = self.dist_acc(dists[idx[i]], thr) #dist[i]为第i个points dist[channels,bs]
            if acc[i + 1] >= 0:
                avg_acc = avg_acc + acc[i + 1]
                cnt += 1

        avg_acc = avg_acc / cnt if cnt != 0 else -1 #无效点的输出-1
        if cnt != 0:
            acc[0] = avg_acc
        else:
            acc[0] = -1
        return acc, avg_acc,cnt


targetheatmaparray = np.random.rand(2,4,2,2)

predheatmap = np.random.rand(2,4,2,2)



def calc_oks(preds, target, normalize):
    preds = preds.astype(np.float32)
    target = target.astype(np.float32)
    dists = np.zeros((preds.shape[1], preds.shape[0])) # 4, batchsize
    oks = np.zeros((preds.shape[1], preds.shape[0]))  # 4, batchsize
    for n in range(preds.shape[0]):
        for c in range(preds.shape[1]):
            if target[n, c, 0] > 0 and target[n, c, 1] > 0: #remove invlid points
                normed_preds = preds[n, c, :]
                normed_targets = target[n, c, :]
                dists[c, n] = np.linalg.norm(normed_preds - normed_targets) #L2范数dists[4,1]
                dists[c, n] = np.power(dists[c, n], 2)
                oks[c, n] = np.exp(-dists[c, n] / normalize / normalize)
                # print( oks[c, n])

            else:
                # print("NaN" % [n,c])
                oks[c, n] = -1 #nan points
    return oks


def updatepred(pointoks, pred, numpred):

    updatep = np.zeros((1, 4, 2))
    max = 0
    maxlist = []
    indexlist = []
    t = 0
    for i in range(4):
        for j in range(numpred):  # num pred
            if (pointoks[j][i][0] > max):
                max = pointoks[j][i][0]
                t = j
        maxlist.append(max)  # max=0 invalid point
        indexlist.append(t)
        max = 0
        updatep[0][i] = pred[indexlist[i]][0][i]
    return updatep

def avepred(pointoks, pred, numpred, thr):

    # avepoint = np.zeros((1, 4, 2))    # mouse=4
    # avepoint = np.zeros((1, 9, 2))  # zebra=9
    avepoint = np.zeros((1, 32, 2))  # zebra=9
    count = [0,0,0,0,0]

    indexlist = []

    for i in range(32):   #mouse=4 fly=32
        sum = pred[numpred-1][0][i]
        for j in range(numpred-1):  # num pred
            if (pointoks[j][i][0] > thr):
                indexlist.append(j)
        if(len(indexlist)!=0):
            for k in indexlist:
                sum = sum + pred[k][0][i]
                leng = len(indexlist)+1
                # print(leng)
                avepoint[0][i] = sum / [leng,leng]
            indexlist = []
        else:
            avepoint[0][i] =  pred[numpred-1][0][i]
    return avepoint


pred = []
pointoks = []
pred.append(np.array([[[3,1],
                  [1,1],
                  [1,6],
                  [1,3]]])
)

pred.append(np.array([[[2,1],
                  [1,2],
                  [1,5],
                  [1,10]]])
)

pred.append(np.array([[[2,1],
                  [1,2],
                  [1,5],
                  [1,7]]])
)
gt = np.array([[[1,1],
                  [1,1],
                  [1,1],
                  [1,1]]])

# for i in range(3):
#     pointoks.append(calc_oks(pred[i], gt, 1))
#
# k2= updatepred(pointoks, pred)

# k=np.zeros((1,4,2))
# max =0
# maxlist = []
# indexlist = []
# t = 0
# for i in range(4):
#     for j in range(3): #num pred
#         if(pointoks[j][i][0] > max):
#             max = pointoks[j][i][0]
#             t = j
#     maxlist.append(max) #max=0 invalid point
#     indexlist.append(t)
#     max = 0
#     k[0][i] = pred[indexlist[i]][0][i]


# for i in range(4):
#     k[0][i] = pred[indexlist[i]][0][i]

# print(pred[1])
# print(a[1][0][0])
# print(maxlist)
# print(indexlist)
# print(pred[indexlist[0]][0][0])
# print(k[0][0])
# print(k)
# print(k2)
# Author: Sining Sun , Zhanheng Yang

import numpy as np
from utils import *
import scipy.cluster.vq as vq

num_gaussian = 5
num_iterations = 100
targets = ['Z', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'O']

class GMM:
    def __init__(self, D, K = 5):
        assert(D > 0)
        self.dim = D
        self.K = K
        #Kmeans Initial
        self.mu, self.sigma, self.pi = self.kmeans_initial()

    def kmeans_initial(self):
        mu = []
        sigma = []
        data = read_all_data('train/feats.scp')
        (centroids, labels) = vq.kmeans2(data, self.K, minit="points", iter=100)
        clusters = [[] for i in range(self.K)]
        for (l, d) in zip(labels, data):
            clusters[l].append(d)

        for cluster in clusters:
            mu.append(np.mean(cluster, axis=0))
            sigma.append(np.cov(cluster, rowvar=0))
        pi = np.array([len(c)*1.0 / len(data) for c in clusters])
        return mu, sigma, pi
    
    def gaussian(self, x, mu, sigma):
        """Calculate gaussion probability.
    
            :param x: The observed data, dim*1.
            :param mu: The mean vector of gaussian, dim*1
            :param sigma: The covariance matrix, dim*dim
            :return: the gaussion probability, scalor
        """
        D=x.shape[0]
        det_sigma = np.linalg.det(sigma)
        inv_sigma = np.linalg.inv(sigma + 0.0001)
        mahalanobis = np.dot(np.transpose(x-mu), inv_sigma)
        mahalanobis = np.dot(mahalanobis, (x-mu))
        const = 1/((2*np.pi)**(D/2))
        return const * (det_sigma)**(-0.5) * np.exp(-0.5 * mahalanobis)
    
    def calc_log_likelihood(self, X):
        """Calculate log likelihood of GMM

            param: X: A matrix including data samples, num_samples * D
            return: log likelihood of current model 
        """
        sum = 0
        log_llh = 0.0
        num_samples = X.shape[0]
        for n in range(0, num_samples):
            for k in range(0, self.K):
                x = np.transpose(X[n])      #第一帧的39维MFCC 39*1
                mu = np.transpose(self.mu[k]) #39*1
                sigma = self.sigma[k]
                pi = self.pi[k]
                sum = sum + pi * self.gaussian(x, mu, sigma)
            log_llh = log_llh + np.log(sum)
        # N = X.shape[0]
        # log_llh = np.sum([np.log(
        #     np.sum([self.pi[k] * self.gaussian(X[n], self.mu[k], self.sigma[k]) for k in range(self.K)])
        # ) for n in range(N)])
        return log_llh

    def em_estimator(self, X):
        """Update parameters of GMM

            param: X: A matrix including data samples, num_samples * D
            return: log likelihood of updated mode
        """
        N = X.shape[0]
        samgauss = [np.zeros(self.K) for i in range(N)]  # 构造样本和高斯函数的列表[N,(K,)]

        # E-step
        for n in range(N):
            postprob = [self.pi[k] * self.gaussian(X[n], self.mu[k], self.sigma[k]) for k in range(self.K)]
            postprob = np.array(postprob)
            postprob_sum = np.sum(postprob)
            samgauss[n] = postprob / postprob_sum
            # pdb.set_trace()

        # M-step
        for k in range(self.K):
            Nk = np.sum([samgauss[n][k] for n in range(N)])
            if Nk == 0:
                continue
            self.pi[k] = Nk / N
            self.mu[k] = (1.0 / Nk) * np.sum([samgauss[n][k] * X[n] for n in range(N)], axis=0)
            diffs = X - self.mu[k]
            self.sigma[k] = (1.0 / Nk) * np.sum(
                [samgauss[n][k] * diffs[n].reshape(self.dim, 1) * diffs[n] for n in range(N)],
                axis=0)
        log_llh = self.calc_log_likelihood(X)
        print(log_llh)
        return log_llh
        # log_llh = 0.0
        # indiv = np.zeros(5)
        # num_samples = X.shape[0]
        # mu_total = np.zeros((self.K, 39))
        # sigma_total = np.zeros((self.K, 39, 39))
        # new_pi = np.zeros(self.K)
        # new_mu = np.zeros((self.K, 39))
        # new_sigma = np.zeros((self.K, 39, 39))
        # weight = np.zeros(self.K)
        # for n in range(0,num_samples):
        #     x = X[n]
        #     for j in range(0, self.K):
        #         indiv[j] = self.pi[j] * self.gaussian(x, self.mu[j], self.sigma[j]) #每一帧对应计算5个高斯分量
        #     for k in range(0, self.K):
        #         weight[k] = weight[k] + indiv[k] / (np.sum(indiv))#后验概率,第k个高斯对产生第n帧的比重
        #         mu_total[k] = mu_total[k] + (indiv[k] /np.sum(indiv)) * x
        # new_pi = weight/num_samples
        # # print(new_pi)
        # for k in range(0, self.K):
        #     new_mu[k] = mu_total[k] / weight[k]
        #
        # for m in range(0, num_samples):
        #     x = X[m]
        #     for l in range(0, self.K):
        #         indiv[l] = self.pi[l] * self.gaussian(x, self.mu[l], self.sigma[l])  # 每一帧对应计算5个高斯分量
        #     for q in range(0, self.K):
        #         sigma_total[q] = sigma_total[q] + (indiv[q]/np.sum(indiv))*(x - new_mu[q])*(np.transpose(np.reshape(x-new_mu[q],(1,39))))
        # for i in range(0, self.K):
        #     new_sigma[i] = sigma_total[i] / weight[i]
        # # print(new_pi)
        # # print(new_mu)
        # # print(new_sigma[1])
        # self.mu = new_mu
        # self.sigma = new_sigma
        # self.pi = new_pi
        # log_llh = self.calc_log_likelihood(X)
        # print(log_llh)

def train(gmms, num_iterations = num_iterations):
    dict_utt2feat, dict_target2utt = read_feats_and_targets('train/feats.scp', 'train/text')
    
    for target in targets:
        feats = get_feats(target, dict_utt2feat, dict_target2utt)   #feats(samples_num,39)
        for i in range(num_iterations):
            log_llh = gmms[target].em_estimator(feats)
    return gmms

def test(gmms):
    correction_num = 0
    error_num = 0
    acc = 0.0
    dict_utt2feat, dict_target2utt = read_feats_and_targets('test/feats.scp', 'test/text')
    dict_utt2target = {}
    for target in targets:
        utts = dict_target2utt[target]
        for utt in utts:
            dict_utt2target[utt] = target
    for utt in dict_utt2feat.keys():
        feats = kaldi_io.read_mat(dict_utt2feat[utt])
        scores = []
        for target in targets:
            scores.append(gmms[target].calc_log_likelihood(feats))
        predict_target = targets[scores.index(max(scores))]
        if predict_target == dict_utt2target[utt]:
            correction_num += 1
        else:
            error_num += 1
    acc = correction_num * 1.0 / (correction_num + error_num)
    return acc


def main():
    gmms = {}
    for target in targets:
        gmms[target] = GMM(39, K=num_gaussian) #Initial model
    gmms = train(gmms)
    acc = test(gmms)
    print('Recognition accuracy: %f' % acc)
    fid = open('acc.txt', 'w')
    fid.write(str(acc))
    fid.close()

if __name__ == '__main__':
    main()

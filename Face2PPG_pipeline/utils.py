# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 10:19:15 2024

@author: ys
"""

import numpy as np
from scipy.spatial import ConvexHull
import scipy.linalg
from PIL import Image, ImageDraw
from copy import deepcopy
from threading import Lock
from numpy.linalg import inv, cholesky
from numba import njit, prange, float32
from scipy.signal import butter, filtfilt, welch

import cv2
from filterpy.kalman import UnscentedKalmanFilter as UKF
from filterpy.kalman import MerweScaledSigmaPoints


def Welch(bvps, fps, minHz=0.65, maxHz=4.0, nfft=2048):
    """
    This function computes Welch'method for spectral density estimation.

    Args:
        bvps(float32 numpy.ndarray): BVP signal as float32 Numpy.ndarray with shape [num_estimators, num_frames].
        fps (float): frames per seconds.
        minHz (float): frequency in Hz used to isolate a specific subband [minHz, maxHz] (esclusive).
        maxHz (float): frequency in Hz used to isolate a specific subband [minHz, maxHz] (esclusive).
        nfft (int): number of DFT points, specified as a positive integer.
    Returns:
        Sample frequencies as float32 numpy.ndarray, and Power spectral density or power spectrum as float32 numpy.ndarray.
    """
    _, n = bvps.shape
    if n < 256:
        seglength = n
        overlap = int(0.8*n)  # fixed overlapping
    else:
        seglength = 256
        overlap = 200
    # -- periodogram by Welch
    F, P = welch(bvps, nperseg=seglength, noverlap=overlap, fs=fps, nfft=nfft)
    F = F.astype(np.float32)
    P = P.astype(np.float32)
    # -- freq subband (0.65 Hz - 4.0 Hz)
    band = np.argwhere((F > minHz) & (F < maxHz)).flatten()
    Pfreqs = 60*F[band]
    Power = P[:, band]
    return Pfreqs, Power

class BPM:
    def __init__(self, fps=30, startTime=0, minHz=0.5, maxHz=4., verb=False):
        self.nFFT = 2048//1  # freq. resolution for STFTs
        self.fps = fps                       # sample rate
        self.startTime = startTime
        self.verb = verb
        self.minHz = minHz
        self.maxHz = maxHz


    def BVP_to_BPM(self, data):
        """
        Return the BPM signal as a float32 Numpy.ndarray with shape [num_estimators, ].

        This method uses the Welch's method to estimate the spectral density of the BVP signal,
        then it chooses as BPM the maximum Amplitude frequency.
        """
        if len(data.shape) == 1:
            data = data.reshape(1, -1)
        if data.shape[0] == 0:
            return np.float32(0.0)
        Pfreqs, Power = Welch(data, self.fps, self.minHz, self.maxHz, self.nFFT)
        # -- BPM estimate
        #Normalized Power에서 획득하는 SNR은, 일반 SNR과 비교하면 min(Power) 값이 penalty term 역할을 함.
        Pmax = np.argmax(Power, axis=1)  # power max
        SNR = float(Power[0][Pmax]/(np.sum(Power)-Power[0][Pmax])[0])
        Power = (Power-np.min(Power))/(np.max(Power)-np.min(Power))
        pSNR = float(Power[0][Pmax]/(np.sum(Power)-Power[0][Pmax])[0])
        return Pfreqs[Pmax.squeeze()], SNR, pSNR, Pfreqs, Power

def bpf(signal, low_band=0.5, high_band=4.0, fs=30, N=2):
    [b_pulse, a_pulse] = butter(N, [low_band / fs * 2, high_band / fs * 2], btype='bandpass')
    rst_signal = filtfilt(b_pulse, a_pulse, np.double(signal))
    return rst_signal

@njit(['float32[:,:](uint8[:,:,:], int32, int32)', ], parallel=True, fastmath=True, nogil=True)
def holistic_mean(im, RGB_LOW_TH, RGB_HIGH_TH):
    """
    This method computes the RGB-Mean Signal excluding 'im' pixels
    that are outside the RGB range [RGB_LOW_TH, RGB_HIGH_TH] (extremes are included).

    Args: 
        im (uint8 ndarray): ndarray with shape [rows, columns, rgb_channels].
        RGB_LOW_TH (numpy.int32): RGB low threshold value.
        RGB_HIGH_TH (numpy.int32): RGB high threshold value.
    
    Returns:
        RGB-Mean Signal as float32 ndarray with shape [1,3], where 1 is the single estimator,
        and 3 are r-mean, g-mean and b-mean.
    """
    mean = np.zeros((1, 3), dtype=np.float32)
    mean_r = np.float32(0.0)
    mean_g = np.float32(0.0)
    mean_b = np.float32(0.0)
    num_elems = np.float32(0.0)
    for x in prange(im.shape[0]):
        for y in prange(im.shape[1]):
            if not((im[x, y, 0] <= RGB_LOW_TH and im[x, y, 1] <= RGB_LOW_TH and im[x, y, 2] <= RGB_LOW_TH)
                    or (im[x, y, 0] >= RGB_HIGH_TH and im[x, y, 1] >= RGB_HIGH_TH and im[x, y, 2] >= RGB_HIGH_TH)):
                mean_r += im[x, y, 0]
                mean_g += im[x, y, 1]
                mean_b += im[x, y, 2]
                num_elems += 1.0
    if num_elems > 1.0:
        mean[0, 0] = mean_r / num_elems
        mean[0, 1] = mean_g / num_elems
        mean[0, 2] = mean_b / num_elems
    else:
        mean[0, 0] = mean_r
        mean[0, 1] = mean_g
        mean[0, 2] = mean_b 
    return mean

class bvps:
    
    def __init__(self):
        self.kargs = {}
        self.kargs['fps'] = 30
    
    def cpu_CHROM(self, signal):
        """
        CHROM method on CPU using Numpy.
    
        De Haan, G., & Jeanne, V. (2013). Robust pulse rate from chrominance-based rPPG. 
        IEEE Transactions on Biomedical Engineering, 60(10), 2878-2886.
        """
        X = signal
        Xcomp = 3*X[:, 0] - 2*X[:, 1]
        Ycomp = (1.5*X[:, 0])+X[:, 1]-(1.5*X[:, 2])
        sX = np.std(Xcomp, axis=1)
        sY = np.std(Ycomp, axis=1)
        alpha = (sX/sY).reshape(-1, 1)
        alpha = np.repeat(alpha, Xcomp.shape[1], 1)
        bvp = Xcomp - np.multiply(alpha, Ycomp)
        return bvp
    
    
    def cpu_LGI(self, signal):
        """
        LGI method on CPU using Numpy.
    
        Pilz, C. S., Zaunseder, S., Krajewski, J., & Blazek, V. (2018). Local group invariance for heart rate estimation from face videos in the wild. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition Workshops (pp. 1254-1262).
        """
        X = signal
        U, _, _ = np.linalg.svd(X)
        S = U[:, :, 0]
        S = np.expand_dims(S, 2)
        sst = np.matmul(S, np.swapaxes(S, 1, 2))
        p = np.tile(np.identity(3), (S.shape[0], 1, 1))
        P = p - sst
        Y = np.matmul(P, X)
        bvp = Y[:, 1, :]
        return bvp
    
    
    def cpu_POS(self, signal):
        """
        POS method on CPU using Numpy.
    
        The dictionary parameters are: {'fps':float}.
    
        Wang, W., den Brinker, A. C., Stuijk, S., & de Haan, G. (2016). Algorithmic principles of remote PPG. IEEE Transactions on Biomedical Engineering, 64(7), 1479-1491. 
        """
        # Run the pos algorithm on the RGB color signal c with sliding window length wlen
        # Recommended value for wlen is 32 for a 20 fps camera (1.6 s)
        eps = 10**-9
        X = signal
        e, c, f = X.shape            # e = #estimators, c = 3 rgb ch., f = #frames
        w = int(1.6 * self.kargs['fps'])   # window length
    
        # stack e times fixed mat P
        P = np.array([[0, 1, -1], [-2, 1, 1]])
        Q = np.stack([P for _ in range(e)], axis=0)
    
        # Initialize (1)
        H = np.zeros((e, f))
        for n in np.arange(w, f):
            # Start index of sliding window (4)
            m = n - w + 1
            # Temporal normalization (5)
            Cn = X[:, :, m:(n + 1)]
            M = 1.0 / (np.mean(Cn, axis=2)+eps)
            M = np.expand_dims(M, axis=2)  # shape [e, c, w]
            Cn = np.multiply(M, Cn)
    
            # Projection (6)
            S = np.dot(Q, Cn)
            S = S[0, :, :, :]
            S = np.swapaxes(S, 0, 1)    # remove 3-th dim
    
            # Tuning (7)
            S1 = S[:, 0, :]
            S2 = S[:, 1, :]
            alpha = np.std(S1, axis=1) / (eps + np.std(S2, axis=1))
            alpha = np.expand_dims(alpha, axis=1)
            Hn = np.add(S1, alpha * S2)
            Hnm = Hn - np.expand_dims(np.mean(Hn, axis=1), axis=1)
            # Overlap-adding (8)
            H[:, m:(n + 1)] = np.add(H[:, m:(n + 1)], Hnm)
    
        return H
    
    
    def cpu_OMIT(self, signal):
        """
        OMIT method on CPU using Numpy.
    
        Álvarez Casado, C., Bordallo López, M. (2022). Face2PPG: An unsupervised pipeline for blood volume pulse extraction from faces. arXiv (eprint 2202.04101).
        """
    
        bvp = []
        for i in range(signal.shape[0]):
            X = signal[i]
            Q, R = np.linalg.qr(X)
            S = Q[:, 0].reshape(1, -1)
            P = np.identity(3) - np.matmul(S.T, S)
            Y = np.dot(P, X)
            bvp.append(Y[1, :])
        bvp = np.array(bvp)
        return bvp
    
    def cpu_SSR(self, raw_signal):
        """
        SSR method on CPU using Numpy.
    
        'raw_signal' is a float32 ndarray with shape [num_frames, rows, columns, rgb_channels]; it can be obtained by
        using the :py:class:‵pyVHR.extraction.sig_processing.SignalProcessing‵ class ('extract_raw_holistic' method).
    
        The dictionary parameters are: {'fps':float}.
    
        Wang, W., Stuijk, S., & De Haan, G. (2015). A novel algorithm for remote photoplethysmography: Spatial subspace rotation. IEEE transactions on biomedical engineering, 63(9), 1974-1984.
        """
        # utils functions #
        def __build_p(τ, k, l, U, Λ):
            """
            builds P
            Parameters
            ----------
            k: int
                The frame index
            l: int
                The temporal stride to use
            U: numpy.ndarray
                The eigenvectors of the c matrix (for all frames up to counter).
            Λ: numpy.ndarray
                The eigenvalues of the c matrix (for all frames up to counter).
            Returns
            -------
            p: numpy.ndarray
                The p signal to add to the pulse.
            """
            # SR'
            SR = np.zeros((3, l), np.float32)  # dim: 3xl
            z = 0
    
            for t in range(τ, k, 1):  # 6, 7
                a = Λ[0, t]
                b = Λ[1, τ]
                c = Λ[2, τ]
                d = U[:, 0, t].T
                e = U[:, 1, τ]
                f = U[:, 2, τ]
                g = U[:, 1, τ].T
                h = U[:, 2, τ].T
                x1 = a / b
                x2 = a / c
                x3 = np.outer(e, g)
                x4 = np.dot(d, x3)
                x5 = np.outer(f, h)
                x6 = np.dot(d, x5)
                x7 = np.sqrt(x1)
                x8 = np.sqrt(x2)
                x9 = x7 * x4
                x10 = x8 * x6
                x11 = x9 + x10
                SR[:, z] = x11  # 8 | dim: 3
                z += 1
    
            # build p and add it to the final pulse signal
            s0 = SR[0, :]  # dim: l
            s1 = SR[1, :]  # dim: l
            p = s0 - ((np.std(s0) / np.std(s1)) * s1)  # 10 | dim: l
            p = p - np.mean(p)  # 11
            return p  # dim: l
            
        def __build_correlation_matrix(V):
            # V dim: (W×H)x3
            #V = np.unique(V, axis=0)
            V_T = V.T  # dim: 3x(W×H)
            N = V.shape[0]
            # build the correlation matrix
            C = np.dot(V_T, V)  # dim: 3x3
            C = C / N
    
            return C
    
        def __eigs(C):
            """
            get eigenvalues and eigenvectors, sort them.
            Parameters
            ----------
            C: numpy.ndarray
                The RGB values of skin-colored pixels.
            Returns
            -------
            Λ: numpy.ndarray
                The eigenvalues of the correlation matrix
            U: numpy.ndarray
                The (sorted) eigenvectors of the correlation matrix
            """
            # get eigenvectors and sort them according to eigenvalues (largest first)
            L, U = np.linalg.eig(C)  # dim Λ: 3 | dim U: 3x3
            idx = L.argsort()  # dim: 3x1
            idx = idx[::-1]  # dim: 1x3
            L_ = L[idx]  # dim: 3
            U_ = U[:, idx]  # dim: 3x3
    
            return L_, U_
        # ----------------------------------- #
    
        fps = int(self.kargs['fps'])
    
        raw_sig = raw_signal
        K = len(raw_sig)
        l = int(fps)
    
        P = np.zeros(K)  # 1 | dim: K
        # store the eigenvalues Λ and the eigenvectors U at each frame
        L = np.zeros((3, K), dtype=np.float32)  # dim: 3xK
        U = np.zeros((3, 3, K), dtype=np.float32)  # dim: 3x3xK
    
        for k in range(K):
            n_roi = len(raw_sig[k])
            VV = []
            V = raw_sig[k].astype(np.float32)
            idx = V!=0
            idx2 = np.logical_and(np.logical_and(idx[:,:,0], idx[:,:,1]), idx[:,:,2])
            V_skin_only = V[idx2]
            VV.append(V_skin_only)
            
            VV = np.vstack(VV)
    
            C = __build_correlation_matrix(VV)  #dim: 3x3
    
            # get: eigenvalues Λ, eigenvectors U
            L[:,k], U[:,:,k] = __eigs(C)  # dim Λ: 3 | dim U: 3x3
    
            # build p and add it to the pulse signal P
            if k >= l:  # 5
                tau = k - l  # 5
                p = __build_p(tau, k, l, U, L)  # 6, 7, 8, 9, 10, 11 | dim: l
                P[tau:k] += p  # 11
    
            if np.isnan(np.sum(P)):
                print('NAN')
                print(raw_sig[k])
                
        bvp = P
        bvp = np.expand_dims(bvp,axis=0)
        return bvp
    
    def bvp(self, algorithm, signal):
        if algorithm == "lgi":
            return self.cpu_LGI(signal)
        elif algorithm == "chrom":
            return self.cpu_CHROM(signal)
        elif algorithm == "pos":
            return self.cpu_POS(signal)
        elif algorithm == "omit":
            return self.cpu_OMIT(signal)
        elif algorithm == "ssr":
            return self.cpu_SSR(signal)


class PointUKF:
    def __init__(self, dt, transition_func="dynamic"):
        self.dt = dt
        self.transition = transition_func
        self.points = MerweScaledSigmaPoints(n=4, alpha=0.1, beta=2.0, kappa=0.0)
        
        if self.transition == "stationary":    
            self.ukf = UKF(dim_x=4, dim_z=2, fx=self.fx, hx=self.hx, dt=dt, points=self.points)
            
        elif self.transition == "dynamic":
            self.ukf = UKF(dim_x=, dim_z=, fx=self.fx_arima, hx=self.hx, dt=dt, points=self.points)
            self.x_arq = []  # (13, 1, 1)
            self.y_arq = []  # (7, 1, 0)
            self.x_maq = []
            self.x_qmax = 14  # 13 + 1(difference)
            self.y_qmax = 8  # 7 + 1(differernce)
            
        elif self.transition == "sinusoidal":
            self.ukf = UKF(dim_x=4, dim_z=2, fx=self.fx_sinusoidal, hx=self.hx, dt=dt, points=self.points)
        else:
            raise ValueError("wrong transition function type")
    
        # initial state vector [px, py, vx, vy]
        self.ukf.x = np.array([0, 0, 0, 0])
        
        # initial covariance matrix
        self.ukf.P = np.eye(4)
        
        # process noise covariance matrix
        self.ukf.Q = np.eye(4) * 0.01
        
        #  observation noise covariance matrix
        self.ukf.R = np.eye(2) * 0.005

    def fx(self, x, dt):
        F = np.array([[1, 0, dt, 0],
                      [0, 1, 0, dt],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1]])
        return np.dot(F, x)
    
    def fx_arima(self, x, dt):
        F = np.array([[]])
        
        return np.dot(F, x)
    
    def fx_sinusoidal(self, x, dt):
        F = np.array([[]])
        return np.dot(F, x)

    def hx(self, x):
        return x[:2]

    def predict(self):
        
        if self.transition == "dynamic":
            if len(self.x_arq) >= self.x_qmax:
                self.ukf.predict()
        else:
            self.ukf.predict()

    def update(self, z):
        if self.transition == "dynamic":
            self.x_arq.append(z[0])
            self.y_arq.append(z[1])
            
            if len(self.x_arq) == self.x_qmax:
                self.y_arq = self.y_arq[:-self.y_qmax]
                self.ukf.update(z)
                
                self.x_arq.pop(0)
            else:
                self.ukf.update(z)
                
        else:
            self.ukf.update(z)

    def get_state(self):
        return self.ukf.x

    def get_covariance(self):
        return self.ukf.P


class MyUKF:
    def __init__(self, transition_func, initial_state, std_acc, std_meas):
        """
            ARIMA state function:
                1st estimation result:
                    X: (13, 1, 1), aic: 10136.434 (min aic)
                    Y: (18, 0, 5), aic: 13054.519 (min aic)
                    X model:
                    Y model: 
                     -> model is not converged to one set of values.
                        to use reffiting process
                    
                    X: (), aic: (The mode model) -> too few samples, no model
                    Y: (), aic: (The mode model) -> too few samples, no model
                    
                2nd estimation result:
                    X: (), aic: (min_aic)
                    y: (), aic: (min_aic)
                    
                    X: (), aic: (The mode model)
                    Y: (), aic: (The mode model)
            
            Sinusodial state function:
                
            
            futher research
                1. sinusoidal function for the state function of UKF (stft for frequency)
        """
        """
        transition_func: Selecing transition function. This parameter is one of
                         the words among 'stationary', 'sinusoidal', 'dynamic'
        """
        """
        UKF input : [x, y, x_acc, y_acc].T
        
        """
        
        self.dt = 1/30
        self.transition_func = transition_func
        
        if transition_func == 'dynamic':
            self.A = self.__dynamic()
            self.queue_x = []
            self.queue_y = []
            self.x_qlen = 
            self.y_qlen = 
        elif transition_func == 'stationary':
            self.A = self.__stationary()
        elif transition_func == 'sinusoidal':
            self.A = self.__sinusoidal()
        else:
            raise ValueError("transition function is one of the ['stionary', 'sinusoidal', 'dynamic']")
        
        self.input_dimension = initial_state.shape[0]
        
        # state vector initialization
        self.x = initial_state
        
        # state covariance initialization
        self.P = np.eye(4*self.input_dimension)
        
        # process noise covariance initizalization
        Q_block = np.array([[0.5 * self.dt**2, 0, 0.5 * self.dt, 0],
                            [0, 0.5 * self.dt**2, 0, 0.5 * self.dt],
                            [0.5 * self.dt, 0, self.dt, 0],
                            [0, 0.5 * self.dt, 0, self.dt]]) * std_acc**2
        self.Q = np.kron(np.eye(self.input_dimension), Q_block)
        
        # observation noise covariance initialization
        self.R = np.eye(2*self.input_dimension) * std_meas ** 2
        
        # sigma point parameter setting
        self.kappa = 3 - 4 * self.input_dimension
        self.n = 4 * self.input_dimension
        
    def __stationary(self):
        A = np.array([[1, 0, self.dt, 0],
                      [0, 1, 0, self.dt],
                      [0, 0, 1, 0], 
                      [0, 0, 0, 1]])
        return A
    
    def __sinusoidal(self):
        A = np.array([[]])
        return A
    
    def __dynamic(self):
        A = np.zrray([[]])
        
        return A
    
    def generate_sigma_points(self):
        sigma_points = np.zeros((2 * self.n + 1, self.n))
        sigma_points[0] = self.x

        sqrt_P = np.linalg.cholesky((self.n + self.kappa) * self.P)

        for i in range(self.n):
            sigma_points[i + 1] = self.x + sqrt_P[:, i]
            sigma_points[self.n + i + 1] = self.x - sqrt_P[:, i]
        return sigma_points
    
    def predict_sigma_points(self, sigma_points):
        sigma_points_pred = np.zeros_like(sigma_points)
        for i, sp in enumerate(sigma_points):
            sigma_points_pred[i] = self.state_transition(sp)
        return sigma_points_pred

    def state_transition(self, x):
        F = np.kron(np.eye(self.input_dimension), self.A)
        return F @ x

    def predict(self):
        if self.transition_func == 'dynamic':
            pass
        else:
            sigma_points = self.generate_sigma_points()
            sigma_points_pred = self.predict_sigma_points(sigma_points)
    
            self.x = np.mean(sigma_points_pred, axis=0)
            self.P = np.cov(sigma_points_pred.T) + self.Q

    def update(self, z):
        if self.transition_func == 'dynamic':
            if len(self.queue_x) > 
        else:
            sigma_points = self.generate_sigma_points()
            sigma_points_pred = self.predict_sigma_points(sigma_points)
    
            z_pred = np.mean(sigma_points_pred[:, :2 * self.input_dimension], axis=0)
            S = np.cov(sigma_points_pred[:, :2 * self.input_dimension].T) + self.R
            cross_cov = np.cov(sigma_points_pred.T)[:2 * self.input_dimension, 2 * self.input_dimension:]
    
            K = cross_cov @ np.linalg.inv(S)
            self.x = self.x + K @ (z - z_pred)
            self.P = self.P - K @ S @ K.T
        
        
    def get_state(self):
        return self.x
    
    def get_covariance(self):
        return self.P


class SkinExtractionConvexHull:
    
    """
        This class performs skin extraction on CPU/GPU using a Convex Hull segmentation obtained from facial landmarks.
    """
    def __init__(self, device='CPU'):
        """
        Args:
            device (str): This class can execute code on 'CPU' or 'GPU'.
        """
        self.device = device
    
    def extract_skin(self,image, ldmks):
        """
        This method extract the skin from an image using Convex Hull segmentation.

        Args:
            image (uint8 ndarray): ndarray with shape [rows, columns, rgb_channels].
            ldmks (float32 ndarray): landmarks used to create the Convex Hull; ldmks is a ndarray with shape [num_landmarks, xy_coordinates].

        Returns:
            Cropped skin-image and non-cropped skin-image; both are uint8 ndarray with shape [rows, columns, rgb_channels].
        """
        aviable_ldmks = ldmks[ldmks[:,0] >= 0][:,:2]        
        # face_mask convex hull 
        hull = ConvexHull(aviable_ldmks)
        verts = [(aviable_ldmks[v,0], aviable_ldmks[v,1]) for v in hull.vertices]
        img = Image.new('L', image.shape[:2], 0)
        ImageDraw.Draw(img).polygon(verts, outline=1, fill=1)
        mask = np.array(img)
        mask = np.expand_dims(mask,axis=0).T

        # left eye convex hull
        left_eye_ldmks = ldmks[MagicLandmarks.left_eye]
        aviable_ldmks = left_eye_ldmks[left_eye_ldmks[:,0] >= 0][:,:2]
        if len(aviable_ldmks) > 3:
            hull = ConvexHull(aviable_ldmks)
            verts = [(aviable_ldmks[v,0], aviable_ldmks[v,1]) for v in hull.vertices]
            img = Image.new('L', image.shape[:2], 0)
            ImageDraw.Draw(img).polygon(verts, outline=1, fill=1)
            left_eye_mask = np.array(img)
            left_eye_mask = np.expand_dims(left_eye_mask,axis=0).T
        else:
            left_eye_mask = np.ones((image.shape[0], image.shape[1],1),dtype=np.uint8)

        # right eye convex hull
        right_eye_ldmks = ldmks[MagicLandmarks.right_eye]
        aviable_ldmks = right_eye_ldmks[right_eye_ldmks[:,0] >= 0][:,:2]
        if len(aviable_ldmks) > 3:
            hull = ConvexHull(aviable_ldmks)
            verts = [(aviable_ldmks[v,0], aviable_ldmks[v,1]) for v in hull.vertices]
            img = Image.new('L', image.shape[:2], 0)
            ImageDraw.Draw(img).polygon(verts, outline=1, fill=1)
            right_eye_mask = np.array(img)
            right_eye_mask = np.expand_dims(right_eye_mask,axis=0).T
        else:
            right_eye_mask = np.ones((image.shape[0], image.shape[1],1),dtype=np.uint8)

        # mounth convex hull
        mounth_ldmks = ldmks[MagicLandmarks.mounth]
        aviable_ldmks = mounth_ldmks[mounth_ldmks[:,0] >= 0][:,:2]
        if len(aviable_ldmks) > 3:
            hull = ConvexHull(aviable_ldmks)
            verts = [(aviable_ldmks[v,0], aviable_ldmks[v,1]) for v in hull.vertices]
            img = Image.new('L', image.shape[:2], 0)
            ImageDraw.Draw(img).polygon(verts, outline=1, fill=1)
            mounth_mask = np.array(img)
            mounth_mask = np.expand_dims(mounth_mask,axis=0).T
        else:
            mounth_mask = np.ones((image.shape[0], image.shape[1],1),dtype=np.uint8)

        # apply masks and crop 
        if self.device == 'GPU':
            image = cupy.asarray(image)
            mask = cupy.asarray(mask)
            left_eye_mask = cupy.asarray(left_eye_mask)
            right_eye_mask = cupy.asarray(right_eye_mask)
            mounth_mask = cupy.asarray(mounth_mask)
        skin_image = image * mask * (1-left_eye_mask) * (1-right_eye_mask) * (1-mounth_mask)

        if self.device == 'GPU':
            rmin, rmax, cmin, cmax = bbox2_GPU(skin_image)
        else:
            rmin, rmax, cmin, cmax = bbox2_CPU(skin_image)

        cropped_skin_im = skin_image
        if rmin >= 0 and rmax >= 0 and cmin >= 0 and cmax >= 0 and rmax-rmin >= 0 and cmax-cmin >= 0:
            cropped_skin_im = skin_image[int(rmin):int(rmax), int(cmin):int(cmax)]

        if self.device == 'GPU':
            cropped_skin_im = cupy.asnumpy(cropped_skin_im)
            skin_image = cupy.asnumpy(skin_image)

        return cropped_skin_im, skin_image
        
    
class SkinExtractionConvexHull_Polygon:
    """
        This class performs skin extraction on CPU/GPU using a Convex Hull segmentation obtained from facial landmarks.
    """
    def __init__(self,device='CPU'):
        """
        Args:
            device (str): This class can execute code on 'CPU' or 'GPU'.
        """
        self.device = device
    
    def extract_polygon_cpu(self, image, ldmks):
        
        return 0
    
    def extract_polygon_gpu(self, image, ldmks):
        
        aviable_ldmks = ldmks[ldmks[:,0] >= 0][:,:2]        
        # face_mask convex hull 
        hull = ConvexHull(aviable_ldmks)
        verts = [(aviable_ldmks[v,0], aviable_ldmks[v,1]) for v in hull.vertices]
        img = Image.new('L', image.shape[:2], 0)
        ImageDraw.Draw(img).polygon(verts, outline=1, fill=1)
        mask = np.array(img)
        mask = np.expand_dims(mask,axis=0).T
        
        return output
    
    def extract_polygon(self, image, ldmks):
        """
        This method extract the polygon skin image from an cropped skin image using Convex Hull segmentation.

        Args:
            image (uint8 ndarray): ndarray with shape [rows, columns, rgb_channels].
            ldmks (float32 ndarray): landmarks used to create the Convex Hull; ldmks is a ndarray with shape [num_landmarks, xy_coordinates].

        Returns:
            Cropped polygon skin-image set;
                output (dictionary)
                    1 : uint8 ndarray with shape [rows, columns, rgb_channels]
                    2 : uint8 ndarray with shape [rows, columns, rgb_channels]
                    ...
                    468: uint8 ndarray with shape [rows, columns, rgb_channels]
            
            The length of dictionary output can be changed by the seletected
            input polygon list
        """
        
        
        if self.device == "GPU":
            output = extract_polygon_gpu(image, ldmks)
        
        else:
            output = extract_polygon_cpu(image, ldmks)
        
        return output
    
    
def cpu_LGI(signal):
    """
    LGI method on CPU using Numpy.

    Pilz, C. S., Zaunseder, S., Krajewski, J., & Blazek, V. (2018). Local group invariance for heart rate estimation from face videos in the wild. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition Workshops (pp. 1254-1262).
    """
    X = signal
    U, _, _ = np.linalg.svd(X)
    S = U[:, :, 0]
    S = np.expand_dims(S, 2)
    sst = np.matmul(S, np.swapaxes(S, 1, 2))
    p = np.tile(np.identity(3), (S.shape[0], 1, 1))
    P = p - sst
    Y = np.matmul(P, X)
    bvp = Y[:, 1, :]
    return bvp
        

def bbox2_CPU(img):
    """
    Args:
        img (ndarray): ndarray with shape [rows, columns, rgb_channels].

    Returns: 
        Four cropping coordinates (row, row, column, column) for removing black borders (RGB [O,O,O]) from img.
    """
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    nzrows = np.nonzero(rows)
    nzcols = np.nonzero(cols)
    if nzrows[0].size == 0 or nzcols[0].size == 0:
        return -1, -1, -1, -1
    rmin, rmax = np.nonzero(rows)[0][[0, -1]]
    cmin, cmax = np.nonzero(cols)[0][[0, -1]]
    return rmin, rmax, cmin, cmax


def bbox2_GPU(img):
    """
    Args:
        img (cupy.ndarray): cupy.ndarray with shape [rows, columns, rgb_channels].

    Returns: 
        Four cropping coordinates (row, row, column, column) for removing black borders (RGB [O,O,O]) from img. 
        The returned variables are on GPU.
    """
    rows = cupy.any(img, axis=1)
    cols = cupy.any(img, axis=0)
    nzrows = cupy.nonzero(rows)
    nzcols = cupy.nonzero(cols)
    if nzrows[0].size == 0 or nzcols[0].size == 0:
        return -1, -1, -1, -1
    rmin, rmax = cupy.nonzero(rows)[0][[0, -1]]
    cmin, cmax = cupy.nonzero(cols)[0][[0, -1]]
    return rmin, rmax, cmin, cmax


class segment_estimation:
    def __init__(self, kfd_window, kfd_overlap, ed_window, ed_overlap, snr_parameter,
                 metric_selection, device='CPU'):
        
        """
            kfd_window(int): KFD time window (seconds) as integer
            kfd_overlap(int): KFD overlap time window (seconds) as integer
            ed_window(int): Euclidean distance time window (seconds) as integer
            ed_overlap(int): Euclidean distance overlap time window (seconds) as integer
            snr_parameter(list): [min frequency, max frequency, frequency_border]
            metric_selection(list): boolean list for whether to use the metric or not
                for example, [True, False, ...] => using kfd, ignore euclidean distance ... and so on
            device(str): CPU or GPU as string input
        """
        
        self.kfd_window = kdf_window
        self.kfd_overlap = kfd_overlap
        self.ed_windnow = ed_window
        self.ed_overlap = ed_overlap
        self.snr_min = snr_parameter[0]
        self.snr_max = snr_parameter[1]
        self.snr_border = snr_parameter[2]
    
    
    def bpf(self):
        return bpf_signal
    
    
    def __kfd(self):
        return kfd_output
    
    
    def kfd_run(self):
        return kfd
    
    
    def __ed(self):
        return ed_output
    
    
    def ed_run(self):
        return ed
    
    
    def __snr(self):
        return snr_output
    
    
    def snr_run(self):
        return snr
    
    
    def __bp(self):
        return bp_output
    
    
    def back_propagation(self):
        return bp
    
    def est(self):
        
        """
            The output of est is a list of number of segment. The 1 menas that
            the signal on corresponding segment is appropriate for rPPG.
        """
        
        
        return output


class faceldmk_utils():
    left_eye = [157,144, 145, 22, 23, 25, 154, 31, 160, 33, 46, 52, 53, 55, 56, 189, 190, 63, 65, 66, 70, 221, 222, 223, 225, 226, 228, 229, 230, 231, 232, 105, 233, 107, 243, 124]
    right_eye = [384, 385, 386, 259, 388, 261, 265, 398, 276, 282, 283, 285, 413, 293, 296, 300, 441, 442, 445, 446, 449, 451, 334, 463, 336, 464, 467, 339, 341, 342, 353, 381, 373, 249, 253, 255]
    mounth = [391, 393, 11, 269, 270, 271, 287, 164, 165, 37, 167, 40, 43, 181, 313, 314, 186, 57, 315, 61, 321, 73, 76, 335, 83, 85, 90, 106]
    
    face_trimesh = {
        0: [103, 67, 104], 1: [67, 104, 69], 2: [103, 104, 68], 3: [103, 54, 68], 4: [54, 68, 71], 5: [54, 71, 21],
        6: [21, 71, 139], 7: [21, 139, 162], 8: [162, 34, 127], 9: [162, 139, 34], 10: [127, 34, 227],
        
        11: [127, 227, 234], 12: [234, 227, 137], 13: [234, 137, 93], 14: [137, 177, 93], 15: [93, 177, 132],
        16: [177, 215, 56], 17: [132, 177, 56], 18: [215, 172, 56], 19: [215, 138, 172], 20: [172, 138, 136],
        
        21: [136, 138, 135], 22: [136, 135, 150], 23: [150, 135, 169], 24: [150, 169, 149], 25: [149, 169, 170],
        26: [149, 170, 140], 27: [149, 140, 176], 28: [140, 176, 171], 29: [176, 171, 148], 30: [171, 175, 152],
        
        31: [171, 152, 148], 32: [175, 396, 152], 33: [396, 152, 377], 34: [396, 400, 369], 35: [396, 377, 400],
        36: [369, 395, 378], 37: [369, 378, 400], 38: [395, 394, 378], 39: [378, 394, 379], 40: [394, 364, 379],
        
        41: [379, 364, 365], 42: [364, 367, 365], 43: [367, 365, 397], 44: [367, 435, 397], 45: [397, 435, 288],
        46: [401, 435, 288], 47: [401, 361, 288], 48: [366, 401, 323], 49: [323, 401, 361], 50: [447, 366, 454],
        
        51: [366, 454, 323], 52: [264, 356, 447], 53: [447, 356, 454], 54: [368, 389, 264], 55: [264, 389, 356],
        56: [301, 368, 251], 57: [251, 368, 389], 58: [284, 301, 251], 59: [284, 301, 298], 60: [332, 284, 298],
        
        61: [332, 298, 333], 62: [332, 333, 297], 63: [299, 297, 333], 64: [338, 297, 299], 65: [338, 337, 299],
        66: [10, 338, 337], 67: [10, 151, 337], 68: [109, 10, 108], 69: [108, 10, 151], 70: [67, 69, 109], 

        71: [108, 151, 107], 72: [151, 107, 9], 73: [108, 69, 66], 74: [108, 66, 107], 75: [104, 69, 105],
        76: [69, 105, 66], 77: [104, 63, 105], 78: [68, 104, 63], 79: [68, 71, 70], 80: [68, 70, 63], 

        81: [71, 139, 156], 82: [71, 156, 70], 83: [139, 156, 143], 84: [139, 143, 34], 85: [34, 143, 116],
        86: [34, 116, 227], 87: [227, 116, 123], 88: [227, 123, 137], 89: [137, 123, 147], 90: [137, 147, 177], 

        91: [177, 147, 213], 92: [177, 213, 215], 93: [213, 215, 192], 94: [215, 138, 192], 95: [192, 138, 135],
        96: [192, 135, 214], 97: [214, 135, 210], 98: [135, 169, 210], 99: [210, 169, 170], 100: [210, 170, 211], 

        101: [211, 170, 32], 102: [170, 32, 140], 103: [32, 140, 208], 104: [140, 208, 171], 105: [208, 199, 175],
        106: [208, 171, 175], 107: [199, 428, 175], 108: [175, 396, 428], 109: [428, 369, 396], 110: [428, 369, 262], 

        111: [262, 395, 369], 112: [262, 395, 431], 113: [431, 430, 395], 114: [395, 430, 394], 115: [430, 364, 394],
        116: [430, 364, 434], 117: [434, 416, 364], 118: [364, 416, 367], 119: [416, 433, 435], 120: [416, 367, 435], 

        121: [376, 401, 433], 122: [433, 435, 401], 123: [376, 401, 366], 124: [376, 366, 352], 125: [352, 366, 447],
        126: [352, 447, 345], 127: [345, 447, 264], 128: [264, 372, 345], 129: [383, 368, 372], 130: [372, 368, 264], 

        131: [300, 383, 301], 132: [383, 301, 368], 133: [293, 300, 298], 134: [298, 300, 301], 135: [333, 298, 293],
        136: [333, 293, 334], 137: [299, 333, 334], 138: [299, 334, 296], 139: [337, 299, 296], 140: [337, 296, 336], 

        141: [151, 337, 336], 142: [151, 336, 9], 143: [107, 9, 55], 144: [55, 9, 8], 145: [66, 107, 65],
        146: [65, 107, 55], 147: [105, 66, 52], 148: [52, 66, 65], 149: [63, 105, 53], 150: [53, 105, 52], 

        151: [70, 63, 46], 152: [46, 63, 53], 153: [156, 70, 124], 154: [124, 70, 46], 155: [156, 143, 35],
        156: [156, 35, 124], 157: [143, 111, 116], 158: [143, 35, 111], 159: [111, 117, 116], 160: [116, 117, 123], 

        161: [117, 123, 50], 162: [117, 118, 50], 163: [123, 50, 187], 164: [123, 147, 187], 165: [147, 187, 192],
        166: [147, 192, 213], 167: [187, 207, 192], 168: [192, 207, 214], 169: [207, 216, 214], 170: [214, 212, 216], 

        171: [214, 212, 210], 172: [212, 210, 202], 173: [202, 210, 211], 174: [211, 202, 204], 175: [211, 204, 194],
        176: [211, 32, 194], 177: [32, 194, 201], 178: [32, 201, 208], 179: [201, 200, 208], 180: [208, 200, 199], 

        181: [200, 199, 428], 182: [200, 428, 421], 183: [421, 428, 262], 184: [421, 418, 262], 185: [418, 262, 431],
        186: [418, 431, 424], 187: [424, 422, 431], 188: [431, 422, 430], 189: [422, 432, 430], 190: [432, 430, 434], 

        191: [436, 432, 434], 192: [434, 427, 436], 193: [427, 434, 416], 194: [427, 411, 416], 195: [411, 376, 416],
        196: [416, 376, 433], 197: [280, 411, 352], 198: [352, 411, 376], 199: [346, 280, 352], 200: [346, 352, 345], 

        201: [346, 345, 340], 202: [340, 345, 372], 203: [340, 265, 372], 204: [265, 372, 383], 205: [353, 383, 265],
        206: [353, 300, 383], 207: [300, 353, 276], 208: [293, 300, 276], 209: [293, 276, 283], 210: [334, 293, 283], 

        211: [334, 282, 283], 212: [296, 334, 282], 213: [296, 282, 295], 214: [336, 296, 295], 215: [336, 295, 285],
        216: [9, 336, 285], 217: [9, 285, 8], 218: [55, 8, 193], 219: [193, 8, 168], 220: [55, 221, 193], 

        221: [55, 222, 221], 222: [65, 55, 222], 223: [65, 52, 222], 224: [52, 222, 223], 225: [53, 52, 224],
        226: [224, 52, 223], 227: [53, 46, 225], 228: [53, 225, 224], 229: [46, 113, 225], 230: [46, 124, 113], 

        231: [124, 35, 226], 232: [124, 226, 113], 233: [35, 226, 31], 234: [35, 31, 111], 235: [111, 31, 117],
        236: [31, 117, 228], 237: [117, 228, 118], 238: [228, 118, 229], 239: [347, 346, 280], 240: [8, 168, 417], 

        241: [8, 417, 285], 242: [285, 417, 441], 243: [285, 441, 442], 244: [285, 295, 442], 245: [295, 282, 442],
        246: [442, 282, 443], 247: [282, 443, 444], 248: [444, 282, 283], 249: [444, 283, 445], 250: [445, 283, 276], 

        251: [445, 342, 276], 252: [342, 276, 353], 253: [342, 353, 446], 254: [446, 353, 265], 255: [446, 261, 265],
        256: [261, 265, 340], 257: [448, 261, 436], 258: [261, 346, 340], 259: [448, 346, 347], 260: [449, 448, 347], 

        261: [110, 24, 229], 262: [110, 228, 229], 263: [25, 110, 228], 264: [25, 31, 228], 265: [226, 25, 31],
        266: [226, 130, 25], 267: [113, 226, 130], 268: [113, 247, 130], 269: [113, 247, 225], 270: [225, 247, 30], 

        271: [225, 30, 224], 272: [224, 30, 29], 273: [224, 29, 223], 274: [29, 223, 27], 275: [223, 27, 28],
        276: [223, 222, 28], 277: [222, 28, 56], 278: [222, 56, 221], 279: [56, 221, 189], 280: [221, 189, 193], 

        281: [254, 449, 339], 282: [339, 449, 448], 283: [339, 255, 448], 284: [448, 255, 261], 285: [255, 446, 261],
        286: [359, 255, 446], 287: [467, 342, 359], 288: [359, 342, 446], 289: [445, 342, 467], 290: [260, 445, 467], 

        291: [260, 445, 444], 292: [444, 260, 259], 293: [444, 443, 259], 294: [443, 259, 257], 295: [443, 258, 257],
        296: [442, 443, 258], 297: [442, 441, 286], 298: [442, 286, 258], 299: [441, 413, 286], 300: [441, 413, 417], 

        301: [286, 414, 413], 302: [414, 413, 464], 303: [417, 413, 465], 304: [413, 465, 464], 305: [417, 351, 465],
        306: [168, 417, 351], 307: [168, 351, 6], 308: [193, 168, 122], 309: [122, 168, 6], 310: [193, 122, 245], 

        311: [193, 245, 189], 312: [189, 244, 245], 313: [189, 190, 244], 314: [56, 190, 189], 315: [254, 449, 450],
        316: [254, 450, 253], 317: [253, 450, 451], 318: [253, 252, 451], 319: [252, 451, 452], 320: [252, 452, 256], 

        321: [256, 341, 452], 322: [341, 452, 453], 323: [341, 453, 463], 324: [463, 453, 464], 325: [414, 464, 463],
        326: [190, 243, 255], 327: [243, 244, 233], 328: [243, 112, 233], 329: [112, 26, 232], 330: [112, 232, 233], 

        331: [26, 22, 232], 332: [22, 232, 231], 333: [23, 22, 231], 334: [23, 231, 230], 335: [24, 23, 230],
        336: [24, 230, 229], 337: [244, 233, 128], 338: [244, 128, 245], 339: [232, 233, 128], 340: [232, 128, 121], 

        341: [232, 121, 231], 342: [231, 121, 120], 343: [231, 120, 230], 344: [230, 120, 119], 345: [229, 230, 119],
        346: [229, 119, 118], 347: [464, 465, 357], 348: [464, 453, 357], 349: [453, 452, 357], 350: [357, 350, 452], 

        351: [452, 451, 350], 352: [350, 451, 349], 353: [451, 450, 349], 354: [349, 450, 348], 355: [450, 449, 348],
        356: [449, 348, 347], 357: [351, 465, 412], 358: [465, 412, 343], 359: [343, 465, 357], 360: [357, 350, 343], 

        361: [343, 350, 277], 362: [350, 277, 349], 363: [277, 349, 329], 364: [349, 348, 329], 365: [329, 348, 330],
        366: [348, 347, 330], 367: [330, 347, 280], 368: [245, 122, 188], 369: [245, 128, 114], 370: [114, 245, 188], 

        371: [128, 121, 114], 372: [121, 114, 47], 373: [121, 120, 47], 374: [47, 100, 120], 375: [120, 100, 119],
        376: [119, 100, 101], 377: [119, 118, 101], 378: [118, 101, 50], 379: [50, 187, 205], 380: [50, 205, 101], 

        381: [101, 205, 36], 382: [100, 126, 142], 383: [100, 47, 126], 384: [101, 100, 36], 385: [100, 36, 142],
        386: [47, 217, 126], 387: [47, 217, 114], 388: [114, 217, 174], 389: [114, 174, 188], 390: [188, 174, 196], 

        391: [188, 196, 122], 392: [122, 196, 6], 393: [6, 196, 197], 394: [6, 197, 419], 395: [6, 419, 351],
        396: [351, 419, 412], 397: [412, 419, 399], 398: [412, 343, 399], 399: [399, 343, 437], 400: [343, 277, 437], 

        401: [437, 277, 355], 402: [277, 355, 329], 403: [355, 329, 371], 404: [329, 371, 266], 405: [329, 266, 330],
        406: [330, 266, 425], 407: [425, 330, 280], 408: [280, 425, 411], 409: [411, 425, 427], 410: [207, 216, 206], 

        411: [207, 205, 206], 412: [205, 36, 206], 413: [36, 206, 203], 414: [36, 203, 129], 415: [129, 36, 142],
        416: [142, 129, 209], 417: [142, 209, 126], 418: [126, 209, 198], 419: [126, 198, 217], 420: [217, 198, 174], 

        421: [174, 198, 236], 422: [174, 236, 196], 423: [196, 236, 3], 424: [3, 196, 197], 425: [3, 197, 195],
        426: [197, 195, 248], 427: [197, 248, 419], 428: [419, 248, 456], 429: [456, 399, 419], 430: [399, 456, 420], 

        431: [420, 437, 399], 432: [437, 420, 355], 433: [355, 420, 429], 434: [355, 429, 371], 435: [429, 371, 358],
        436: [371, 358, 266], 437: [358, 266, 423], 438: [423, 266, 426], 439: [426, 266, 425], 440: [426, 425, 427], 

        441: [426, 427, 436], 442: [209, 129, 49], 443: [209, 49, 198], 444: [198, 49, 131], 445: [198, 131, 134],
        446: [198, 134, 236], 447: [235, 134, 51], 448: [51, 236, 3], 449: [3, 51, 195], 450: [51, 195, 5], 

        451: [195, 5, 281], 452: [195, 281, 248], 453: [248, 281, 456], 454: [456, 281, 363], 455: [456, 363, 420],
        456: [363, 420, 360], 457: [420, 360, 279], 458: [420, 279, 429], 459: [429, 279, 358], 460: [129, 102, 49], 

        461: [49, 102, 48], 462: [49, 48, 131], 463: [131, 48, 115], 464: [131, 115, 220], 465: [131, 220, 134],
        466: [134, 220, 45], 467: [134, 45, 51], 468: [51, 45, 5], 469: [5, 45, 4], 470: [5, 4, 275], 

        471: [5, 275, 281], 472: [281, 275, 363], 473: [363, 275, 440], 474: [363, 440, 360], 475: [440, 360, 344],
        476: [360, 344, 278], 477: [360, 278, 279], 478: [279, 278, 331], 479: [279, 331, 358], 480: [212, 57, 43],
        
        481: [129, 102, 64], 482: [102, 48, 64], 483: [48, 64, 219], 484: [48, 115, 219], 485: [219, 115, 218],
        486: [115, 218, 220], 487: [220, 218, 237], 488: [220, 237, 44], 489: [220, 44, 45], 490: [45, 44, 1],
        
        491: [45, 1, 4], 492: [4, 1, 275], 493: [1, 275, 274], 494: [275, 274, 440], 495: [440, 457, 274],
        496: [440, 457, 438], 497: [440, 438, 344], 498: [344, 438, 439], 499: [344, 278, 439], 500: [278, 439, 294],
        
        501: [278, 294, 331], 502: [331, 294, 358], 503: [294, 358, 327], 504: [327, 358, 423], 505: [327, 423, 391],
        506: [423, 391, 426], 507: [391, 426, 322], 508: [322, 426, 436], 509: [322, 436, 410], 510: [410, 436, 432],
        
        511: [410, 287, 432], 512: [287, 432, 273], 513: [273, 432, 422], 514: [273, 335, 422], 515: [355, 422, 424],
        516: [335, 424, 406], 517: [406, 424, 418], 518: [406, 421, 418], 519: [313, 406, 421], 520: [313, 200, 421],
        
        521: [313, 200, 18], 522: [18, 83, 200], 523: [83, 200, 201], 524: [182, 83, 201], 525: [182, 201, 194],
        526: [106, 182, 204], 527: [182, 204, 194], 528: [202, 106, 204], 529: [202, 43, 106], 530: [212, 43, 202], 
        
        531: [216, 212, 186], 532: [186, 212, 57], 533: [206, 216, 92], 534: [92, 216, 186], 535: [203, 206, 165],
        536: [206, 92, 165], 537: [203, 98, 165], 538: [129, 203, 98], 539: [129, 98, 64], 540: [64, 98, 240], 

        541: [64, 240, 235], 542: [64, 219, 235], 543: [235, 240, 75], 544: [235, 59, 75], 545: [59, 75, 166],
        546: [219, 235, 59], 547: [219, 59, 166], 548: [219, 166, 218], 549: [218, 166, 79], 550: [218, 79, 239], 

        551: [218, 239, 237], 552: [237, 239, 241], 553: [241, 237, 44], 554: [44, 241, 125], 555: [44, 125, 19],
        556: [44, 1, 19], 557: [1, 19, 274], 558: [274, 19, 354], 559: [354, 274, 461], 560: [274, 461, 457], 

        561: [457, 461, 459], 562: [457, 459, 428], 563: [459, 438, 309], 564: [438, 309, 392], 565: [438, 392, 439],
        566: [392, 439, 289], 567: [289, 439, 455], 568: [439, 455, 294], 569: [392, 305, 289], 570: [289, 305, 455], 

        571: [305, 455, 460], 572: [455, 460, 294], 573: [294, 460, 327], 574: [98, 97, 167], 575: [98, 167, 165],
        576: [97, 2, 164], 577: [97, 164, 167], 578: [2, 164, 326], 579: [326, 164, 383], 580: [326, 327, 393], 

        581: [393, 327, 391], 582: [164, 267, 0], 583: [164, 393, 267], 584: [393, 267, 269], 585: [393, 269, 391],
        586: [391, 322, 269], 587: [269, 270, 322], 588: [322, 270, 410], 589: [410, 270, 409], 590: [409, 410, 287], 

        591: [409, 287, 291], 592: [291, 287, 375], 593: [375, 287, 273], 594: [375, 321, 273], 595: [321, 273, 335],
        596: [321, 335, 406], 597: [321, 406, 405], 598: [405, 406, 314], 599: [314, 406, 313], 600: [314, 313, 18], 

        601: [18, 314, 17], 602: [17, 84, 18], 603: [18, 84, 83], 604: [84, 83, 182], 605: [182, 84, 181],
        606: [181, 182, 91], 607: [91, 182, 106], 608: [91, 106, 43], 609: [146, 91, 43], 610: [57, 146, 43], 

        611: [57, 146, 61], 612: [57, 61, 185], 613: [185, 57, 186], 614: [186, 185, 40], 615: [186, 40, 92],
        616: [92, 40, 39], 617: [39, 92, 165], 618: [165, 39, 167], 619: [167, 39, 37], 620: [167, 37, 164], 

        621: [164, 37, 0] 
        }
    
    
class MagicLandmarks():
    """
    This class contains usefull lists of landmarks identification numbers.
    """
    high_prio_forehead = [10, 67, 69, 104, 108, 109, 151, 299, 337, 338]
    high_prio_nose = [3, 4, 5, 6, 45, 51, 115, 122, 131, 134, 142, 174, 195, 196, 197, 198,
                      209, 217, 220, 236, 248, 275, 277, 281, 360, 363, 399, 419, 420, 429, 437, 440, 456]
    high_prio_left_cheek = [36, 47, 50, 100, 101, 116, 117,
                            118, 119, 123, 126, 147, 187, 203, 205, 206, 207, 216]
    high_prio_right_cheek = [266, 280, 329, 330, 346, 347,
                             347, 348, 355, 371, 411, 423, 425, 426, 427, 436]

    mid_prio_forehead = [8, 9, 21, 68, 103, 251,
                         284, 297, 298, 301, 332, 333, 372, 383]
    mid_prio_nose = [1, 44, 49, 114, 120, 121, 128, 168, 188, 351, 358, 412]
    mid_prio_left_cheek = [34, 111, 137, 156, 177, 192, 213, 227, 234]
    mid_prio_right_cheek = [340, 345, 352, 361, 454]
    mid_prio_chin = [135, 138, 169, 170, 199, 208, 210, 211,
                     214, 262, 288, 416, 428, 430, 431, 432, 433, 434]
    mid_prio_mouth = [92, 164, 165, 167, 186, 212, 322, 391, 393, 410]
    # more specific areas
    forehead_left = [21, 71, 68, 54, 103, 104, 63, 70,
                     53, 52, 65, 107, 66, 108, 69, 67, 109, 105]
    forehead_center = [10, 151, 9, 8, 107, 336, 285, 55, 8]
    forehoead_right = [338, 337, 336, 296, 285, 295, 282,
                       334, 293, 301, 251, 298, 333, 299, 297, 332, 284]
    eye_right = [283, 300, 368, 353, 264, 372, 454, 340, 448,
                 450, 452, 464, 417, 441, 444, 282, 276, 446, 368]
    eye_left = [127, 234, 34, 139, 70, 53, 124,
                35, 111, 228, 230, 121, 244, 189, 222, 143]
    nose = [193, 417, 168, 188, 6, 412, 197, 174, 399, 456,
            195, 236, 131, 51, 281, 360, 440, 4, 220, 219, 305]
    mounth_up = [186, 92, 167, 393, 322, 410, 287, 39, 269, 61, 164]
    mounth_down = [43, 106, 83, 18, 406, 335, 273, 424, 313, 194, 204]
    chin = [204, 170, 140, 194, 201, 171, 175,
            200, 418, 396, 369, 421, 431, 379, 424]
    cheek_left_bottom = [215, 138, 135, 210, 212, 57, 216, 207, 192]
    cheek_right_bottom = [435, 427, 416, 364,
                          394, 422, 287, 410, 434, 436]
    cheek_left_top = [116, 111, 117, 118, 119, 100, 47, 126, 101, 123,
                      137, 177, 50, 36, 209, 129, 205, 147, 177, 215, 187, 207, 206, 203]
    cheek_right_top = [349, 348, 347, 346, 345, 447, 323,
                       280, 352, 330, 371, 358, 423, 426, 425, 427, 411, 376]
    
    # dense zones used for convex hull masks
    left_eye = [157,144, 145, 22, 23, 25, 154, 31, 160, 33, 46, 52, 53, 55,
                56, 189, 190, 63, 65, 66, 70, 221, 222, 223, 225, 226, 228, 
                229, 230, 231, 232, 105, 233, 107, 243, 124]
    right_eye = [384, 385, 386, 259, 388, 261, 265, 398, 276, 282, 283, 285, 
                 413, 293, 296, 300, 441, 442, 445, 446, 449, 451, 334, 463, 
                 336, 464, 467, 339, 341, 342, 353, 381, 373, 249, 253, 255]
    mounth = [391, 393, 11, 269, 270, 271, 287, 164, 165, 37, 167, 40, 43, 181,
              313, 314, 186, 57, 315, 61, 321, 73, 76, 335, 83, 85, 90, 106]
    
    # equispaced facial points - mouth and eyes are excluded.
    equispaced_facial_points = [2, 3, 4, 5, 6, 8, 9, 10, 18, 21, 32, 35, 36, 43, 46, 47, 48, 50, 54, \
             58, 67, 68, 69, 71, 92, 93, 101, 103, 104, 108, 109, 116, 117, \
             118, 123, 132, 134, 135, 138, 139, 142, 148, 149, 150, 151, 152, 182, 187, 188, 193, 197, 201, 205, 206, 207, \
             210, 211, 212, 216, 234, 248, 251, 262, 265, 266, 273, 277, 278, 280, \
             284, 288, 297, 299, 322, 323, 330, 332, 333, 337, 338, 345, \
             346, 361, 363, 364, 367, 368, 371, 377, 379, 411, 412, 417, 421, 425, 426, 427, 430, 432, 436]


def get_magic_landmarks():
    """ returns high_priority and mid_priority list of landmarks identification number """
    return [*MagicLandmarks.forehead_center, *MagicLandmarks.cheek_left_bottom, *MagicLandmarks.cheek_right_bottom], [*MagicLandmarks.forehoead_right, *MagicLandmarks.forehead_left, *MagicLandmarks.cheek_left_top, *MagicLandmarks.cheek_right_top]


if __name__ == "__main__":
    # print("torch cuda state: ", torch.cuda.is_available())
    
    polygon_test = False
    ukf_test = True
    
    if polygon_test:
        import os
        import matplotlib.pyplot as plt
        import cv2
        input_dir = "D:/home/BCML/drax/PAPER/data/frames/KDY_1_illuminance_30f80s/"
        file_list = os.listdir(input_dir)
        
        file_load = np.load(input_dir + file_list[0])
        file_load = cv2.cvtColor(file_load, cv2.COLOR_BGR2RGB)
        plt.imshow(file_load)
        plt.show()
        
        ex_polygon = SkinExtractionConvexHull_Polygon('CPU')
    
    fps = 30
    dt = 1/fps
    
    pukf = PointUKF(dt=dt)
    
    if ukf_test:
        import os
        import matplotlib.pyplot as plt
        
        base_dir = "D:/home/BCML/drax/PAPER/data/coordinates/raw/"
        file_list = os.listdir(base_dir)
    
        file_start = 8
    
        for sel_file in file_list[file_start:]:
            print(f"\n\n{sel_file} start\n\n")
            raw_empty_points = []
            raw_empty_value = []
            
            anomaly_points = []
            
            with open(base_dir+sel_file, 'r') as f:
                lines = f.readlines()
                total_length = int(lines[-1])
                ldmk_len = len(lines[0].split(','))-1
                print("ldmk_len: ", ldmk_len)
                
                coords = np.zeros((ldmk_len, 2, total_length))
                idx = 0
                line_idx = 0
                while idx < total_length:
                    temp_array = np.zeros((ldmk_len, 2))
                    try:
                        line_split = lines[line_idx].strip().split(',')
                        # print("line_split: ", line_split)
                    except IndexError:
                        print("indexError")
                        raw_empty_points.append(idx)
                        raw_empty_value.append(0)
                        coords[:, :, idx] = np.ones((coords.shape[0], coords.shape[1]))
                        line_idx += 1
                        idx += 1
                        continue
                    if "None" in lines[line_idx]:
                        # print("None")
                        raw_empty_points.append(idx)
                        raw_empty_value.append(0)
                        coords[:, :, idx] = np.ones((coords.shape[0], coords.shape[1]))
                        line_idx += 1
                        idx += 1
                        continue
                    if len(line_split) <= 1:
                        line_idx += 1
                        continue
                    for jdx in range(ldmk_len):
                        temp_point = line_split[jdx].split(' ')
                        # print(f"temp_point: {temp_point}, jdx: {jdx}")
                        y = int(temp_point[0].split('.')[0])
                        x = int(temp_point[1].split('.')[0])
                        temp_array[jdx, :] = np.array([y, x])
                        if x > 1000 or y > 1000:
                            anomaly_points.append([lines[line_idx], x, y, idx, jdx])
                    # print("temp_array:\n", temp_array)
                    coords[:, :, idx] = temp_array
                    idx += 1
                    line_idx += 1
                    if idx % 100 == 0:
                        print(f"{sel_file} {round(idx/total_length * 100, 2)} %")

            # print(coords)
            print_point = 4
            
            # filter setting
            transition = "stationary"
            q_scale_factor = 0.01
            r_scale_factor = 1
            # sel_filter = UKF(transition_func=transition, initial_state=coords[print_point, :, 0],
            #                  std_acc=q_scale_factor, std_meas=r_scale_factor)
            
            # filter test
            filtered_coords = np.zeros_like(coords)
            print("filtered_coords.shape: ", filtered_coords.shape)
            
            filters = [PointUKF(dt=dt) for _ in range(coords.shape[0])]
            
            found_0 = 1
            for i in range(filtered_coords.shape[2]):
                if i % 300 == 0:
                    print(f"i:{i}, {round(i//filtered_coords.shape[2]*100, 2)}%")
                for j in range(coords.shape[0]):
                    filters[j].predict()
                    if coords[j, 0, i] > 1 and coords[j, 1, i] > 1:
                        found_0 = 1
                        filters[j].update(coords[j, :, i])
                        filtered_coords[j, :, i] = filters[j].get_state()[:2]
                    else:
                        try:
                            found_0 += 1
                            filters[j].update(coords[j, :, i-found_0])
                            filtered_coords[j, :, i] = filters[j].get_state()[:2]
                        except:
                            filtered_coords[j, :, i] = [1, 1]
            np.save(f"D:/home/BCML/drax/PAPER/data/coordinates/stationary_kalman/{sel_file}.npy", filtered_coords)
# =============================================================================
#             for i in range(coords.shape[2]-1):
#                 input_coords = coords[print_point:print_point+2, :, i+1]
#                 print("filter input: ", input_coords)
#                 sel_filter.predict()
#                 sel_filter.update(input_coords)
#                 filtered_coords[:, :, i] = sel_filter.get_state()
#                 covariance_list.append(sel_filter.get_covariance())
#                 
#                 if i % 300 == 0:
#                     print(f"{sel_file}, filtering {round(i/(total_length-1) * 100, 2)} %")
# =============================================================================
            
            fig, ax = plt.subplots(2, 1, figsize=(20,9))
            ax[0].plot(coords[print_point, 0, :], label="Y")
            ax[0].plot(filtered_coords[0, :], label="stionary_filtered_Y")
            ax[0].set_ylabel("y point")
            ax[0].set_xlabel("frame")
            ax[0].scatter(raw_empty_points, raw_empty_value, c='r', s=20, label="not detected")
            ax[0].set_title("Y points", fontsize=18)
            
            ax[1].plot(coords[print_point, 1, :], label="X")
            ax[1].plot(filtered_coords[1, :], label="stationary_filtered_X")
            ax[1].set_ylabel("x point")
            ax[1].set_xlabel("frame")
            ax[1].scatter(raw_empty_points, raw_empty_value, c='r', s=20, label="not detected")
            ax[1].set_title("X points", fontsize=18)
            
            fig.suptitle(f"{sel_file}", fontsize=30)
            plt.legend()
            # plt.xlim(100,210)
            plt.show()
            
            # print("anomaly_points: ", anomaly_points)


    
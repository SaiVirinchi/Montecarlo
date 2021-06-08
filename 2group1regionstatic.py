
# %tensorflow_version 1.x

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from scipy.interpolate import griddata
from pyDOE import lhs
from mpl_toolkits.mplot3d import Axes3D
import time
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable


np.random.seed(1234)
tf.set_random_seed(1234)

class PhysicsInformedNN:
    # Initialize the class
    def __init__(self, x0, u0, v0, X_f, layers, lb, ub):
        
        X0 = x0 # (x0, 0)

        self.lb = lb
        self.ub = ub

        
               
        self.x0 = X0

        
        self.x_f = X_f
        
        self.u0 = u0
        self.v0 = v0
        
        # Initialize NNs
        self.layers = layers
        self.weights, self.biases = self.initialize_NN(layers)
        
        # tf Placeholders        
        print(self.v0.shape[0],"lol",np.shape(u0))

        self.x0_tf = tf.placeholder(tf.float32, shape=[None, self.x0.shape[1]])

        self.u0_tf = tf.placeholder(tf.float32, shape=[None, self.u0.shape[1]])
        self.v0_tf = tf.placeholder(tf.float32, shape=[None, self.v0.shape[1]])
        
        self.x_f_tf = tf.placeholder(tf.float32, shape=[None, self.x_f.shape[1]])


        #tf save
        self.saver = tf.train.Saver()

        # tf Graphs
        print("1")
        self.u0_pred, self.v0_pred, _ , _ = self.net_uv(self.x0_tf)
        print("2")
        self.f_u_pred, self.f_v_pred = self.net_f_uv(self.x_f_tf)
        
        # Loss
        self.loss = tf.reduce_mean(tf.square(self.u0_tf - self.u0_pred)) + \
                    tf.reduce_mean(tf.square(self.v0_tf - self.v0_pred)) + \
                    tf.reduce_mean(tf.square(self.f_u_pred)) + \
                    tf.reduce_mean(tf.square(self.f_v_pred))
        
        # Optimizers
        self.optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.loss, 
                                                                method = 'L-BFGS-B', 
                                                                options = {'maxiter': 50000,
                                                                           'maxfun': 50000,
                                                                           'maxcor': 50,
                                                                           'maxls': 50,
                                                                           'ftol' : 1.0* np.finfo(float).eps})
    
        self.optimizer_Adam = tf.train.AdamOptimizer()
        self.train_op_Adam = self.optimizer_Adam.minimize(self.loss)
                

        

        # tf session
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                     log_device_placement=True))
        
        init = tf.global_variables_initializer()
        self.sess.run(init)

        
              
    def initialize_NN(self, layers):        
        weights = []
        biases = []
        num_layers = len(layers) 
        vee= 0
        for l in range(0,num_layers-1):
            W = self.xavier_init(size=[layers[l], layers[l+1]])
            b = tf.Variable(tf.zeros([1,layers[l+1]], dtype=tf.float32), dtype=tf.float32)
            weights.append(W)
            biases.append(b)  
            vee = vee +1
            print(np.shape(weights[l]),np.shape(biases[l]),"lmao",vee)      
        return weights, biases
        
    def xavier_init(self, size):
        in_dim = size[0]
        out_dim = size[1]        
        xavier_stddev = np.sqrt(2/(in_dim + out_dim))
        return tf.Variable(tf.truncated_normal([in_dim, out_dim], stddev=xavier_stddev), dtype=tf.float32)
    
    def neural_net(self, X, weights, biases):
        num_layers = len(weights) + 1
        
        H = 2.0*(X - self.lb)/(self.ub - self.lb) - 1.0
        for l in range(0,num_layers-2):
            W = weights[l]
            b = biases[l]
            print(np.shape(W),np.shape(b),"WbH",l,np.shape(H))
            H = tf.tanh(tf.add(tf.matmul(H, W), b))
        W = weights[-1]
        b = biases[-1]
        Y = tf.add(tf.matmul(H, W), b)
        return Y
    
    def net_uv(self, x):
        X = x
        
        uv = self.neural_net(X, self.weights, self.biases)
        u = uv[:,0:1]
        v = uv[:,1:2]
        
        u_x = tf.gradients(u, x)[0]
        v_x = tf.gradients(v, x)[0]

        return u, v, u_x, v_x

    def net_f_uv(self, x):
        u, v, u_x, v_x = self.net_uv(x)
        
        u_xx = tf.gradients(u_x, x)[0]
        
        v_xx = tf.gradients(v_x, x)[0]
        
        

        f_u = (-1)*d1*v_xx + r1*u + (-1)*(1/k)*(nu*f1*u + nu*f2*v)
        f_v = (-1)*d2*u_xx + r2*v + (-1)*s1*u   
        
        return f_u, f_v
    
    def callback(self, loss):
        print('Loss:', loss)
        
    def train(self, nIter):
        
        tf_dict = {self.x0_tf: self.x0,
                   self.u0_tf: self.u0, self.v0_tf: self.v0,
                   self.x_f_tf: self.x_f}
        
        start_time = time.time()

#        restore_from_dir(self.sess, "./ckpt1/") 

        for it in range(nIter):
            self.sess.run(self.train_op_Adam, tf_dict)
            
            # Print
            if it % 10 == 0:
                elapsed = time.time() - start_time
                loss_value = self.sess.run(self.loss, tf_dict)
                print('It: %d, Loss: %.3e, Time: %.2f' % 
                      (it, loss_value, elapsed))
                start_time = time.time()
#             if it % 100 == 0:
#                 self.saver.save(self.sess, './ckpt1/model',global_step=it,meta_graph_suffix='meta', write_meta_graph=True, write_state=True,
#                                 strip_default_attrs=False, save_debug_info=False)

        self.optimizer.minimize(self.sess, 
                                feed_dict = tf_dict,         
                                fetches = [self.loss], 
                                loss_callback = self.callback)        
                                    
    
    def predict(self, X_star):
        
        tf_dict = {self.x0_tf: X_star[:,0:1]}
        
        u_star = self.sess.run(self.u0_pred, tf_dict)  
        v_star = self.sess.run(self.v0_pred, tf_dict)  
        
        
        tf_dict = {self.x_f_tf: X_star[:,0:1]}
        
        f_u_star = self.sess.run(self.f_u_pred, tf_dict)
        f_v_star = self.sess.run(self.f_v_pred, tf_dict)
               
        return u_star, v_star, f_u_star, f_v_star
    
if __name__ == "__main__": 
     
    noise = 0.0        
    
    # Doman bounds
    lb = -5.0
    ub = 5.0

    N0 = 100
    N_f = 20000
    layers = [1, 100, 100, 100, 100, 2]
        
    #CONSTANTS 
    d1 = 4
    d2 = 5
    r1 = 1
    r2 = 2
    s1 = 3
    f1 = 2
    f2 = 3
    F = 3
    nu = 2
    k = ((r2+d2*((np.pi/10)**2))*nu*f1+s1*nu*f2)/((d1*((np.pi/10)**2)+r1)*(d2*((np.pi/10)**2)+r2))

    x = np.linspace(lb,ub,400)
    x = np.reshape(x,(400,1))
    print("lota'",x.shape)
    Exact_u = F * np.cos((np.pi)*x/10)
    Exact_v = (F * np.cos((np.pi)*x/10))*s1/(r2+d2*((np.pi/10)**2))
    
    X_star = x.flatten()[:,None]
    u_star = Exact_u.T.flatten()[:,None]
    v_star = Exact_v.T.flatten()[:,None]
    ###########################
    
    idx_x = np.random.choice(x.shape[0], N0, replace=False)
    x0 = x[idx_x,:]
    u0 = Exact_u[idx_x,:]
    v0 = Exact_v[idx_x,:]
    
    
    X_f = lb + (ub-lb)*lhs(1, N_f)

    print(X_f.shape)        
    model = PhysicsInformedNN(x0, u0, v0, X_f, layers, lb, ub)
             
    start_time = time.time()                
    model.train(5000)
    elapsed = time.time() - start_time                
    print('Training time: %.4f' % (elapsed))
    
        
    u_pred, v_pred, f_u_pred, f_v_pred = model.predict(X_star)
    h_pred = np.sqrt(u_pred**2 + v_pred**2)
            
    error_u = np.linalg.norm(u_star-u_pred,2)/np.linalg.norm(u_star,2)
    error_v = np.linalg.norm(v_star-v_pred,2)/np.linalg.norm(v_star,2)
    print('Error u: %e' % (error_u))
    print('Error v: %e' % (error_v))

    
 
    ######################################################################
    ############################# Plotting ###############################
    ######################################################################    
    
      

    plt.plot(x,Exact_v[:], linewidth = 2,label = 'Exact')       
    plt.plot(x,v_pred[:], 'r', linewidth = 2, label = 'Prediction')
    plt.xlabel('x')
    plt.ylabel('T(t,x)')
    plt.legend(loc='upper center')
    plt.show()    
    
    plt.plot(x,Exact_u[:], linewidth = 2,label = 'Exact')       
    plt.plot(x,u_pred[:], 'r', linewidth = 2, label = 'Prediction')
    plt.xlabel('x')
    plt.ylabel('U(t,x)')  
    plt.legend(loc='upper center')
    plt.show()   

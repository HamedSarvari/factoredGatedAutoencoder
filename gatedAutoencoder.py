import numpy as np
import tensorflow as tf

SMALL = 0.0000001

# ===========================
# ======= U T I L S =========
# ===========================

# this function is 'borrowed' from keras
def random_binomial(shape, p=0.0, dtype=None, seed=None):
    """Returns a tensor with random binomial distribution of values.
    # Arguments
        shape: A tuple of integers, the shape of tensor to create.
        p: A float, `0. <= p <= 1`, probability of binomial distribution.
        dtype: String, dtype of returned tensor.
        seed: Integer, random seed.
    # Returns
        A tensor.
    """
    if dtype is None:
        dtype = 'float32'
    if seed is None:
        seed = np.random.randint(10e6)
    return tf.where(tf.random_uniform(shape, dtype=dtype, seed=seed) <= p,
                    tf.ones(shape, dtype=dtype),
                    tf.zeros(shape, dtype=dtype))


# ===========================
# ========= G A E ===========
# ===========================

class FactoredGatedAutoencoder:
    
    def __init__(self, numFactors, numHidden, learningRate=0.01, 
                 corrutionLevel=0.0,
                 normalize_data=True,
                 weight_decay_vis=0.0,
                 weight_decay_map=0.0):
        """ Create a factored autoencoder
            tbd
        """
        assert(corrutionLevel >= 0 and corrutionLevel <= 1)
        self.F = int(numFactors)
        self.H = int(numHidden)
        self.learningRate = learningRate
        self.p = corrutionLevel
        self.eps_vis = weight_decay_vis
        self.eps_map = weight_decay_map
        self.is_trained = False
        self.normalize_data = normalize_data
    
    def save(self, modelname):
        """ saves the model onto disk
        """
        assert(self.is_trained)
        np.save(modelname + "Wxf", self.Wxf_np)
        np.save(modelname + "Wyf", self.Wyf_np)
        np.save(modelname + "Whf", self.Whf_np)
        np.save(modelname + "Whf_in", self.Whf_in_np)
        np.save(modelname + "bmap", self.bmap_np)
        np.save(modelname + "bx", self.bx_np)
        np.save(modelname + "by", self.by_np)
        np.save(modelname + "Wtf", self.Wtf_np)
        np.save(modelname + "bt", self.bt_np)

    
    def load_from_weights(self, modelname):
        """ restores weights from disk
        """
        self.Wxf_np = np.load(modelname + "Wxf.npy")
        self.Wyf_np = np.load(modelname + "Wyf.npy")
        self.Whf_np = np.load(modelname + "Whf.npy")
        self.Whf_in_np = np.load(modelname + "Whf_in.npy")
        self.bmap_np = np.load(modelname + "bmap.npy")
        self.bx_np = np.load(modelname + "bx.npy")
        self.by_np = np.load(modelname + "by.npy")
        self.Wtf_np= np.load(modelname + "Wtf.npy")
        self.bt_np = np.load(modelname + "bt.npy")


        self.is_trained = True
    
    def inference(self, X, Y):
        """ inference
        """
        assert(self.is_trained)
        F = self.F
        H = self.H
        p = self.p
        lr = self.learningRate
        eps_vis = self.eps_vis
        eps_map = self.eps_map
        
        n, dim = X.shape
        assert (dim == Y.shape[1])
        #dim=len(X)


        
        numpy_rng = np.random.RandomState(1)
        
        # if self.normalize_data:
        #     X -= X.mean(0)[None, :]
        #     Y -= Y.mean(0)[None, :]
        #     X /= X.std(0)[None, :] + X.std() * 0.1
        #     Y /= Y.std(0)[None, :] + Y.std() * 0.1

        # x = tf.placeholder(tf.float32, [1,dim])
        # y = tf.placeholder(tf.float32, [1,dim])

        x = tf.placeholder(tf.float32, [None, dim])
        y = tf.placeholder(tf.float32, [None, dim])


        Wxf = tf.Variable(self.Wxf_np)
        Wyf = tf.Variable(self.Wyf_np)
        Whf = tf.Variable(self.Whf_np)
        Whf_in = tf.Variable(self.Whf_in_np)
        bmap = tf.Variable(self.bmap_np)
        Wtf=tf.Variable(self.Wtf_np)
        bt=tf.Variable(self.bt_np)

        fx = tf.matmul(x, Wxf)
        fy = tf.matmul(y, Wyf)
        mappings = tf.sigmoid(tf.matmul(tf.multiply(fx , fy), 
                                        tf.transpose(Whf_in)) + bmap)
        fH = tf.matmul(mappings, Whf)
        T = tf.sigmoid(tf.matmul(mappings, Wtf) + bt)


        with tf.Session() as sess:
            init = tf.global_variables_initializer()
            sess.run(init)
            out=sess.run(T, feed_dict={x: X, y: Y})
            
            print('export')
            #H = np.array(fH.eval())
            #out = T.eval()

            return out
        
    def train(self, X, Y, L,
              epochs=150,
              batch_size=1,
              print_debug=True):
        """ train the factored autoencoder
        X: x-input
        Y: y-input
        """
        F = self.F
        H = self.H
        p = self.p
        lr = self.learningRate
        eps_vis = self.eps_vis
        eps_map = self.eps_map
        
        n, dim = X.shape

        assert(dim == Y.shape[1])
        
        numpy_rng = np.random.RandomState(1)
        
        if self.normalize_data:
            X -= X.mean(0)[None, :]
            Y -= Y.mean(0)[None, :]
            X /= X.std(0)[None, :] + X.std() * 0.1
            Y /= Y.std(0)[None, :] + Y.std() * 0.1
        
        x = tf.placeholder(tf.float32, [None, dim])
        y = tf.placeholder(tf.float32, [None, dim])
        l=  tf.placeholder(tf.float32, [None,1])

        Wxf = tf.Variable(tf.random_normal(shape=(dim, F)) * 0.01)
        Wyf = tf.Variable(tf.random_normal(shape=(dim, F)) * 0.01)
        
        Whf = tf.Variable(np.exp(numpy_rng.uniform(
            low=-3.0, high=-2.0, size=(H, F)),dtype='float32'))
        Whf_in = tf.Variable(
            numpy_rng.uniform(
                low=-0.01, high=+0.01, size=(H, F)).astype('float32'))

        bmap = tf.Variable(np.zeros(H, dtype='float32'), name='bmap')
        bx = tf.Variable(np.zeros(dim, dtype='float32'), name='bx')
        by = tf.Variable(np.zeros(dim, dtype='float32'), name='by')

        if p > 0.0:
            x_corrupted = tf.multiply(random_binomial(tf.shape(x),p=p),x)
            y_corrupted = tf.multiply(random_binomial(tf.shape(y),p=p),y)
        else:
            x_corrupted = x
            y_corrupted = y
        
        fx = tf.matmul(x_corrupted , Wxf)
        fy = tf.matmul(y_corrupted , Wyf)
        
        mappings = tf.sigmoid(tf.matmul(tf.multiply(fx , fy), 
                                        tf.transpose(Whf_in)) + bmap)

        #### Add discriminative layer
        Wtf = tf.Variable(tf.random_normal(shape=(H, 1)) * 0.01)
        bt = tf.Variable(np.zeros(1, dtype='float32'), name='bt')
        T = tf.sigmoid(tf.matmul(mappings, Wtf)+bt)

        ####

        fH = tf.matmul(mappings, Whf)
        
        ox = tf.matmul(tf.multiply(fy , fH),tf.transpose(Wxf)) + bx
        oy = tf.matmul(tf.multiply(fx , fH),tf.transpose(Wyf)) + by

        cost_gen = tf.nn.l2_loss(ox-x) + tf.nn.l2_loss(oy-y)
        # Add descriminative loss
        cost_desc= tf.nn.l2_loss(T-l)

        # Define the hybrid cost

        cost = cost_gen + 0.4* cost_desc
        optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(cost)
        
        norm_Wxf = tf.nn.l2_normalize(Wxf, [0,1], epsilon=1e-12, name=None)
        norm_Wyf = tf.nn.l2_normalize(Wyf, [0,1], epsilon=1e-12, name=None)
        norm_Wtf = tf.nn.l2_normalize(Wtf, [0, 1], epsilon=1e-12, name=None)

        Wxf_normalize = Wxf.assign(norm_Wxf)
        Wyf_normalize = Wxf.assign(norm_Wyf)
        Wtf_normalize = Wtf.assign(norm_Wtf)
        
        with tf.Session() as sess:
            init = tf.global_variables_initializer()
            sess.run(init)
            
            for epoch in range(epochs):
                total_runs = int(n / batch_size)
                for i in range(total_runs):
                    randidx = np.random.randint(
                        n, size=batch_size).astype('int32')
                    batch_xs = X[randidx]
                    batch_ys = Y[randidx]
                    label = L[randidx]
                    sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys, l:label})
                    sess.run(Wxf_normalize)
                    sess.run(Wyf_normalize)
                    sess.run(Wtf_normalize)

                cost_ = sess.run(cost, feed_dict={x: X, y: Y, l: L }) / n
                if print_debug:
                    print ("Epoch: %03d/%03d cost: %.9f" %\
                           (epoch,epochs ,cost_) )

            # store weights
            self.Wxf_np = np.array(Wxf.eval(sess))
            print('ShapeX',self.Wxf_np.shape)

            self.Wyf_np = np.array(Wyf.eval(sess))

            print('ShapeY', self.Wyf_np.shape)

            self.Whf_np = np.array(Whf.eval(sess))

            print('ShapeH', self.Whf_np.shape)

            self.Whf_in_np = np.array(Whf_in.eval(sess))
            self.bmap_np = np.array(bmap.eval(sess))
            self.bx_np = np.array(bx.eval(sess))
            self.by_np = np.array(by.eval(sess))
            self.Wtf_np= np.array(Wtf.eval(sess))
            self.bt_np=np.array(bt.eval(sess))


            self.is_trained = True
            
        
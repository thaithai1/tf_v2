import tensorflow as tf
from utils import *

class Baseline:
    def __init__(self, X_train, Y_train, X_test, Y_test):
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_test = X_test
        self.Y_test = Y_test
        self.n0 = X_train.shape[0]
        self.nl = Y_train.shape[0]
        self.arch=[]                #format arch = [('Conv', parameters), ('Conv', parameters), ('FC', parameters)]
        
    def open(self):
        """
        Open and store tensorflow session
        """
        self.sess = tf.Session()

    def close(self):
        """
        Close session to prevent from leaks
        """
        self.sess.close()
    
    def add_FC(self, nh, act_func = 'relu'):
        """
        Add a fully connected layer

        INPUTS:
        --- h_units : Number of hidden units in the layer
        --- act_func : 'relu', 'sigmoid'
        """
        self.arch.append(('FC',{'nh':nh, 'act_func':act_func}))
    
    def add_Conv(self, f_h, f_w, c_out, act_func = 'relu', strides = [1, 1, 1, 1], padding = "SAME"):
        """
        Add a fully connected layer

        INPUTS:
        --- f_h : Filter heights
        --- f_w : Filter width
        --- c_out : Number of filters
        --- act_func : 'relu', 'sigmoid'
        """
        self.arch.append(('Conv',{'f_h':f_h, 'f_w':f_w, 'c_out':c_out, 'act_func':act_func, 'strides':strides, 'padding':padding}))

    def add_Max_pool(self):
        pass


    def init_param(self):
        """
        Setup Input and Output placeholders
        """
        X = tf.placeholder(dtype = tf.float32, shape = [self.n0, None], name='input')
        Y = tf.placeholder(dtype = tf.float32, shape = [self.nl, None],name='label')
        return X, Y
    
    def compute_loss(self,output, Y):
        pass
    
    def accuracy(self, output, Y):
        pass

    def optimizer(self, loss, lr = 1e-4, optimizer = 'adam'):
        """
        Set up an optimizer
        
        INPUTS:
        --- optimizer : adam / grad_desc
        """
        with tf.name_scope('train'):
            if optimizer == 'adam':
                train_step = tf.train.AdamOptimizer(lr).minimize(loss)
            elif optimizer == 'grad_desc':
                train_step = tf.train.GradientDescentOptimizer(lr).minimize(loss)
        return train_step

    def Forward_prop(self, X):
        pass

    def compile(self):
        """
        Create graph structure
        """
        tf.reset_default_graph()
        self.open()
        self.X, self.Y = self.init_param()
        self.output = self.Forward_prop(self.X)
        self.loss = self.compute_loss(self.output, self.Y)
        self.accuracy = self.accuracy(self.output, self.Y)
        self.sess.run(tf.global_variables_initializer())
        
    def predict_probs(self, X):
        return self.sess.run(self.output, feed_dict={'input:0' : X})

    def predict(self, X):
        pass

    def train(self, epoch, batch_size = 32, lr = 1e-2, optimizer = 'adam', filename = 'log_graph'):
        # loss = tf.get_default_graph().get_tensor_by_name("Loss/loss:0")
        output, accuracy, loss = self.output, self.accuracy, self.loss
        train_step = self.optimizer(loss, lr = lr, optimizer = optimizer)

        #summary
        loss_summary = tf.summary.scalar('Loss', loss)
        acc_summary = tf.summary.scalar('Accuracy', accuracy)
        test_summary = tf.summary.merge([loss_summary,acc_summary]) 
        merged_summary = tf.summary.merge_all()

        #Run
        writer = tf.summary.FileWriter(filename + '/train', self.sess.graph)
        writer_test = tf.summary.FileWriter(filename + '/test')
        self.sess.run(tf.global_variables_initializer())

        i=0
        for e in range(epoch):
            self.sess.run(train_step, feed_dict={'input:0':self.X_train, 'label:0':self.Y_train})
            if e%300 == 0:
                loss_train = self.sess.run(loss, feed_dict={'input:0': self.X_train, 'label:0': self.Y_train})
                loss_test, accuracy_test = self.sess.run([loss, accuracy], feed_dict={'input:0': self.X_test, 'label:0': self.Y_test})
                train_accuracy = self.sess.run(accuracy, feed_dict={'input:0':self.X_train, 'label:0':self.Y_train})
                print("epoch %s, training loss %s, test loss %s, test accuracy %s" % (e, loss_train, loss_test, accuracy_test))
            
            for X_temp, Y_temp in mini_batch(self.X_train, self.Y_train, batch_size = batch_size):
                self.sess.run(train_step, feed_dict={'input:0':X_temp, 'label:0':Y_temp})

                if i%20 == 0 :
                    s=self.sess.run(merged_summary, feed_dict={'input:0': X_temp, 'label:0': Y_temp})
                    writer.add_summary(s,i)
                    s=self.sess.run(test_summary, feed_dict={'input:0': self.X_test, 'label:0': self.Y_test})
                    writer_test.add_summary(s,i)
                i+=1
    

class FC_binary_class(Baseline):

    def accuracy(self, output, Y):
        """
        Accuracy for binary classification
        """
        with tf.name_scope('Accuracy'):
            predicted_class = tf.greater(output,0.5)
            correct = tf.equal(predicted_class, tf.equal(Y,1.0))
            accuracy = tf.reduce_mean( tf.cast(correct, 'float'))
        return accuracy

    def compute_loss(self,output, Y):
        """
        Binary crossentropy
        """
        with tf.name_scope('Loss'):
            loss = tf.reduce_mean(-Y * tf.log(output+ 1e-8) - (1 - Y) * tf.log(1 - output+ 1e-8), name ='loss')
        return loss
    
    def Forward_prop(self, X): #Change for new class
        nh_prev = self.n0
        layer = X
        for i, (_, parameters) in enumerate(self.arch):
            nh = parameters['nh']
            act_func = parameters['act_func']
            temp = layer
            layer = FC_layer(temp, nh, nh_prev, act_func = act_func)
            nh_prev = nh
        
        output = FC_layer(layer, self.nl, nh_prev,  act_func = 'sigmoid')
        return output
    
    def predict(self, X):
        probs = self.predict_probs(X)
        return (probs>0.5).astype(int)

class ConvNet_multiclass(Baseline):
    def Forward_prop(self, X):
        i = 0
        layer = X
        while self.arch[i][0] == 'Conv':
            parameters = self.arch[i][1]
            layer = Conv_layer(**parameters) 
            i+=1
        
        # layer = tf.reshape(layer,tf.shape[] 




def FC_layer(input, nl, nl_prev, act_func = 'relu'):
    with tf.name_scope(("FC")):
        W = tf.Variable(tf.truncated_normal([nl, nl_prev]), name = 'W') 
        b = tf.Variable(tf.constant(0.1, shape = [nl,1]), name = 'b')
        Z = tf.matmul(W, input) + b
        if act_func == 'relu':
            A = tf.nn.relu(Z, name = 'A')
        elif act_func =='sigmoid':
            A = tf.nn.sigmoid(Z, name = 'A')
        #Summary
        tf.summary.histogram("weights", W)
        tf.summary.histogram("biases", b)
        tf.summary.histogram("activations", A)
    return A


def Conv_layer(input, f_h, f_w,c_in, c_out, strides = [1, 1, 1, 1], padding = "SAME", act_func ='relu'):
    """
    INPUTS:
    --- f_h : filter heights
    --- f_w : filter width
    --- c_in : number of input channels
    --- c_out : number of filters
    """
    W = tf.Variable(tf.truncated_normal([f_h, f_w, c_in, c_out]), name = 'Kernel')
    b = tf.Variable(tf.constant(0.1, shape = [1, 1, c_out]))
    conv = tf.nn.conv2d(input, W, strides=strides, padding=padding,data_format = "HWCH", name = "Conv")
    Z = conv + b
    if act_func == 'relu':
        A = tf.nn.relu(Z, name = 'A')
    elif act_func =='sigmoid':
        A = tf.nn.sigmoid(Z, name = 'A')
    return A

def Pool(input, ksize = []):
    pass
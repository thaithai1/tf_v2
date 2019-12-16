import tensorflow as tf
import numpy as np

def monitor_operation():
    graph = tf.get_default_graph()  # The nodes of the TensorFlow graph are called “operations,” or “ops.” 
    operations = graph.get_operations()
    print("Operations\n",operations,"\n\n======================\n")
    print("Operations Details :\n")
    for op in operations: 
        print(op.name)      # We can see what operations are in the graph
        print("Input:")
        for op_input in op.inputs:print(op_input)
        print("---------------")

def mini_batch(X, Y, batch_size):
	"""
	OUTPUT:
	List : [(X1, Y1), (X2, Y2), ....] 
	"""
	res = []
	m = X.shape[1]
	if batch_size>m:
		return [(X,Y)]
	else:
		idx = np.random.permutation(m)
		nb_batch_full = m // batch_size
		for i in range(nb_batch_full):
				idx_0 = i * batch_size
				idx_1 = (i+1) * batch_size
				idx_temp = idx[idx_0 : idx_1]
				res.append((X[:,idx_temp],Y[:,idx_temp]))
		idx_temp = idx[idx_1: ]
		res.append((X[:,idx_temp],Y[:,idx_temp]))
		return res

def one_hot_encode(labels,C):
    """
    Arguments:
    labels -- vector containing the labels -- shape=(n,)
    C -- int. number of different classes

    Return:
    one-hot Matrix -- shape(n,C)
    """
    with tf.Session() as sess:
        res=tf.one_hot(labels,C)
        res=sess.run(res)
    return res
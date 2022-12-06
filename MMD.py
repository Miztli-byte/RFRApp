import numpy as np
import tensorflow as tf

#Note this file has been modified differently than orignal. Only the Maximum mean discrepancy loss was placed here 



def maximum_mean_discrepancy(x, y, sigma=1., kernel=gaussian_kernel_matrix):
  r"""Computes the Maximum Mean Discrepancy (MMD) of two samples: x and y.
  Maximum Mean Discrepancy (MMD) is a distance-measure between the samples of
  the distributions of x and y. Here we use the kernel two sample estimate
  using the empirical mean of the two distributions.
  MMD^2(P, Q) = || \E{\phi(x)} - \E{\phi(y)} ||^2
              = \E{ K(x, x) } + \E{ K(y, y) } - 2 \E{ K(x, y) },
  where K = <\phi(x), \phi(y)>,
    is the desired kernel function, in this case a radial basis kernel.
  Args:
      x: a tensor of shape [num_samples, num_features]
      y: a tensor of shape [num_samples, num_features]
      kernel: a function which computes the kernel in MMD. Defaults to the
              GaussianKernelMatrix.
  Returns:
      a scalar denoting the squared maximum mean discrepancy loss.
  """
  with tf.name_scope('MaximumMeanDiscrepancy'):
    # \E{ K(x, x) } + \E{ K(y, y) } - 2 \E{ K(x, y) }
    cost = tf.reduce_mean(kernel(x, x, sigma))
    cost += tf.reduce_mean(kernel(y, y, sigma))
    cost -= 2 * tf.reduce_mean(kernel(x, y, sigma))

    # We do not allow the loss to become negative.
    cost = tf.where(cost > 0, cost, 0, name='value')
  return cost


def compute_mmd_on_samples(Xs, Xt, max_size=2000, n_iters=10, sigma=1):
    size = min(len(Xs), len(Xt))
    if size > max_size: size = max_size
    mmds = np.zeros(n_iters)
    with tf.Session() as sess:

        K.set_session(sess)
        Xs_ = tf.placeholder(tf.float32, shape=(None, Xs.shape[1]))
        Xt_ = tf.placeholder(tf.float32, shape=(None, Xt.shape[1]))
        mmd = maximum_mean_discrepancy(Xs_, Xt_, sigma=sigma)
        sess.run(tf.global_variables_initializer())

        for i in range(n_iters):
            s_idx = np.random.choice(len(Xs), size=size, replace=False)
            t_idx = np.random.choice(len(Xt), size=size, replace=False)
            mmds[i] = sess.run(mmd, feed_dict={Xs_: Xs[s_idx], Xt_: Xt[t_idx]})

    return np.mean(mmds), np.std(mmds)

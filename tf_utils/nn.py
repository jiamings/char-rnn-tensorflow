import tensorflow as tf


@tf.RegisterGradient('BernoulliSampleStraightThrough')
def _bernoulli_sample_straight_through(op, grad):
    return [grad, tf.zeros(tf.shape(op.inputs[1]))]


def binary_round(x):
    g = tf.get_default_graph()

    with tf.name_scope('binary_round') as name:
        with g.gradient_override_map({'Round': 'Identity'}):
            return tf.round(x, name=name)


def bernoulli_sample(x):
    g = tf.get_default_graph()
    with tf.name_scope('bernoilli_sample') as name:
        with g.gradient_override_map({'Ceil': 'Identity', 'Sub': 'BernoulliSampleStraightThrough'}):
            return tf.ceil(x - tf.random_uniform(tf.shape(x)), name=name)


def pass_through_sigmoid(x):
    g = tf.get_default_graph()
    with tf.name_scope('pass_through_sigmoid') as name:
        with g.gradient_override_map({'Sigmoid': 'Identity'}):
            return tf.sigmoid(x, name=name)


def hard_sigmoid(x, slope=1.0):
    return tf.clip_by_value(0.5 * slope * x + 0.5, 0.0, 1.0)


def binary_stochastic_neuron_straight_through(x, *, slope=None, sigmoid_type='hard', stochastic=True):
    """
    Binary Stochastic Neuron with Straight Through Gradient Estimators
    :param x: logit tensor
    :param slope: slope for the Slope Annealing Trick (see http://arxiv.org/abs/1609.01704)
    :param sigmoid_type: the type of sigmoid to use.
    :param stochastic: binary stochastic function if True, or step function if false.
    :return:
    """
    if slope is None:
        slope = tf.constant(1.0)
    if sigmoid_type == 'pass_through':
        p = pass_through_sigmoid(x)
    elif sigmoid_type == 'hard':
        p = hard_sigmoid(x, slope)
    elif sigmoid_type == 'regular':
        p = tf.sigmoid(x * slope)
    else:
        raise ValueError('sigmoid_type only accepts "pass_through", "hard", and "regular"')

    if stochastic:
        return bernoulli_sample(p)
    else:
        return binary_round(p)


if __name__ == '__main__':
    # Sample code for a binary stochastic neuron.
    tf.reset_default_graph()

    x = tf.placeholder(tf.float32, shape=[])
    y = binary_stochastic_neuron_straight_through(x, slope=1.0, stochastic=True)
    g = tf.gradients(y, x)[0]
    s = tf.Session()
    print(s.run([y, g], feed_dict={x: -3.0}))
    print(s.run([y, g], feed_dict={x: -1.0}))
    print(s.run([y, g], feed_dict={x: 0.0}))
    print(s.run([y, g], feed_dict={x: 1.0}))
    print(s.run([y, g], feed_dict={x: 3.0}))

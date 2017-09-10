import tensorflow as tf
import collections
from nn import binary_stochastic_neuron_straight_through, bernoulli_sample

_HMRNNStateTuple = collections.namedtuple('HMRNNStateTuple', ('c', 'h', 'z'))


class HMRNNStateTuple(_HMRNNStateTuple):
    __slots__ = ()

    @property
    def dtype(self):
        (c, h, z) = self
        if not c.dtype == h.dtype:
            raise TypeError("Inconsistent internal state: %s vs %s" %
                            (str(c.dtype), str(h.dtype)))
        return c.dtype


class HierarchicalMultiscaleRNNCell(tf.nn.rnn_cell.RNNCell):
    def __init__(self, num_units, num_layers, layer_norm, reuse=None):
        """
        Initialization.
        :param num_units: a tuple, which contains the # of neurons at each layer.
        :param reuse: whether to resue the weights or not.
        """
        super(HierarchicalMultiscaleRNNCell, self).__init__(_reuse=reuse)
        self._num_units = num_units
        self._num_layers = num_layers
        self._layer_norm = layer_norm
        self.slope = 1.0  # self.slope = tf.placeholder(dtype=tf.float32, shape=[], name='slope')

    @property
    def state_size(self):
        return HMRNNStateTuple(
            self._num_units * self._num_layers,
            self._num_units * self._num_layers,
            self._num_layers)

    @property
    def output_size(self):
        return self._num_units * self._num_layers

    def __call__(self, inputs, state):
        c, h, z = state
        c = tf.unstack(tf.reshape(c, [-1, self._num_layers, self._num_units]), axis=1)
        h = tf.unstack(tf.reshape(h, [-1, self._num_layers, self._num_units]), axis=1)
        z = tf.unstack(tf.reshape(z, [-1, self._num_layers]), axis=1)
        new_c, new_h, new_z = [], [], []
        for l in range(0, self._num_layers):
            r = h[l]
            if l + 1 < self._num_layers:
                r = tf.concat([r, h[l+1] * tf.expand_dims(z[l], axis=1)], axis=1)
            r = tf.concat([r, inputs if l == 0 else new_h[l-1] * tf.expand_dims(new_z[l-1], axis=1)], axis=1)
            cc = self._linear(r, 4 * self._num_units + 1, name='hmrnn_{}'.format(l))
            f, i, o, g, nz = tf.split(cc, [self._num_units] * 4 + [1], axis=1)
            if self._layer_norm:
                i = self._norm(i, 'input_{}'.format(l))
                g = self._norm(g, 'transform_{}'.format(l))
                f = self._norm(f, 'forget_{}'.format(l))
                o = self._norm(o, 'output_{}'.format(l))

            i, f, o, g = tf.sigmoid(i), tf.sigmoid(f), tf.sigmoid(o), tf.tanh(g)
            nz = tf.squeeze(binary_stochastic_neuron_straight_through(nz, slope=self.slope), axis=1)
            # nz = tf.zeros_like(z[0])
            if l > 0:
                copy = tf.logical_and(tf.equal(new_z[l-1], 0.0), tf.equal(z[l], 0.0))
                flush = tf.equal(z[l], 1.0)
                update = tf.logical_and(tf.equal(new_z[l-1], 1.0), tf.equal(z[l], 0.0))

                nc = tf.where(flush, x=i * g, y=tf.where(update, x=f * c[l] + i * g, y=c[l]))
                nh = tf.where(copy, x=h[l], y=o * tf.tanh(nc))
            else:
                flush = tf.equal(z[l], 1.0)
                nc = tf.where(flush, x=i * g, y=f * c[l] + i * g)
                nh = o * tf.tanh(nc)
            new_c.append(nc)
            new_h.append(nh)
            new_z.append(nz)

        new_c = tf.reshape(tf.stack(new_c, axis=1), [-1, self._num_layers * self._num_units])
        new_h = tf.reshape(tf.stack(new_h, axis=1), [-1, self._num_layers * self._num_units])
        new_z = tf.stack(new_z, axis=1)
        return new_h, HMRNNStateTuple(new_c, new_h, new_z)

    def _linear(self, x, num_outputs, name):
        num_inputs = x.get_shape().as_list()[-1] # tf.shape(x)[-1]
        weights = tf.get_variable(name + '/weights', [num_inputs, num_outputs])
        out = tf.matmul(x, weights)
        if not self._layer_norm:
            bias = tf.get_variable(name + '/bias', [num_outputs], initializer=tf.zeros_initializer)
            out = out + bias
        return out

    @staticmethod
    def _norm(inp, scope):
        shape = inp.get_shape()[-1:]
        gamma_init = tf.constant_initializer(1.0)
        beta_init = tf.constant_initializer(0.0)
        with tf.variable_scope(scope):
            # Initialize beta and gamma for use by layer_norm.
            tf.get_variable("gamma", shape=shape, initializer=gamma_init)
            tf.get_variable("beta", shape=shape, initializer=beta_init)
        normalized = tf.contrib.layers.layer_norm(inp, reuse=True, scope=scope)
        return normalized


def gated_embedding_layer(x, num_outputs, num_layers, scope='gated_embedding', reuse=False):
    batch_size = x.get_shape().as_list()[0]
    time_steps = x.get_shape().as_list()[1]
    x = tf.reshape(x, [batch_size * time_steps, -1])
    x = tf.split(x, num_or_size_splits=num_layers, axis=1)

    with tf.variable_scope(scope, reuse=reuse):
        g = []
        xs = tf.concat(x, axis=1)
        num_inputs = xs.get_shape().as_list()[-1]
        for l in range(0, len(x)):
            weights = tf.get_variable("gates_{}/weights".format(l), shape=[num_inputs, 1])
            ng = tf.sigmoid(tf.matmul(xs, weights))
            g.append(ng)

        e = tf.zeros(tf.stack([tf.shape(x[0])[0], num_outputs]))
        for l in range(0, len(x)):
            weights = tf.get_variable('weights_{}'.format(l), shape=[x[l].get_shape().as_list()[-1], num_outputs])
            ne = g[l] * tf.matmul(x[l], weights)
            e = e + ne
        logits = tf.nn.relu(e)
    logits = tf.reshape(logits, [batch_size, time_steps, -1])
    return logits


if __name__ == '__main__':
    num_units, num_layers, batch_size, time_steps, num_outputs = 1, 3, 10, 5, 15
    x = tf.random_uniform(shape=[batch_size, time_steps, 10])
    cell = HierarchicalMultiscaleRNNCell(num_units=num_units, num_layers=num_layers, layer_norm=True)
    outputs, states = tf.nn.dynamic_rnn(
        cell, x, initial_state=HMRNNStateTuple(
            c=tf.random_uniform(shape=[batch_size, num_layers * num_units], minval=-1.0, maxval=1.0),
            h=tf.random_uniform(shape=[batch_size, num_layers * num_units], minval=-1.0, maxval=1.0),
            z=bernoulli_sample(tf.random_uniform(shape=[batch_size, num_layers]))
        )
    )
    logits = gated_embedding_layer(outputs, num_outputs, num_layers, scope='gated_embedding')
    s = tf.Session()
    s.run(tf.global_variables_initializer())
    lg = s.run(logits)
    print(lg.shape)

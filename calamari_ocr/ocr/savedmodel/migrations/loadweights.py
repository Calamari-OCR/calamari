def load_and_convert_weights(ckpt, dry_run=True):
    import tensorflow as tf

    tf.compat.v1.reset_default_graph()
    with tf.compat.v1.Session() as sess:
        available_vars = tf.compat.v1.train.list_variables(ckpt)
        available_var_names = [var_name for var_name, _ in available_vars]

        for var_name in available_var_names:
            var = tf.compat.v1.train.load_variable(ckpt, var_name)
            # bias and kernel changed, unfortunately I do not know how to transform it...
            if var_name.endswith('_lstm/kernel'):
                rec_size = var.shape[1] // 4

                # this split into recurrent kernel should be valid
                kernel = var[:-rec_size]
                rec_kernel = var[-rec_size:]

                tf.Variable(kernel, name=var_name)
                tf.Variable(rec_kernel, name=var_name.replace('kernel', 'recurrent_kernel'))
            elif var_name.endswith('_lstm/bias'):
                # this might be required
                dims = len(var) // 4
                var[dims:dims*2] += 1
                tf.Variable(var, name=var_name)
            else:
                tf.Variable(var, name=var_name)

        if not dry_run:
            # Save the variables
            saver = tf.compat.v1.train.Saver()
            sess.run(tf.compat.v1.global_variables_initializer())
            saver.save(sess, ckpt)

    tf.compat.v1.reset_default_graph()

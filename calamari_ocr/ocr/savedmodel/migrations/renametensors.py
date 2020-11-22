def from_to_prefix(replace_from, replace_to, add_prefix, force_prefix=False):
    def op(var_name):
        # Set the new name
        new_name = var_name
        if None not in [replace_from, replace_to]:
            new_name = new_name.replace(replace_from, replace_to)
        if add_prefix:
            if force_prefix or not new_name.startswith(add_prefix):
                # force prefix or add prefix if it does not exist yet
                new_name = add_prefix + new_name

        return new_name

    return op


def rename(checkpoint, op, dry_run):
    import tensorflow as tf
    tf.compat.v1.reset_default_graph()
    with tf.compat.v1.Session() as sess:
        for var_name, _ in tf.compat.v1.train.list_variables(checkpoint):
            # Load the variable
            var = tf.compat.v1.train.load_variable(checkpoint, var_name)

            # Set the new name
            new_name = op(var_name)

            if dry_run:
                print('%s would be renamed to %s.' % (var_name, new_name))
            else:
                if var_name == new_name:
                    print('No change for {}'.format(var_name))
                else:
                    print('Renaming %s to %s.' % (var_name, new_name))

                # Rename the variable
                tf.Variable(var, name=new_name)

        if not dry_run:
            # Save the variables
            saver = tf.compat.v1.train.Saver()
            sess.run(tf.compat.v1.global_variables_initializer())
            saver.save(sess, checkpoint)

    tf.compat.v1.reset_default_graph()

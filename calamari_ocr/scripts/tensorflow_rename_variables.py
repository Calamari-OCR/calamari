import argparse

from calamari_ocr.utils import glob_all
from tqdm import tqdm
import os

usage_str = 'python tensorflow_rename_variables.py --checkpoints=path_to_models.json ' \
            '--replace_from=substr --replace_to=substr --add_prefix=abc --dry_run'


def rename(checkpoint, replace_from, replace_to, add_prefix, dry_run, force_prefix=False):
    import tensorflow as tf
    tf.compat.v1.reset_default_graph()
    with tf.compat.v1.Session() as sess:
        for var_name, _ in tf.compat.v1.train.list_variables(checkpoint):
            # Load the variable
            var = tf.compat.v1.train.load_variable(checkpoint, var_name)

            # Set the new name
            new_name = var_name
            if None not in [replace_from, replace_to]:
                new_name = new_name.replace(replace_from, replace_to)
            if add_prefix:
                if force_prefix or not new_name.startswith(add_prefix):
                    # force prefix or add prefix if it does not exist yet
                    new_name = add_prefix + new_name

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


def main():
    parser = argparse.ArgumentParser(description=usage_str)
    parser.add_argument('--checkpoints', nargs='+', type=str, required=True)
    parser.add_argument('--replace_from')
    parser.add_argument('--replace_to')
    parser.add_argument('--add_prefix')
    parser.add_argument('--dry_run', action='store_true')

    args = parser.parse_args()

    for ckpt in tqdm(glob_all(args.checkpoints)):
        ckpt = os.path.splitext(ckpt)[0]
        rename(ckpt, args.replace_from, args.replace_to, args.add_prefix, args.dry_run)


if __name__ == '__main__':
    main()

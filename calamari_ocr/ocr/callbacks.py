import bidi.algorithm as bidi


class TrainingCallback:
    def display(self, train_cer, train_loss, train_dt, iter, steps_per_epoch, display_epochs,
                example_pred, example_gt):
        pass

    def early_stopping(self, eval_cer, n_total, n_best, iter):
        pass

    def training_finished(self, total_time, total_iters):
        pass


class ConsoleTrainingCallback:
    def display(self, train_cer, train_loss, train_dt, iter, steps_per_epoch, display_epochs,
                example_pred, example_gt):
        if display_epochs:
            print("#{:08f}: loss={:.8f} ler={:.8f} dt={:.8f}s".format(
                iter / steps_per_epoch, train_loss, train_cer,
                train_dt))
        else:
            print("#{:08d}: loss={:.8f} ler={:.8f} dt={:.8f}s".format(
                iter, train_loss, train_cer, train_dt))

        lr = "\u202A\u202B"
        print("  PRED: '{}{}{}'".format(lr[bidi.get_base_level(example_pred)], example_pred, "\u202C"))
        print("  TRUE: '{}{}{}'".format(lr[bidi.get_base_level(example_gt)], example_gt, "\u202C"))

    def early_stopping(self, eval_cer, n_total, n_best, iter):
        pass

    def training_finished(self, total_time, total_iters):
        print("Total training time {}s for {} iterations.".format(total_time, total_iters))

from absl import app, flags, logging
from absl.flags import FLAGS
import os
import tensorflow as tf

from modules.models import CifarModel
from modules.dataset import load_cifar10_dataset
from modules.lr_scheduler import CosineAnnealingLR
from modules.losses import CrossEntropyLoss
from modules.utils import (
    set_memory_growth, load_yaml, count_parameters_in_MB, ProgressBar,
    AvgrageMeter, accuracy)


flags.DEFINE_string('cfg_path', './configs/pcdarts_cifar10.yaml',
                    'config file path')
flags.DEFINE_string('gpu', '0', 'which gpu to use')


def main(_):
    # init
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu

    logger = tf.get_logger()
    logger.disabled = True
    logger.setLevel(logging.FATAL)
    set_memory_growth()

    cfg = load_yaml(FLAGS.cfg_path)

    # define network
    model = CifarModel(cfg, training=True)
    model.summary(line_length=80)
    print("param size = {:f}MB".format(count_parameters_in_MB(model)))

    # load dataset
    train_dataset = load_cifar10_dataset(
        cfg['batch_size'], split='train', shuffle=True, drop_remainder=True,
        using_normalize=cfg['using_normalize'], using_crop=cfg['using_crop'],
        using_flip=cfg['using_flip'], using_cutout=cfg['using_cutout'],
        cutout_length=cfg['cutout_length'])
    val_dataset = load_cifar10_dataset(
        cfg['val_batch_size'], split='test', shuffle=False,
        drop_remainder=False, using_normalize=cfg['using_normalize'],
        using_crop=False, using_flip=False, using_cutout=False)

    # define optimizer
    steps_per_epoch = cfg['dataset_len'] // cfg['batch_size']
    learning_rate = CosineAnnealingLR(
        initial_learning_rate=cfg['init_lr'],
        t_period=cfg['epoch'] * steps_per_epoch, lr_min=cfg['lr_min'])
    optimizer = tf.keras.optimizers.SGD(
        learning_rate=learning_rate, momentum=cfg['momentum'])

    # define losses function
    criterion = CrossEntropyLoss()

    # load checkpoint
    checkpoint_dir = './checkpoints/' + cfg['sub_name']
    checkpoint = tf.train.Checkpoint(step=tf.Variable(0, name='step'),
                                     optimizer=optimizer,
                                     model=model)
    manager = tf.train.CheckpointManager(checkpoint=checkpoint,
                                         directory=checkpoint_dir,
                                         max_to_keep=3)
    if manager.latest_checkpoint:
        checkpoint.restore(manager.latest_checkpoint)
        print('[*] load ckpt from {} at step {}.'.format(
            manager.latest_checkpoint, checkpoint.step.numpy()))
    else:
        print("[*] training from scratch.")

    # define training step function
    @tf.function
    def train_step(inputs, labels, drop_path_prob):
        with tf.GradientTape() as tape:
            logits, logits_aux = model((inputs, drop_path_prob), training=True)

            losses = {}
            losses['reg'] = tf.reduce_sum(model.losses)
            losses['ce'] = criterion(labels, logits)
            losses['ce_auxiliary'] = \
                cfg['auxiliary_weight'] * criterion(labels, logits_aux)
            total_loss = tf.add_n([l for l in losses.values()])

        grads = tape.gradient(total_loss, model.trainable_variables)
        grads = [(tf.clip_by_norm(grad, cfg['grad_clip'])) for grad in grads]
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        return logits, total_loss, losses

    # training loop
    summary_writer = tf.summary.create_file_writer('./logs/' + cfg['sub_name'])
    total_steps = steps_per_epoch * cfg['epoch']
    remain_steps = max(total_steps - checkpoint.step.numpy(), 0)
    prog_bar = ProgressBar(steps_per_epoch,
                           checkpoint.step.numpy() % steps_per_epoch)

    train_acc = AvgrageMeter()
    val_acc = AvgrageMeter()
    best_acc = 0.
    for inputs, labels in train_dataset.take(remain_steps):
        checkpoint.step.assign_add(1)
        drop_path_prob = cfg['drop_path_prob'] * (
            tf.cast(checkpoint.step, tf.float32) / total_steps)
        steps = checkpoint.step.numpy()
        epochs = ((steps - 1) // steps_per_epoch) + 1

        logits, total_loss, losses = train_step(inputs, labels, drop_path_prob)
        train_acc.update(
            accuracy(logits.numpy(), labels.numpy())[0], cfg['batch_size'])

        prog_bar.update(
            "epoch={}/{}, loss={:.4f}, acc={:.2f}, lr={:.2e}".format(
                epochs, cfg['epoch'], total_loss.numpy(), train_acc.avg,
                optimizer.lr(steps).numpy()))

        if steps % cfg['val_steps'] == 0 and steps > 1:
            print("\n[*] validate...", end='')
            val_acc.reset()
            for inputs_val, labels_val in val_dataset:
                logits_val, _ = model((inputs_val, tf.constant([0.])))
                val_acc.update(
                    accuracy(logits_val.numpy(), labels_val.numpy())[0],
                    inputs_val.shape[0])

            if val_acc.avg > best_acc:
                best_acc = val_acc.avg
                model.save_weights(f"checkpoints/{cfg['sub_name']}/best.ckpt")

            val_str = " val acc {:.2f}%, best acc {:.2f}%"
            print(val_str.format(val_acc.avg, best_acc), end='')

        if steps % 10 == 0:
            with summary_writer.as_default():
                tf.summary.scalar('acc/train', train_acc.avg, step=steps)
                tf.summary.scalar('acc/val', val_acc.avg, step=steps)

                tf.summary.scalar(
                    'loss/total_loss', total_loss, step=steps)
                for k, l in losses.items():
                    tf.summary.scalar('loss/{}'.format(k), l, step=steps)
                tf.summary.scalar(
                    'learning_rate', optimizer.lr(steps), step=steps)

        if steps % cfg['save_steps'] == 0:
            manager.save()
            print("\n[*] save ckpt file at {}".format(
                manager.latest_checkpoint))

        if steps % steps_per_epoch == 0:
            train_acc.reset()

    manager.save()
    print("\n[*] training done! save ckpt file at {}".format(
        manager.latest_checkpoint))


if __name__ == '__main__':
    app.run(main)

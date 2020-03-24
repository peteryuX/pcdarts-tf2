from absl import app, flags, logging
from absl.flags import FLAGS
import os
import tensorflow as tf

from modules.models_search import SearchNetArch
from modules.dataset import load_cifar10_dataset
from modules.lr_scheduler import CosineAnnealingLR
from modules.losses import CrossEntropyLoss
from modules.utils import (
    set_memory_growth, load_yaml, count_parameters_in_MB, ProgressBar,
    AvgrageMeter, accuracy)


flags.DEFINE_string('cfg_path', './configs/pcdarts_cifar10_search.yaml',
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
    sna = SearchNetArch(cfg)
    sna.model.summary(line_length=80)
    print("param size = {:f}MB".format(count_parameters_in_MB(sna.model)))

    # load dataset
    t_split = f"train[0%:{int(cfg['train_portion'] * 100)}%]"
    v_split = f"train[{int(cfg['train_portion'] * 100)}%:100%]"
    train_dataset = load_cifar10_dataset(
        cfg['batch_size'], split=t_split, shuffle=True, drop_remainder=True,
        using_normalize=cfg['using_normalize'], using_crop=cfg['using_crop'],
        using_flip=cfg['using_flip'], using_cutout=cfg['using_cutout'],
        cutout_length=cfg['cutout_length'])
    val_dataset = load_cifar10_dataset(
        cfg['batch_size'], split=v_split, shuffle=True, drop_remainder=True,
        using_normalize=cfg['using_normalize'], using_crop=cfg['using_crop'],
        using_flip=cfg['using_flip'], using_cutout=cfg['using_cutout'],
        cutout_length=cfg['cutout_length'])

    # define optimizer
    steps_per_epoch = int(
        cfg['dataset_len'] * cfg['train_portion'] // cfg['batch_size'])
    learning_rate = CosineAnnealingLR(
        initial_learning_rate=cfg['init_lr'],
        t_period=cfg['epoch'] * steps_per_epoch, lr_min=cfg['lr_min'])
    optimizer = tf.keras.optimizers.SGD(
        learning_rate=learning_rate, momentum=cfg['momentum'])
    optimizer_arch = tf.keras.optimizers.Adam(
        learning_rate=cfg['arch_learning_rate'], beta_1=0.5, beta_2=0.999)

    # define losses function
    criterion = CrossEntropyLoss()

    # load checkpoint
    checkpoint_dir = './checkpoints/' + cfg['sub_name']
    checkpoint = tf.train.Checkpoint(step=tf.Variable(0, name='step'),
                                     optimizer=optimizer,
                                     optimizer_arch=optimizer_arch,
                                     model=sna.model,
                                     alphas_normal=sna.alphas_normal,
                                     alphas_reduce=sna.alphas_reduce,
                                     betas_normal=sna.betas_normal,
                                     betas_reduce=sna.betas_reduce)
    manager = tf.train.CheckpointManager(checkpoint=checkpoint,
                                         directory=checkpoint_dir,
                                         max_to_keep=3)
    if manager.latest_checkpoint:
        checkpoint.restore(manager.latest_checkpoint)
        print('[*] load ckpt from {} at step {}.'.format(
            manager.latest_checkpoint, checkpoint.step.numpy()))
    else:
        print("[*] training from scratch.")
    print(f"[*] searching model after {cfg['start_search_epoch']} epochs.")

    # define training step function for model
    @tf.function
    def train_step(inputs, labels):
        with tf.GradientTape() as tape:
            logits = sna.model((inputs, *sna.arch_parameters), training=True)

            losses = {}
            losses['reg'] = tf.reduce_sum(sna.model.losses)
            losses['ce'] = criterion(labels, logits)
            total_loss = tf.add_n([l for l in losses.values()])

        grads = tape.gradient(total_loss, sna.model.trainable_variables)
        grads = [(tf.clip_by_norm(grad, cfg['grad_clip'])) for grad in grads]
        optimizer.apply_gradients(zip(grads, sna.model.trainable_variables))

        return logits, total_loss, losses

    # define training step function for arch_parameters
    @tf.function
    def train_step_arch(inputs, labels):
        with tf.GradientTape() as tape:
            logits = sna.model((inputs, *sna.arch_parameters), training=True)

            losses = {}
            losses['reg'] = cfg['arch_weight_decay'] * tf.add_n(
                [tf.reduce_sum(p**2) for p in sna.arch_parameters])
            losses['ce'] = criterion(labels, logits)
            total_loss = tf.add_n([l for l in losses.values()])

        grads = tape.gradient(total_loss, sna.arch_parameters)
        optimizer_arch.apply_gradients(zip(grads, sna.arch_parameters))

        return losses

    # training loop
    summary_writer = tf.summary.create_file_writer('./logs/' + cfg['sub_name'])
    total_steps = steps_per_epoch * cfg['epoch']
    remain_steps = max(total_steps - checkpoint.step.numpy(), 0)
    prog_bar = ProgressBar(steps_per_epoch,
                           checkpoint.step.numpy() % steps_per_epoch)

    train_acc = AvgrageMeter()
    for inputs, labels in train_dataset.take(remain_steps):
        checkpoint.step.assign_add(1)
        steps = checkpoint.step.numpy()
        epochs = ((steps - 1) // steps_per_epoch) + 1

        if epochs > cfg['start_search_epoch']:
            inputs_val, labels_val = next(iter(val_dataset))
            arch_losses = train_step_arch(inputs_val, labels_val)

        logits, total_loss, losses = train_step(inputs, labels)
        train_acc.update(
            accuracy(logits.numpy(), labels.numpy())[0], cfg['batch_size'])

        prog_bar.update(
            "epoch={:d}/{:d}, loss={:.4f}, acc={:.2f}, lr={:.2e}".format(
                epochs, cfg['epoch'], total_loss.numpy(), train_acc.avg,
                optimizer.lr(steps).numpy()))

        if steps % 10 == 0:
            with summary_writer.as_default():
                tf.summary.scalar('acc/train', train_acc.avg, step=steps)

                tf.summary.scalar(
                    'loss/total_loss', total_loss, step=steps)
                for k, l in losses.items():
                    tf.summary.scalar('loss/{}'.format(k), l, step=steps)
                tf.summary.scalar(
                    'learning_rate', optimizer.lr(steps), step=steps)

                if epochs > cfg['start_search_epoch']:
                    for k, l in arch_losses.items():
                        tf.summary.scalar(
                            'arch_losses/{}'.format(k), l, step=steps)
                    tf.summary.scalar('arch_learning_rate',
                                      cfg['arch_learning_rate'], step=steps)

        if steps % cfg['save_steps'] == 0:
            manager.save()
            print("\n[*] save ckpt file at {}".format(
                manager.latest_checkpoint))

        if steps % steps_per_epoch == 0:
            train_acc.reset()
            if epochs > cfg['start_search_epoch']:
                genotype = sna.get_genotype()
                print(f"\nsearch arch: {genotype}")
                f = open(os.path.join(
                    './logs', cfg['sub_name'], 'search_arch_genotype.py'), 'a')
                f.write(f"\n{cfg['sub_name']}_{epochs} = {genotype}\n")
                f.close()

    manager.save()
    print("\n[*] training done! save ckpt file at {}".format(
        manager.latest_checkpoint))


if __name__ == '__main__':
    app.run(main)

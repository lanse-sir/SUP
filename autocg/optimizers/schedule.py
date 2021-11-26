def LinearWarmupRsqrtDecay(warm_up, init_lr, max_lr, step):
    if step < warm_up:
        lr_step = max_lr - init_lr
        lr_step = lr_step / warm_up
        lr = init_lr + lr_step * step
    else:
        step = step / warm_up
        lr = max_lr * (step ** -0.5)
    return lr
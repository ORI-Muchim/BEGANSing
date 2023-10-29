import sys
sys.path.append('utils')
sys.path.append('model')

import os
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm

from utils.config_parser import Config
from utils.file_utils import create_path
from utils.torch_utils import set_device, save_checkpoint, load_checkpoint

import dataprocess
from model.models import Generator, Discriminator
from utils.logger import Logger

class AverageMeter(object):
    def __init__(self):
        self.steps = 0
        self.reset()

    def reset(self):
        self.val = 0.0
        self.sum = 0.0
        self.num = 0
        self.avg = 0.0

    def step(self, val, num=1):
        self.val = val
        self.sum += num*val
        self.num += num
        self.steps += 1
        self.avg = self.sum/self.num

def criterionAdv(D, x):
    return torch.mean(torch.abs(D(x) - x))

def main():
    config = Config()
    config_basename = os.path.basename(config.config[0])
    print("Configuration file: \'%s\'" % (config_basename))

    checkpoint_path = create_path(config.checkpoint_path, action=config.checkpoint_path_action)
    config.save(os.path.join(checkpoint_path, config_basename))
    logger = Logger(os.path.join(checkpoint_path, 'log'))

    dataloader = dataprocess.load_train(config)
    step_size = config.step_epoch*len(dataloader.train)

    G = Generator(config)
    D = Discriminator(config)
    G, D = set_device((G, D), config.device, config.use_cpu)

    criterionL1 = nn.L1Loss()
    optimizerG = torch.optim.Adam(G.parameters(), lr=config.learn_rate, betas=config.betas, weight_decay=config.weight_decay)
    optimizerD = torch.optim.Adam(D.parameters(), lr=config.learn_rate, betas=config.betas, weight_decay=config.weight_decay)
    schedulerG = StepLR(optimizerG, step_size=step_size, gamma=config.decay_factor)
    schedulerD = StepLR(optimizerD, step_size=step_size, gamma=config.decay_factor)
    
    checkpoint_G = os.path.join(checkpoint_path, 'latest_G.pt')
    checkpoint_D = os.path.join(checkpoint_path, 'latest_D.pt')
    
        
    if os.path.exists(checkpoint_G) and os.path.exists(checkpoint_D):
        print("Loading checkpoints...")
        G, optimizerG, learning_rate_G, iteration_G = load_checkpoint(checkpoint_G, G, optimizerG, verbose=True)
        D, optimizerD, learning_rate_D, iteration_D = load_checkpoint(checkpoint_D, D, optimizerD, verbose=True)
        
        epoch = iteration_G // len(dataloader.train)
        print("Resuming from epoch %d" % epoch)
    else:
        epoch = 0
        print("No checkpoint found, starting from epoch 0")

    
    k = 0.0
    M = AverageMeter()
    lossG_train = AverageMeter()
    lossG_valid = AverageMeter()
    lossD_train = AverageMeter()

    print('Training start')
    for epoch in range(epoch, config.stop_epoch + 1):
        # Training Loop
        G.train()
        D.train()
        for batch in tqdm(dataloader.train, leave=False, ascii=True):
            x, y_prev, y = set_device(batch, config.device, config.use_cpu)
            y = y.unsqueeze(1)

            optimizerG.zero_grad()
            y_gen = G(x, y_prev)
            lossL1 = criterionL1(y_gen, y)
            loss_advG = criterionAdv(D, y_gen)
            lossG = lossL1 + loss_advG
            lossG.backward()
            optimizerG.step()
            schedulerG.step()

            optimizerD.zero_grad()
            loss_real = criterionAdv(D, y)
            loss_fake = criterionAdv(D, y_gen.detach())
            loss_advD = loss_real - k*loss_fake
            loss_advD.backward()
            optimizerD.step()
            schedulerD.step()

            diff = torch.mean(config.gamma*loss_real - loss_fake)
            k = k + config.lambda_k*diff.item()
            k = min(max(k, 0), 1)

            measure = (loss_real + torch.abs(diff)).data
            M.step(measure, y.size(0))

            logger.log_train(lossL1, loss_advG, lossG, loss_real, loss_fake, loss_advD, M.avg, k, lossG_train.steps)
            lossG_train.step(lossG.item(), y.size(0))
            lossD_train.step(loss_advD.item(), y.size(0))

        # Validation Loop
        G.eval()
        D.eval()
        for batch in tqdm(dataloader.valid, leave=False, ascii=True):
            x, y_prev, y = set_device(batch, config.device, config.use_cpu)
            y = y.unsqueeze(1)

            y_gen = G(x, y_prev)
            lossL1 = criterionL1(y_gen, y)
            loss_advG = criterionAdv(D, y_gen)
            lossG = lossL1 + loss_advG

            logger.log_valid(lossL1, loss_advG, lossG, lossG_valid.steps)
            lossG_valid.step(lossG.item(), y.size(0))

        for param_group in optimizerG.param_groups:
            learn_rate = param_group['lr']

        print("[Epoch %d/%d] [loss G train: %.5f] [loss G valid: %.5f] [loss D train: %.5f] [lr: %.6f]" %
            (epoch, config.stop_epoch, lossG_train.avg, lossG_valid.avg, lossD_train.avg, learn_rate))
        
        lossG_train.reset()
        lossG_valid.reset()
        lossD_train.reset()

    def save_model_and_optimizer(epoch, G, D, optimizerG, optimizerD, lossG_train, lossD_train, checkpoint_path, learn_rate, config):
        def save(model, optimizer, loss, model_name):
            savename = os.path.join(checkpoint_path, model_name)
            save_checkpoint(savename, model, epoch, optimizer, learn_rate, loss.steps, verbose=True)

        save(G, optimizerG, lossG_train, 'latest_G.pt')
        save(D, optimizerD, lossD_train, 'latest_D.pt')

        if epoch % config.save_epoch == 0:
            save(G, optimizerG, lossG_train, f'epoch{epoch}_G.pt')
            save(D, optimizerD, lossD_train, f'epoch{epoch}_D.pt')

    save_model_and_optimizer(epoch, G, D, optimizerG, optimizerD, lossG_train, lossD_train, checkpoint_path, learn_rate, config)

    print('Training finished')

if __name__ == "__main__":
    main()
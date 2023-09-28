import sys

# sys.path.append('../../')

from utils.utils import save_checkpoint, AverageMeter, Logger, accuracy, mkdirs, time_to_str
from utils.evaluate import eval
from utils.dataset import get_dataset
from utils.dataset import get_dataset_one_to_one
from fas import fas_model_fix

import random
import numpy as np
from config import configC, configM, configI, configO, config_cefa, config_surf, config_wmca
from config import config_CI, config_CO , config_CM, config_MC, config_MI, config_MO, config_IC, config_IO, config_IM, config_OC, config_OI, config_OM
from datetime import datetime
import time
from timeit import default_timer as timer

import os
import torch
import torch.nn as nn
import torch.optim as optim
import argparse


torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = 'cuda'


def train(config):
  # # 5-shot
  # src1_train_dataloader_fake, src1_train_dataloader_real, src2_train_dataloader_fake, src2_train_dataloader_real, src3_train_dataloader_fake, src3_train_dataloader_real, _, _, src5_train_dataloader_fake, src5_train_dataloader_real, test_dataloader = get_dataset(
  # # src1_train_dataloader_fake, src1_train_dataloader_real, src2_train_dataloader_fake, src2_train_dataloader_real, src3_train_dataloader_fake, src3_train_dataloader_real, src4_train_dataloader_fake, src4_train_dataloader_real, src5_train_dataloader_fake, src5_train_dataloader_real, test_dataloader = get_dataset( # for mcio
  #     config.src1_data, config.src1_train_num_frames, config.src2_data,
  #     config.src2_train_num_frames, config.src3_data,
  #     config.src3_train_num_frames, config.src4_data,
  #     config.src4_train_num_frames, config.src5_data,
  #     config.src5_train_num_frames, config.tgt_data, config.tgt_test_num_frames)

  # 0-shot
  # src1_train_dataloader_fake, src1_train_dataloader_real, src2_train_dataloader_fake, src2_train_dataloader_real, src3_train_dataloader_fake, src3_train_dataloader_real, _, _, _, _, test_dataloader = get_dataset( # for wcs
  src1_train_dataloader_fake, src1_train_dataloader_real, src2_train_dataloader_fake, src2_train_dataloader_real, src3_train_dataloader_fake, src3_train_dataloader_real, src4_train_dataloader_fake, src4_train_dataloader_real, _, _, test_dataloader = get_dataset( # for MCIO
      config.src1_data, config.src1_train_num_frames, config.src2_data,
      config.src2_train_num_frames, config.src3_data,
      config.src3_train_num_frames, config.src4_data,
      config.src4_train_num_frames, config.src5_data,
      config.src5_train_num_frames, config.tgt_data, config.tgt_test_num_frames)

  # # 1-1 setting
  # src1_train_dataloader_fake, src1_train_dataloader_real, src2_train_dataloader_fake, src2_train_dataloader_real, src3_train_dataloader_fake, src3_train_dataloader_real, test_dataloader = get_dataset_one_to_one(
  # # src1_train_dataloader_fake, src1_train_dataloader_real, src2_train_dataloader_fake, src2_train_dataloader_real, test_dataloader = get_dataset_one_to_one(
  # # src1_train_dataloader_fake, src1_train_dataloader_real, test_dataloader = get_dataset_one_to_one(
  #     config.src1_data, config.src1_train_num_frames, 
  #     config.src2_data, config.src2_train_num_frames,
  #     config.src3_data, config.src3_train_num_frames,
  #     config.tgt_data, config.tgt_test_num_frames)

  best_model_ACC = 0.0
  best_model_HTER = 1.0
  best_model_ACER = 1.0
  best_model_AUC = 0.0
  best_TPR_FPR = 0.0

  valid_args = [np.inf, 0, 0, 0, 0, 0, 0, 0]

  loss_classifier = AverageMeter()
  classifer_top1 = AverageMeter()

  log = Logger()
  log.write(
      '\n----------------------------------------------- [START %s] %s\n\n' %
      (datetime.now().strftime('%Y-%m-%d %H:%M:%S'), '-' * 51))
  log.write('** start training target model! **\n')
  log.write(
      '--------|------------- VALID -------------|--- classifier ---|------ Current Best ------|--------------|\n'
  )
  log.write(
      '  iter  |   loss   top-1   HTER    AUC    |   loss   top-1   |   top-1   HTER    AUC    |    time      |\n'
  )
  log.write(
      '-------------------------------------------------------------------------------------------------------|\n'
  )
  start = timer()
  criterion = {'softmax': nn.CrossEntropyLoss().cuda()}

  net1 = fas_model_fix().to(device) # conventional vit

  for name, param in net1.named_parameters():
      param.requires_grad = True
  
  optimizer_dict = [
      {
          'params': filter(lambda p: p.requires_grad, net1.parameters()),
          'lr': 0.0001 # LR = 1e-4
      },
  ]
  optimizer1 = optim.Adam(optimizer_dict, lr=0.0001, weight_decay=0.000001) # LR = 1e-4

  src1_train_iter_real = iter(src1_train_dataloader_real)
  src1_iter_per_epoch_real = len(src1_train_iter_real)
  src2_train_iter_real = iter(src2_train_dataloader_real)
  src2_iter_per_epoch_real = len(src2_train_iter_real)
  src3_train_iter_real = iter(src3_train_dataloader_real)
  src3_iter_per_epoch_real = len(src3_train_iter_real)
  src4_train_iter_real = iter(src4_train_dataloader_real)
  src4_iter_per_epoch_real = len(src4_train_iter_real)
  # src5_train_iter_real = iter(src5_train_dataloader_real)
  # src5_iter_per_epoch_real = len(src5_train_iter_real)

  src1_train_iter_fake = iter(src1_train_dataloader_fake)
  src1_iter_per_epoch_fake = len(src1_train_iter_fake)
  src2_train_iter_fake = iter(src2_train_dataloader_fake)
  src2_iter_per_epoch_fake = len(src2_train_iter_fake)
  src3_train_iter_fake = iter(src3_train_dataloader_fake)
  src3_iter_per_epoch_fake = len(src3_train_iter_fake)
  src4_train_iter_fake = iter(src4_train_dataloader_fake)
  src4_iter_per_epoch_fake = len(src4_train_iter_fake)
  # src5_train_iter_fake = iter(src5_train_dataloader_fake)
  # src5_iter_per_epoch_fake = len(src5_train_iter_fake)

  epoch = 1
  iter_per_epoch = 100

  for iter_num in range(4000 + 1):
    if (iter_num % src1_iter_per_epoch_real == 0):
      src1_train_iter_real = iter(src1_train_dataloader_real)
    if (iter_num % src2_iter_per_epoch_real == 0):
      src2_train_iter_real = iter(src2_train_dataloader_real)
    if (iter_num % src3_iter_per_epoch_real == 0):
      src3_train_iter_real = iter(src3_train_dataloader_real)
    if (iter_num % src4_iter_per_epoch_real == 0):
      src4_train_iter_real = iter(src4_train_dataloader_real)
    # if (iter_num % src5_iter_per_epoch_real == 0):
    #   src5_train_iter_real = iter(src5_train_dataloader_real)

    if (iter_num % src1_iter_per_epoch_fake == 0):
      src1_train_iter_fake = iter(src1_train_dataloader_fake)
    if (iter_num % src2_iter_per_epoch_fake == 0):
      src2_train_iter_fake = iter(src2_train_dataloader_fake)
    if (iter_num % src3_iter_per_epoch_fake == 0):
      src3_train_iter_fake = iter(src3_train_dataloader_fake)
    if (iter_num % src4_iter_per_epoch_fake == 0):
      src4_train_iter_fake = iter(src4_train_dataloader_fake)
    # if (iter_num % src5_iter_per_epoch_fake == 0):
    #   src5_train_iter_fake = iter(src5_train_dataloader_fake)

    if (iter_num != 0 and iter_num % iter_per_epoch == 0):
      epoch = epoch + 1

    net1.train(True)
    optimizer1.zero_grad()
    ######### data prepare #########
    src1_img_real, src1_label_real = src1_train_iter_real.next()
    src1_img_real = src1_img_real.cuda()
    src1_label_real = src1_label_real.cuda()
    input1_real_shape = src1_img_real.shape[0]

    src2_img_real, src2_label_real = src2_train_iter_real.next()
    src2_img_real = src2_img_real.cuda()
    src2_label_real = src2_label_real.cuda()
    input2_real_shape = src2_img_real.shape[0]

    src3_img_real, src3_label_real = src3_train_iter_real.next()
    src3_img_real = src3_img_real.cuda()
    src3_label_real = src3_label_real.cuda()
    input3_real_shape = src3_img_real.shape[0]

    src4_img_real, src4_label_real = src4_train_iter_real.next()
    src4_img_real = src4_img_real.cuda()
    src4_label_real = src4_label_real.cuda()
    input4_real_shape = src4_img_real.shape[0]

    # src5_img_real, src5_label_real = src5_train_iter_real.next()
    # src5_img_real = src5_img_real.cuda()
    # src5_label_real = src5_label_real.cuda()
    # input5_real_shape = src5_img_real.shape[0]

    src1_img_fake, src1_label_fake = src1_train_iter_fake.next()
    src1_img_fake = src1_img_fake.cuda()
    src1_label_fake = src1_label_fake.cuda()
    input1_fake_shape = src1_img_fake.shape[0]

    src2_img_fake, src2_label_fake = src2_train_iter_fake.next()
    src2_img_fake = src2_img_fake.cuda()
    src2_label_fake = src2_label_fake.cuda()
    input2_fake_shape = src2_img_fake.shape[0]

    src3_img_fake, src3_label_fake = src3_train_iter_fake.next()
    src3_img_fake = src3_img_fake.cuda()
    src3_label_fake = src3_label_fake.cuda()
    input3_fake_shape = src3_img_fake.shape[0]

    src4_img_fake, src4_label_fake = src4_train_iter_fake.next()
    src4_img_fake = src4_img_fake.cuda()
    src4_label_fake = src4_label_fake.cuda()
    input4_fake_shape = src4_img_fake.shape[0]

    # src5_img_fake, src5_label_fake = src5_train_iter_fake.next()
    # src5_img_fake = src5_img_fake.cuda()
    # src5_label_fake = src5_label_fake.cuda()
    # input5_fake_shape = src5_img_fake.shape[0]

    if config.tgt_data in ['cefa', 'surf', 'wmca']:
      input_data = torch.cat([
          src1_img_real, src1_img_fake, 
          src2_img_real, src2_img_fake,
          src3_img_real, src3_img_fake, 
          # src5_img_real, src5_img_fake
      ],
                             dim=0)
    else:
      input_data = torch.cat([
          src1_img_real, src1_img_fake, 
          src2_img_real, src2_img_fake,
          src3_img_real, src3_img_fake, 
          src4_img_real, src4_img_fake,
          # src5_img_real, src5_img_fake
      ],
                             dim=0)

    if config.tgt_data in ['cefa', 'surf', 'wmca']:
      source_label = torch.cat([
          src1_label_real.fill_(1),
          src1_label_fake.fill_(0),
          src2_label_real.fill_(1),
          src2_label_fake.fill_(0),
          src3_label_real.fill_(1),
          src3_label_fake.fill_(0),
          # src5_label_real.fill_(1),
          # src5_label_fake.fill_(0)
      ],
                               dim=0)
    else:
      source_label = torch.cat([
          src1_label_real.fill_(1),
          src1_label_fake.fill_(0),
          src2_label_real.fill_(1),
          src2_label_fake.fill_(0),
          src3_label_real.fill_(1),
          src3_label_fake.fill_(0),
          src4_label_real.fill_(1),
          src4_label_fake.fill_(0),
          # src5_label_real.fill_(1),
          # src5_label_fake.fill_(0)
      ],
                               dim=0)

    ######### forward #########
    classifier_label_out, feature = net1(input_data, True)

    cls_loss = criterion['softmax'](classifier_label_out.narrow(
        0, 0, input_data.size(0)), source_label)
    total_loss = cls_loss
    total_loss.backward()
    optimizer1.step()
    optimizer1.zero_grad()

    loss_classifier.update(cls_loss.item())
    acc = accuracy(
        classifier_label_out.narrow(0, 0, input_data.size(0)),
        source_label,
        topk=(1,))
    classifer_top1.update(acc[0])

    if (iter_num != 0 and (iter_num + 1) % (iter_per_epoch) == 0):
        valid_args = eval(test_dataloader, net1, True)
        # judge model according to HTER
        is_best = valid_args[3] <= best_model_HTER
        best_model_HTER = min(valid_args[3], best_model_HTER)
        threshold = valid_args[5]
        if (valid_args[3] <= best_model_HTER):
            best_model_ACC = valid_args[6]
            best_model_AUC = valid_args[4]
            best_TPR_FPR = valid_args[-1]

        save_list = [
        epoch, valid_args, best_model_HTER, best_model_ACC, best_model_ACER,
        threshold
        ]
        save_checkpoint(save_list, is_best, net1,
                        os.path.join(config.op_dir , config.tgt_data + f'_vit_0_shot_checkpoint_run_{str(config.run)}.pth.tar'))

        print('\r', end='', flush=True)
        log.write(
            '  %4.1f  |  %5.3f  %6.3f  %6.3f  %6.3f  |  %6.3f  %6.3f  |  %6.3f  %6.3f  %6.3f  | %s   %s'
            % ((iter_num + 1) / iter_per_epoch, valid_args[0], valid_args[6],
                valid_args[3] * 100, valid_args[4] * 100, loss_classifier.avg,
                classifer_top1.avg, float(best_model_ACC),
                float(best_model_HTER * 100), float(
                    best_model_AUC * 100), time_to_str(timer() - start, 'min'), 0))
        log.write('\n')

  return best_model_HTER*100.0, best_model_AUC*100.0, best_TPR_FPR*100.0
  


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--config', type=str)
  parser.add_argument('--op_dir', type=str, default=None)
  parser.add_argument('--report_logger_path', type=str, default=None)
  args = parser.parse_args()

  # 0-shot / 5-shot
  if args.config == 'I':
    config = configI
  if args.config == 'C':
    config = configC
  if args.config == 'M':
    config = configM
  if args.config == 'O':
    config = configO
  if args.config == 'cefa':
    config = config_cefa
  if args.config == 'surf':
    config = config_surf
  if args.config == 'wmca':
    config = config_wmca


  # 1-1 setting
  if args.config == 'CI':
    config = config_CI
  elif args.config == 'CO':
    config = config_CO
  elif args.config == 'CM':
    config = config_CM
  elif args.config == 'MC':
    config = config_MC
  elif args.config == 'MI':
    config = config_MI
  elif args.config == 'MO':
    config = config_MO
  elif args.config == 'IC':
    config = config_IC
  elif args.config == 'IM':
    config = config_IM
  elif args.config == 'IO':
    config = config_IO
  elif args.config == 'OC':
    config = config_OC
  elif args.config == 'OM':
    config = config_OM
  elif args.config == 'OI':
    config = config_OI



  for attr in dir(config):
    if attr.find('__') == -1:
      print('%s = %r' % (attr, getattr(config, attr)))
  
  config.op_dir = str(args.op_dir)
  
  with open(args.report_logger_path, "w") as f:
    f.write('Run, HTER, AUC, TPR@FPR=1%\n')
    hter_avg = []
    auc_avg = []
    tpr_fpr_avg = []
    for i in range(5):
      # To reproduce results
      torch.manual_seed(i)
      np.random.seed(i)

      config.run = i
      hter, auc, tpr_fpr = train(config)

      hter_avg.append(hter)
      auc_avg.append(auc)
      tpr_fpr_avg.append(tpr_fpr)

      f.write(f'{i},{hter},{auc},{tpr_fpr}\n')
    
    hter_mean = np.mean(hter_avg)
    auc_mean = np.mean(auc_avg)
    tpr_fpr_mean = np.mean(tpr_fpr_avg)
    f.write(f'Mean,{hter_mean},{auc_mean},{tpr_fpr_mean}\n')

    hter_std = np.std(hter_avg)
    auc_std = np.std(auc_avg)
    tpr_fpr_std = np.std(tpr_fpr_avg)
    f.write(f'Std dev,{hter_std},{auc_std},{tpr_fpr_std}\n')


import torch
from torch.utils.data import DataLoader

import argparse
import numpy as np
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ['CUDA_VISIBLE_DEVICES'] = "1"

import nets
# import dataloader
from dataloader import load_MVSEC
import dataloader
from dataloader import transforms
from utils import utils
import model

# StereoSpike
# import time
# import random
# from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torchvision import transforms
# from torch.utils.tensorboard import SummaryWriter

# from spikingjelly.clock_driven import functional
# from spikingjelly.clock_driven import surrogate

# from datasets.MVSEC import load_MVSEC
# from dataloader.data_augmentation import ToTensor, RandomHorizontalFlip, RandomVerticalFlip, RandomTimeMirror, \
#     RandomEventDrop
#
# from network.SNN_models import StereoSpike, fromZero_feedforward_multiscale_tempo_Matt_SpikeFlowNetLike
# from network.ANN_models import StereoSpike_equivalentANN
#
# from network.metrics import MeanDepthError, log_to_lin_depths, disparity_to_depth
# from network.loss import Total_Loss
#
# from viz import show_learning


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

parser = argparse.ArgumentParser()
# --mode train_all \
# --no_validate

parser.add_argument('--mode', default='train_all', type=str,
                    help='Validation mode on small subset or test mode on full test data')

# Training data (using MVSEC for training)
parser.add_argument('--data_dir', default='/data/tr/StereoData/MVSEC_DVS', type=str, help='Training dataset')
parser.add_argument('--dataset_name', default='MVSEC_DVS', type=str, help='Dataset name')

parser.add_argument('--batch_size', default=2, type=int, help='Batch size for training')
parser.add_argument('--val_batch_size', default=5, type=int, help='Batch size for validation')
parser.add_argument('--num_workers', default=8, type=int, help='Number of workers for data loading')
parser.add_argument('--img_height', default=384, type=int, help='Image height for training')
parser.add_argument('--img_width', default=1248, type=int, help='Image width for training')

# For KITTI, using 384x1248 for validation
# For MVSEC, using 260*346 for validation
parser.add_argument('--val_img_height', default=260, type=int, help='Image height for validation')
parser.add_argument('--val_img_width', default=346, type=int, help='Image width for validation')

# Model
parser.add_argument('--seed', default=326, type=int, help='Random seed for reproducibility')
parser.add_argument('--checkpoint_dir', default='checkpoints/saanet+_testMV', type=str,
                    help='Directory to save model checkpoints and logs')
parser.add_argument('--learning_rate', default=1e-4, type=float, help='Learning rate')
parser.add_argument('--weight_decay', default=1e-4, type=float, help='Weight decay for optimizer')
parser.add_argument('--max_disp', default=192, type=int, help='Max disparity')
parser.add_argument('--max_epoch', default=100, type=int, help='Maximum epoch number for training')
parser.add_argument('--resume', action='store_true', help='Resume training from latest checkpoint')

# AANet
parser.add_argument('--feature_type', default='ganet', type=str, help='Type of feature extractor')
parser.add_argument('--no_feature_mdconv', action='store_true', help='Whether to use mdconv for feature extraction')
parser.add_argument('--feature_pyramid', action='store_true', default=True, help='Use pyramid feature')
parser.add_argument('--feature_pyramid_network', action='store_true', help='Use FPN')
parser.add_argument('--feature_similarity', default='correlation', type=str,
                    help='Similarity measure for matching cost')
parser.add_argument('--num_downsample', default=2, type=int, help='Number of downsample layer for feature extraction')
parser.add_argument('--aggregation_type', default='adaptive', type=str, help='Type of cost aggregation')
parser.add_argument('--num_scales', default=3, type=int, help='Number of stages when using parallel aggregation')
parser.add_argument('--num_fusions', default=6, type=int, help='Number of multi-scale fusions when using parallel'
                                                               'aggragetion')
parser.add_argument('--num_stage_blocks', default=1, type=int, help='Number of deform blocks for ISA')
parser.add_argument('--num_deform_blocks', default=3, type=int, help='Number of DeformBlocks for aggregation')
parser.add_argument('--no_intermediate_supervision', action='store_true',
                    help='Whether to add intermediate supervision')
parser.add_argument('--deformable_groups', default=2, type=int, help='Number of deformable groups')
parser.add_argument('--mdconv_dilation', default=2, type=int, help='Dilation rate for deformable conv')
parser.add_argument('--refinement_type', default='hourglass', help='Type of refinement module')

parser.add_argument('--pretrained_aanet', default=None, type=str, help='Pretrained network')
parser.add_argument('--freeze_bn', action='store_true', default=True, help='Switch BN to eval mode to fix running statistics')

# Learning rate
parser.add_argument('--lr_decay_gamma', default=0.5, type=float, help='Decay gamma')
parser.add_argument('--lr_scheduler_type', default='MultiStepLR', help='Type of learning rate scheduler')
parser.add_argument('--milestones', default='400,600,800,900', type=str, help='Milestones for MultiStepLR')

# Loss
parser.add_argument('--highest_loss_only', action='store_true', default=True, help='Only use loss on highest scale for finetuning')
# parser.add_argument('--load_pseudo_gt', action='store_true', default=True, help='Load pseudo gt for supervision')
parser.add_argument('--load_pseudo_gt', action='store_true', help='Load pseudo gt for supervision')

# Log
parser.add_argument('--print_freq', default=100, type=int, help='Print frequency to screen (iterations)')
parser.add_argument('--summary_freq', default=100, type=int, help='Summary frequency to tensorboard (iterations)')
parser.add_argument('--no_build_summary', action='store_true', help='Dont save sammary when training to save space')
parser.add_argument('--save_ckpt_freq', default=10, type=int, help='Save checkpoint frequency (epochs)')

parser.add_argument('--evaluate_only', action='store_true', help='Evaluate pretrained models')
parser.add_argument('--no_validate', action='store_true', default=True, help='No validation')
parser.add_argument('--strict', action='store_true', help='Strict mode when loading checkpoints')
parser.add_argument('--val_metric', default='epe', help='Validation metric to select best model')

args = parser.parse_args()
logger = utils.get_logger()

utils.check_path(args.checkpoint_dir)
utils.save_args(args)

filename = 'command_test.txt' if args.mode == 'test' else 'command_train.txt'
utils.save_command(args.checkpoint_dir, filename)


def main():
    # For reproducibility
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)

    torch.backends.cudnn.benchmark = True

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #StereoSpike

    ######################
    # GENERAL PARAMETERS #
    ######################

    nfpdm = 1  # number of frames per depth map (1 label every 50 ms)
    N_inference = 1  # number of chunks for training/testing (1 chunk = 50 ms = nfpdm frames)
    N_warmup = 1  # number of chunks for warmup (if you want to use a stateful model)
    batchsize = 1
    learned_metric = 'LIN'  # learn metric depth ('LIN'), normalized log depth ('LOG') or disparity ('DISP')
    learning_rate = 0.0002
    weight_decay = 0.0
    n_epochs = 70
    show = False  # display network's predictions during training / validation

    ###########################
    # VISUALIZATION FUNCTIONS #
    ###########################

    plt.ion()
    fig = plt.figure()

    ########
    # DATA #
    ########

    # Train loader
    train_transform_list = [transforms.RandomCrop(args.img_height, args.img_width),
                            #transforms.RandomColor(),
                            # transforms.RandomVerticalFlip(),
                            transforms.ToTensor(),
                            # transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
                            ]
    # tsfm = transforms.Compose([
    #     ToTensor(),
    #     RandomHorizontalFlip(p=0.5),
    #     RandomVerticalFlip(p=0.5),
    #     RandomTimeMirror(p=0.5),
    #     RandomEventDrop(p=0.5, min_drop_rate=0., max_drop_rate=0.4)
    # ])
    # tsfmva = transforms.Compose([
    #     # transforms.RandomCrop(args.val_img_height, args.val_img_width, validate=True),
    #     ToTensor(),
    #     # transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    # ])
    tsfm = transforms.Compose(train_transform_list)

    # tsfm = transforms.Compose(train_transform_list)
    # tsfmva = transforms.Compose(test_val_trans)
    train_set, val_set, test_set = load_MVSEC('/data/tr/StereoData/MVSEC/',
                                              '/data/tr/StereoData/MVSEC_DVS/',
                                              scenario='indoor_flying', split='1',
                                              num_frames_per_depth_map=nfpdm, warmup_chunks=1, train_chunks=1,
                                              transform=tsfm, transformval=tsfm, normalize=False, learn_on='LIN')




    logger.info('=> {} training samples found in the training set'.format(len(train_set)))

    train_loader = DataLoader(dataset=train_set,
                              batch_size=batchsize,
                              shuffle=True,
                              drop_last=True,
                              pin_memory=True)


    # Test loader
    val_loader = DataLoader(dataset=val_set,
                                 batch_size=1,
                                 shuffle=False,
                                 drop_last=True,
                                 pin_memory=True)

    test_loader = DataLoader(dataset=test_set,
                                  batch_size=1,
                                  shuffle=False,
                                  drop_last=True,
                                  pin_memory=True)

    # train_data = dataloader_back.StereoDataset(data_dir=args.data_dir,
    #                                       dataset_name=args.dataset_name,
    #                                       mode='train' if args.mode != 'train_all' else 'train_all',
    #                                       load_pseudo_gt=args.load_pseudo_gt,
    #                                       transform=train_transform)

    # train_loader = DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=True,
    #                           num_workers=args.num_workers, pin_memory=True, drop_last=True)


    # Validation loader
    # val_transform_list = [transforms.RandomCrop(args.val_img_height, args.val_img_width, validate=True),
    #                       transforms.ToTensor(),
    #                       transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    #                      ]
    # val_transform = transforms.Compose(val_transform_list)
    # val_data = dataloader_back.StereoDataset(data_dir=args.data_dir,
    #                                     dataset_name=args.dataset_name,
    #                                     mode=args.mode,
    #                                     transform=val_transform)
    #
    # val_loader = DataLoader(dataset=val_data, batch_size=args.val_batch_size, shuffle=False,
    #                         num_workers=args.num_workers, pin_memory=True, drop_last=False)

    # Network
    aanet = nets.AANet(args.max_disp,
                       num_downsample=args.num_downsample,
                       feature_type=args.feature_type,
                       no_feature_mdconv=args.no_feature_mdconv,
                       feature_pyramid=args.feature_pyramid,
                       feature_pyramid_network=args.feature_pyramid_network,
                       feature_similarity=args.feature_similarity,
                       aggregation_type=args.aggregation_type,
                       num_scales=args.num_scales,
                       num_fusions=args.num_fusions,
                       num_stage_blocks=args.num_stage_blocks,
                       num_deform_blocks=args.num_deform_blocks,
                       no_intermediate_supervision=args.no_intermediate_supervision,
                       refinement_type=args.refinement_type,
                       mdconv_dilation=args.mdconv_dilation,
                       deformable_groups=args.deformable_groups).to(device)

    logger.info('%s' % aanet)

    if args.pretrained_aanet is not None:
        logger.info('=> Loading pretrained AANet: %s' % args.pretrained_aanet)
        # Enable training from a partially pretrained model
        utils.load_pretrained_net(aanet, args.pretrained_aanet, no_strict=(not args.strict))

    if torch.cuda.device_count() > 1:
        logger.info('=> Use %d GPUs' % torch.cuda.device_count())
        aanet = torch.nn.DataParallel(aanet)

    # Save parameters
    num_params = utils.count_parameters(aanet)
    logger.info('=> Number of trainable parameters: %d' % num_params)
    save_name = '%d_parameters' % num_params
    open(os.path.join(args.checkpoint_dir, save_name), 'a').close()

    # Optimizer
    # Learning rate for offset learning is set 0.1 times those of existing layers
    specific_params = list(filter(utils.filter_specific_params,
                                  aanet.named_parameters()))
    base_params = list(filter(utils.filter_base_params,
                              aanet.named_parameters()))

    specific_params = [kv[1] for kv in specific_params]  # kv is a tuple (key, value)
    base_params = [kv[1] for kv in base_params]

    specific_lr = args.learning_rate * 0.1
    params_group = [
        {'params': base_params, 'lr': args.learning_rate},
        {'params': specific_params, 'lr': specific_lr},
    ]

    optimizer = torch.optim.Adam(params_group, weight_decay=args.weight_decay)

    # Resume training
    if args.resume:
        # AANet
        start_epoch, start_iter, best_epe, best_epoch = utils.resume_latest_ckpt(
            args.checkpoint_dir, aanet, 'aanet')

        # Optimizer
        utils.resume_latest_ckpt(args.checkpoint_dir, optimizer, 'optimizer')
    else:
        start_epoch = 0
        start_iter = 0
        best_epe = None
        best_epoch = None

    # LR scheduler
    if args.lr_scheduler_type is not None:
        last_epoch = start_epoch if args.resume else start_epoch - 1
        if args.lr_scheduler_type == 'MultiStepLR':
            milestones = [int(step) for step in args.milestones.split(',')]
            lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                                milestones=milestones,
                                                                gamma=args.lr_decay_gamma,
                                                                last_epoch=last_epoch)
        else:
            raise NotImplementedError

    train_model = model.Model(args, logger, optimizer, aanet, device, start_iter, start_epoch,
                              best_epe=best_epe, best_epoch=best_epoch)

    logger.info('=> Start training...')

    if args.evaluate_only:
        assert args.val_batch_size == 1
        train_model.validate(val_loader)
    else:
        for _ in range(start_epoch, args.max_epoch):
            if not args.evaluate_only:
                train_model.train(train_loader)
            if not args.no_validate:
                train_model.validate(val_loader)
            if args.lr_scheduler_type is not None:
                lr_scheduler.step()

        logger.info('=> End training\n\n')


if __name__ == '__main__':
    main()

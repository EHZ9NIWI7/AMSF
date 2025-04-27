import logging
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from GZSSAR import dataset
from GZSSAR.fusion_strategy import get_fusion_module
from GZSSAR.model import MLP, Decoder, Encoder, LogisticRegression


class Initializer:
    def __init__(self, args):
        self.args = args
        
        self.data_dir, self.log_dir, self.result_dir = args.data_dir, args.log_dir, args.result_dir
        self.gpu = args.gpu
        
        self.skeleton_feature_extractor = args.skeleton_model
        self.text_feature_extractor = args.language_model
        self.fusion_strategy = args.fusion_strategy
        
        self.set_temp = args.temp
        self.set_thresh = args.thresh
        
        self.text_type = args.text_type
        self.split_type = args.split_type
        
        self.load_weight = args.load_weight
        self.mode = args.mode
        
        self.init_logging()
        self.init_environment()
        self.init_loss_function()
        
    def init_logging(self):
        log_dir = f'{self.log_dir}/{self.split_type}/{self.text_type.replace("_", "-")}'
        log_dir += f'/{self.skeleton_feature_extractor}-{self.text_feature_extractor.split("/")[-1]}'
        if self.set_thresh:
            log_dir += f'-th{self.set_thresh}'
        if self.set_temp:
            log_dir += f'-t{self.set_temp}'
            
        os.makedirs(log_dir, exist_ok=True)
        log_txt = f"{log_dir}/{self.fusion_strategy['module_name']}.txt"
        
        log_format = '[ %(asctime)s ] %(message)s'
        
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)

        fh = logging.FileHandler(log_txt, mode='w', encoding='UTF-8')
        fh.setLevel(logging.INFO)
        fh.setFormatter(logging.Formatter(log_format))
        logger.addHandler(fh)

        sh = logging.StreamHandler()
        sh.setLevel(logging.INFO)
        sh.setFormatter(logging.Formatter(log_format))
        logger.addHandler(sh)

    def init_environment(self):
        seed = 5
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.set_num_threads(2)
        np.random.seed(seed)
        self.device = torch.device(f'cuda:{self.gpu}')
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True

    def init_data(self):
        # Load Skeleton Features
        self.skeleton_data_dir = {
            'train': f'{self.data_dir}/sk_feats/{self.split_type}/{self.skeleton_feature_extractor}_{self.split_size}_{self.split_type}',
            'val': f'{self.data_dir}/sk_feats/{self.split_type}/{self.skeleton_feature_extractor}_val_{self.split_size}_{self.split_type}',
        }
        feeders, self.sk_emb_dim = dataset.create(self.skeleton_data_dir)
        self.feeders = feeders
        self.train_loader = DataLoader(feeders['train'], batch_size=64, shuffle=True, num_workers=0, pin_memory=True)
        self.zsl_loader = DataLoader(feeders['zsl'], batch_size=64, shuffle=False, num_workers=0, pin_memory=True)
        self.gzsl_loader = DataLoader(feeders['gzsl'], batch_size=64, shuffle=False, num_workers=0, pin_memory=True)
        self.val_loader = DataLoader(feeders['val'], batch_size=64, shuffle=True, num_workers=0, pin_memory=True)
        self.val_seen_loader = DataLoader(feeders['val_seen'], batch_size=64, shuffle=False, num_workers=0, pin_memory=True)
        self.val_unseen_loader = DataLoader(feeders['val_unseen'], batch_size=64, shuffle=False, num_workers=0, pin_memory=True)

        # Load Text Features
        self.text_data_dir = f'{self.data_dir}/text_feats/{self.text_feature_extractor}'

        self.text_feat = torch.stack(
            [torch.from_numpy(np.load(f'{self.text_data_dir}/{t}_{self.num_class}.npy')) for t in self.text_type.split('_')],
            dim=1
        )
        
        logging.info(f'Load training skeleton features from: {self.skeleton_data_dir["train"]}')
        logging.info(f'Load evaluation skeleton features from: {self.skeleton_data_dir["val"]}')
        logging.info(f'Load semantic features from: {self.text_data_dir}')
        logging.info('')
        logging.info(f'Training samples: (training: {len(feeders["train"])}, ZSL-eval: {len(feeders["zsl"])}, GZSL-eval: {len(feeders["gzsl"])}.)')
        logging.info(f'Validation samples: (val: {len(feeders["val"])}, val-seen: {len(feeders["val_seen"])}, val-unseen: {len(feeders["val_unseen"])}.)')
    
        logging.info('============================================================')

    def init_phase(self, phase):
        self.phase = phase

        # Init Split
        idx_path = {
            'train': {
                'seen': f'{self.data_dir}/splits/{self.split_type}s{str(self.num_class - self.split_size)}.npy',
                'unseen': f'{self.data_dir}/splits/{self.split_type}u{str(self.split_size)}.npy',
            },
            'val': {
                'seen': f'{self.data_dir}/splits/{self.split_type}s{str(self.num_class - 2 * self.split_size)}_0.npy',
                'unseen': f'{self.data_dir}/splits/{self.split_type}v{str(self.split_size)}_0.npy',
            }
        }
        seen_idx_path = idx_path[self.phase]['seen']
        unseen_idx_path = idx_path[self.phase]['unseen']
        self.seen_classes = sorted(list(np.load(seen_idx_path)))
        self.unseen_classes = sorted(list(np.load(unseen_idx_path)))
        
        # logging.info(f'Using {phase} data ({len(self.seen_classes)}-seen & {len(self.unseen_classes)}-unseen)')

        # Init Text Embeddings
        self.unseen_text_emb = self.text_feat[self.unseen_classes, :]
        self.seen_text_emb = self.text_feat[self.seen_classes, :]
        
        # Init Save Dir
        save_dir = {
            'train': f'{self.result_dir}/{self.split_type}/{self.split_size}/train/{self.text_feature_extractor.replace("/", "-")}/{self.text_type}/{self.fusion_strategy["module_name"]}',
            'val': f'{self.result_dir}/{self.split_type}/{self.split_size}/val/{self.text_feature_extractor.replace("/", "-")}/{self.text_type}/{self.fusion_strategy["module_name"]}',
        }
        self.save_dir = save_dir[self.phase]
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        
        # Init Path
        self.res_path = {
            'train': {
                'gzsl': self.skeleton_data_dir['train'] + '/gtest_out.npy',
                'val_seen': self.skeleton_data_dir['val'] + '/val_out.npy',
                'val_unseen': self.skeleton_data_dir['val'] + '/ztest_out.npy',
            },
            'zs': {
                'gzsl': save_dir['train'] + f'/AMSF_{str(self.split_size)}_{self.split_type}_gzsl_zs.npy',
                'val_seen': save_dir['val'] + f'/AMSF_{str(self.split_size)}_{self.split_type}_seen_zs.npy',
                'val_unseen': save_dir['val'] + f'/AMSF_{str(self.split_size)}_{self.split_type}_unseen_zs.npy',
            }
        }

        self.gate_path = save_dir['train'] + '/gating_model.pkl'
        self.gzsl_label_path = self.skeleton_data_dir['train'] + '/g_label.npy'

        # logging.info('----------------------------------------')
        
    def init_model(self):
        self.num_cycles = self.args.num_cycles if self.args.num_cycles else 10
        self.cycle_length = self.args.num_epoch_per_cycle if self.args.num_epoch_per_cycle != 0 else \
                            (1700 if self.num_class == 60 else 1900)
        self.latent_size = self.args.latent_size if self.args.latent_size != 0 else \
                            (100 if self.num_class == 60 else 200)
        
        self.fusion_strategy['channel'] = self.text_feat.size(1)
        self.fusion_strategy['in_dim'] = self.text_feat.size(-1)
        text_emb_dim = self.fusion_strategy['out_dim'] = self.text_feat.size(-1) * self.fusion_strategy['ratio']
        
        self.fusion_module = get_fusion_module(**self.fusion_strategy)
        self.fusion_module.to(self.device)
        self.sequence_encoder = Encoder([self.sk_emb_dim, self.latent_size]).to(self.device)
        self.sequence_decoder = Decoder([self.latent_size, self.sk_emb_dim]).to(self.device)
        self.text_encoder = Encoder([text_emb_dim, self.latent_size]).to(self.device)
        self.text_decoder = Decoder([self.latent_size, text_emb_dim]).to(self.device)
        self.cls = MLP([self.latent_size, self.split_size]).to(self.device)
        
        self.module_dict = {
            'se': self.sequence_encoder, 'sd': self.sequence_decoder, 
            'te': self.text_encoder, 'td': self.text_decoder, 
            'cls': self.cls, 'fm': self.fusion_module
        }
        
        if self.load_weight:
            for m in self.module_dict.keys():
                ckp = torch.load(f'{self.save_dir}/{m}.pth.tar')
                self.load_epoch = ckp['epoch']
                self.module_dict[m].load_state_dict(ckp['state_dict'])
            
    def init_optimizer(self):
        params = []
        module_list = [self.fusion_module, self.sequence_encoder, self.sequence_decoder, self.text_encoder, self.text_decoder]
        for module in module_list:
            params += list(module.parameters())
        self.optimizer = optim.Adam(params, lr = 0.0001)
        self.cls_optimizer = optim.Adam(self.cls.parameters(), lr = 0.001)
        
    def init_loss_function(self):
        self.ce_loss = nn.CrossEntropyLoss().to(self.device)
        self.mse_loss = nn.MSELoss().to(self.device)

    @staticmethod
    def kl_loss(mu, logvar):
        return 0.5*(torch.sum( - (mu**2) + 1 + logvar - torch.exp(logvar)))/mu.shape[0]

    @staticmethod
    def re_param(mu, logvar):
        sigma = torch.exp(0.5 * logvar).to(mu.device)
        eps = torch.FloatTensor(sigma.size()[0], 1).normal_(0, 1).expand(sigma.size()).to(mu.device) #.cuda()
        
        return eps * sigma + mu
    
    @staticmethod
    def temp_scale(ft, T):
        return np.array([np.exp(i)/np.sum(np.exp(i)) for i in (ft + 1e-12) / T])
    
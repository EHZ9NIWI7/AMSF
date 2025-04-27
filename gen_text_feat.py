import os
import pickle

import numpy as np
import torch

from text_extractor import CLIP_TE, CLIP_TK, L_CLIP_TE, L_CLIP_TK


class Text_Generator:
    def __init__(self, args):
        self.device = torch.device('cuda:{}'.format(args.gpu))
        torch.cuda.set_device(self.device)

        arch = args.language_model
        self.text_type = args.text_type.split('_')
        self.num_classes = [60, 120, 51]
        self.ntu_to_pku = [34, 3, 2, 32, 21, 9, 39, 0, 4, 1,
                           42, 55, 22, 57, 25, 54, 26, 50, 23, 27,
                           52, 5, 28, 53, 30, 49, 51, 19, 120,10, 
                           33, 37, 7, 33, 20, 18, 14, 121, 31, 12,
                           6, 45, 44, 43, 46, 29, 48, 13, 17, 36, 11]
        
        self.sem_dir = f'{args.data_dir}/sem_info'
        self.save_dir = f'{args.data_dir}/text_feats/{arch}'
        os.makedirs(self.save_dir, exist_ok=True)

        if arch == 'longclip-L':
            self.tokenize = L_CLIP_TK
            self.text_encoder = L_CLIP_TE(arch)
        else:
            self.tokenize = CLIP_TK
            self.text_encoder = CLIP_TE(arch)

        self.text_encoder.to(self.device)
    def gen_token(self, text_type, nc):
        with open(f'{self.sem_dir}/{text_type}_{nc}.pkl', 'rb') as f:
            sem_info = pickle.load(f, encoding='latin1')
        token = torch.cat([self.tokenize(i) for i in sem_info])

        return token
    def start(self):
        print('Generating Text Feature ...')
        self.text_encoder.eval()
        with torch.no_grad():
            for text_type in self.text_type:
                for nc in [51, 60, 120]:
                    print(f'Processing {text_type}_{nc} ...')
                    token = self.gen_token(text_type, nc)
                    save_path = self.save_dir + f'/{text_type}_{nc}'
                    text_inp = token.to(self.device)
                    tf = self.text_encoder(text_inp)
                    np.save(save_path, tf.cpu().numpy())
        
        torch.cuda.empty_cache()
        print('Finish generating!')

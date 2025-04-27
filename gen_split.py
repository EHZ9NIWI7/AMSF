import os
import random

import numpy as np
import torch


class Split_Generator:
    def __init__(self, args):
        self.split_size = {60: [5, 12], 120: [10, 24], 51: [4, 8]}
        self.split_type = args.split_type
        self.text_type = args.text_type.split('_')
        self.tf_dir = f'{args.data_dir}/text_feats/{args.language_model}'
        self.save_dir = f'{args.data_dir}/splits'
        os.makedirs(self.save_dir, exist_ok=True)
    
    @staticmethod
    def norm(a):
        return a.div(a.norm(p=2, dim=-1).unsqueeze(-1).expand_as(a) + 1e-5)

    def get_sim_mat(self, num_class, mode='add'):
        def gen_sm(t):
            sm = torch.zeros([num_class, num_class], dtype=torch.float64)
            for i in range(num_class):
                x = self.norm(t[i])
                for j in range(num_class):
                    if j > i:
                        y = self.norm(t[j])
                        sim = torch.dot(x, y)
                        sm[i, j] = sim

            return sm

        if mode == 'add':
            mat_list = []
            for tt in self.text_type:
                tf = torch.from_numpy(np.load(f'{self.tf_dir}/{tt}_{num_class}.npy'))
                sm = gen_sm(tf) 
                mat_list.append(sm)
            sim_mat = sum(mat_list)
        elif mode == 'concat':
            tf = torch.stack(
                [torch.from_numpy(np.load(f'{self.tf_dir}/{tt}_{num_class}.npy')) for tt in self.text_type],
                dim=1
            )
            tf = tf.reshape(tf.size(0), -1)
            sim_mat = gen_sm(tf)
        
        return sim_mat
        
    @staticmethod
    def easy(ss, sm, cl):
        nc = sm.size(0)
        unseen, related = [], []

        for i in range(nc):
            if i not in cl:
                sm[i, :] = 0
                sm[:, i] = 0

        padding_sm = sm + sm.t()
        sim_value_per_class = padding_sm.sum(0)

        for _ in range(nc^2):
            a, b = int(sm.argmax()) // nc, int(sm.argmax()) % nc
            if sim_value_per_class[a] > sim_value_per_class[b]:
                s, r = a, b
            else:
                s, r = b, a
            
            if (s not in unseen + related) and (r not in unseen):
                unseen.append(s)
                related.append(r)
            
            sm[a, b] = 0
                
            if len(unseen) == ss:
                break
        
        return unseen

    @staticmethod
    def hard(ss, sm, cl):
        nc = sm.size(0)
        unseen = []

        for i in range(nc):
            if i not in cl:
                sm[i, :] = 0
                sm[:, i] = 0

        padding_sm = sm + sm.t()

        rank = padding_sm.topk(5, dim=0)[0].sum(0).sort()
        for i, idx in enumerate(rank[1]):
            if rank[0][i] == 0:
                continue
            else:
                unseen.append(int(idx))
            
            if len(unseen) == ss:
                break

        return unseen

    @staticmethod
    def rand(ss, sm, cl):
        return random.sample(cl, ss)
        
    def generate(self, num_class, split_size, class_list):
        sim_mat = self.get_sim_mat(num_class)
        split_type = 'easy' if self.split_type == 'e' else 'hard' if self.split_type == 'h' else 'rand'
        sp_func = getattr(self, split_type)
        unseen_classes = sp_func(split_size, sim_mat, class_list)
        seen_classes = [i for i in class_list if i not in unseen_classes]

        for i in unseen_classes:
            if i not in class_list:
                raise ValueError('Wrong Unseen Classes.')

        return seen_classes, unseen_classes

    def start(self):
        for num_class in self.split_size.keys():
            for split_size in self.split_size[num_class]:
                s, u = self.generate(num_class, split_size, [i for i in range(num_class)])
                vs, vu = self.generate(num_class, split_size, s)
                s, u, vs, vu = sorted(s), sorted(u), sorted(vs), sorted(vu)
                print(f'==========NTU-{num_class}u{split_size}==========')
                # print(f's: {[i + 1 for i in s]}')
                # print('')
                print(f'{len(u)}-unseen: {[i + 1 for i in u]}')
                print('')
                # print(f'vs: {[i + 1 for i in vs]}')
                # print('')
                print(f'{len(vu)}-val_unseen: {[i + 1 for i in vu]}')
                print('')
                
                np.save(f'{self.save_dir}/{self.split_type}s{len(s)}.npy', (np.array(s)))
                np.save(f'{self.save_dir}/{self.split_type}u{len(u)}.npy', (np.array(u)))
                np.save(f'{self.save_dir}/{self.split_type}s{len(vs)}_0.npy', (np.array(vs)))
                np.save(f'{self.save_dir}/{self.split_type}v{len(vu)}_0.npy', (np.array(vu)))

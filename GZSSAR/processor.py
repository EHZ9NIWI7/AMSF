import logging
import pickle as pkl

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression

from GZSSAR.initializer import Initializer
from GZSSAR.utils import AverageMeter


class Processor(Initializer):
    def train_vae(self, cycle_num):# 0-10, 1700
        s_epoch = (cycle_num) * (self.cycle_length) # 0, 1700, 3400,...
        e_epoch = (cycle_num + 1) * (self.cycle_length) # 1700, 3400, 5100, ... 
        cr_fact_epoch = 1400 if self.cycle_length == 1700 else 1500
        loader = self.train_loader if self.phase == 'train' else self.val_loader
        
        for epoch in range(s_epoch, e_epoch):
            losses = AverageMeter()
            ce_loss_vals = []
            
            self.fusion_module.train()
            self.sequence_encoder.train()
            self.sequence_decoder.train()
            self.text_encoder.train()
            self.text_decoder.train()

            # hyper params
            k_fact = max((0.1 * (epoch - (s_epoch + 1000)) / 3000), 0)
            cr_fact = 1 * (epoch > (s_epoch + cr_fact_epoch))
            k_fact2 = max((0.1 * (epoch - (s_epoch + cr_fact_epoch)) / 3000), 0) * (cycle_num > 1)

            (inputs, target) = next(iter(loader))
            s = inputs.to(self.device)
            s = s / s.norm(dim=-1, keepdim=True)
            t = self.text_feat[target].float().to(self.device)
            t = self.fusion_module(t)
            t = t / t.norm(dim=-1, keepdim=True)

            smu, slv = self.sequence_encoder(s)
            sz = self.re_param(smu, slv)
            sout = self.sequence_decoder(sz)

            tmu, tlv = self.text_encoder(t)
            tz = self.re_param(tmu, tlv)
            tout = self.text_decoder(tz)

            sfromt = self.sequence_decoder(tz)
            tfroms = self.text_decoder(sz)

            s_recons = self.mse_loss(s, sout)
            t_recons = self.mse_loss(t, tout)
            s_kld = self.kl_loss(smu, slv).to(self.device) 
            t_kld = self.kl_loss(tmu, tlv).to(self.device)
            t_crecons = self.mse_loss(t, tfroms)
            s_crecons = self.mse_loss(s, sfromt)

            loss = s_recons + t_recons
            loss -= k_fact*(s_kld) + k_fact2*(t_kld)
            loss += cr_fact*(s_crecons) + cr_fact*(t_crecons)

            # backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            losses.update(loss.item(), inputs.size(0))
            ce_loss_vals.append(loss.cpu().detach().numpy())
        #     if epoch % 500 == 0:
        #         logging.info('')
        #         logging.info('Epoch-{:<3d}\t loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(epoch, loss=losses))
        #         logging.info('srecons {:.4f}\t trecons {:.4f}\t'.format(s_recons.item(), t_recons.item()))
        #         logging.info('skld {:.4f}\t tkld {:.4f}\t'.format(s_kld.item(), t_kld.item()))
        #         logging.info('screcons {:.4f}\t tcrecons {:.4f}\t'.format(s_crecons.item(), t_crecons.item()))
        
        # logging.info('')

    def train_classifier(self):
        # Generate Unseen Latent Embedding
        with torch.no_grad():
            self.fusion_module.eval()
            self.text_encoder.eval()
            ut = self.unseen_text_emb.float().to(self.device)
            ut = self.fusion_module(ut)
            ut = ut / ut.norm(dim=-1, keepdim=True).repeat([1, ut.size(-1)])
            x = ut.repeat([500, 1])
            y = torch.tensor(range(self.split_size)).to(self.device)
            y = y.repeat([500])
            mu, lv = self.text_encoder(x)
            gx = self.re_param(mu, lv)

        # Train Unseen Classifier
        for c_e in range(300):
            self.cls.train()
            out = self.cls(gx.detach())
            c_loss = self.ce_loss(out, y)
            self.cls_optimizer.zero_grad()
            c_loss.backward()
            self.cls_optimizer.step()
            c_acc = float(torch.sum(y == torch.argmax(out, -1))) / (self.split_size * 500)
            # if c_e % 300 == 0:
            #     logging.info(f'cls_loss : {c_loss.item() :.4f}, cls_acc: {c_acc :.2%}')
            #     logging.info('')

    def get_zs_res(self, loader):
        res = []

        with torch.no_grad():
            self.sequence_encoder.eval()
            self.cls.eval()
            for (x, _) in loader:
                x = x.to(self.device)
                x_mu, x_lv = self.sequence_encoder(x)
                out = self.cls(x_mu)
                res.append(F.softmax(out, -1))
        
        return np.array([j.cpu().numpy() for i in res for j in i])
    
    def zsl_eval(self):
        loader = self.zsl_loader if self.phase == 'train' else self.val_unseen_loader
        z_matrix = torch.zeros(len(self.unseen_classes), 2, dtype=torch.int32)
        us, pred_res = [], []

        with torch.no_grad():
            self.sequence_encoder.eval()
            self.cls.eval()
            for (x, y) in loader:
                x = x.to(self.device)
                mu, lv = self.sequence_encoder(x)
                out = self.cls(mu)
                pred = torch.argmax(out, -1)
                pred_res.append(pred)
                for i, t in enumerate(y):
                    us.append(out[i].cpu())
                    idx = self.unseen_classes.index(t)
                    z_matrix[idx, 1] += 1
                    if self.unseen_classes[pred[i]] == t:
                        z_matrix[idx, 0] += 1

        if self.args.acc_type == 'avg':
            acc_per_class = [float(z_matrix[i, 0] / z_matrix[i, 1]) for i in range(z_matrix.size(0))]
            acc = sum(acc_per_class) / len(acc_per_class)
        else:
            acc = float(sum(z_matrix[:, 0]) / sum(z_matrix[:, 1]))

        return acc, z_matrix
    
    def train_gate(self):
        self.init_phase('val')
        self.temp = self.set_temp if self.set_temp else 2
        
        unseen_train = np.load(self.res_path['train']['val_unseen'])
        seen_train = np.load(self.res_path['train']['val_seen'])
        unseen_zs = np.load(self.res_path['zs']['val_unseen'])
        seen_zs = np.load(self.res_path['zs']['val_seen'])

        unseen_train = self.temp_scale(unseen_train, self.temp)
        seen_train = self.temp_scale(seen_train, self.temp)

        unseen_train = np.sort(unseen_train, 1)[:,::-1][:,:self.split_size]
        seen_train = np.sort(seen_train, 1)[:,::-1][:,:self.split_size]
        unseen_zs = np.sort(unseen_zs, 1)[:,::-1][:,:self.split_size]
        seen_zs = np.sort(seen_zs, 1)[:,::-1][:,:self.split_size]

        # training samples
        gating_train_x = np.concatenate(
            [
                np.concatenate([unseen_zs, unseen_train], axis=1),
                np.concatenate([seen_zs, seen_train], axis=1)
            ],
            axis=0
        )
        gating_train_y = [0] * len(unseen_train) + [1] * len(seen_train)

        # shuffle
        shuffle_idx = np.arange(gating_train_x.shape[0])
        np.random.shuffle(shuffle_idx)
            
        # logistic regression
        model = LogisticRegression(
            random_state=0, C=1, solver='lbfgs', n_jobs=-1,
            multi_class='multinomial', verbose=0, max_iter=5000, #class_weight='balanced',
        ).fit(gating_train_x[shuffle_idx, :], np.array(gating_train_y)[shuffle_idx])

        prob = model.predict_proba(gating_train_x)

        self.gate = model
        self.thresh = self.set_thresh if self.set_thresh else \
                      (np.mean(prob[:len(unseen_train), 0]) + np.mean(prob[len(unseen_train):, 0])) / 2 * 100

        y = prob[:, 0] > (self.thresh / 100)

        s = np.array(gating_train_y)
        u = 1 - s
        r = ((1 - y) == s)
        gate_acc_s = (r * s).sum() / s.sum()
        gate_acc_u = (r * u).sum() / u.sum()
        gate_acc = r.sum() / len(r)

        with open(self.gate_path, 'wb') as f:
            pkl.dump(self.gate, f)
            f.close()
        
        logging.info(f'Temperature: {self.temp}, Threshold :{self.thresh / 100}.')
        logging.info(f'Gate training acc: {gate_acc :.2%} (Seen: {gate_acc_s :.2%}, Unseen: {gate_acc_u :.2%})')

    def gzsl_eval(self):
        self.init_phase('train')
        
        # feature
        test_unseen = np.load(self.res_path['zs']['gzsl'])
        test_seen = np.load(self.res_path['train']['gzsl'])


        ts = self.temp_scale(test_seen, self.temp)
        tu = test_unseen

        tu = np.sort(tu, 1)[:,::-1][:,:self.split_size]
        ts = np.sort(ts, 1)[:,::-1][:,:self.split_size]
        
        test_x = np.concatenate([tu, ts], 1)
        
        # label
        tars = np.load(self.gzsl_label_path)
        test_y = np.array([0 if i in self.unseen_classes else 1 for i in tars])
        
        prob_gate = self.gate.predict_proba(test_x)
        pred_test = 1 - (prob_gate[:, 0] > (self.thresh / 100))

        s = np.array(test_y)
        u = 1 - s
        r = (pred_test == s)
        gate_acc = r.sum() / len(r)
        gate_acc_s = (r * s).sum() / s.sum()
        gate_acc_u = (r * u).sum() / u.sum()
        
        logging.info(f'Gate evaluation Acc: {float(gate_acc) :.2%} (Seen: {float(gate_acc_s) :.2%}, Unseen: {float(gate_acc_u) :.2%})')

        us, pred_u = [], []
        s_matrix = torch.zeros(len(self.seen_classes), 2, dtype=torch.int32)
        u_matrix = torch.zeros(len(self.unseen_classes), 2, dtype=torch.int32)
        for i in range(len(tars)):
            if pred_test[i] == 1:
                pred = self.seen_classes[np.argmax(test_seen[i, self.seen_classes])]
            else:
                pred = self.unseen_classes[np.argmax(test_unseen[i, :])]
            
            if tars[i] in self.seen_classes:
                s_idx = self.seen_classes.index(tars[i])
                s_matrix[s_idx, -1] += 1
                if pred == tars[i]:
                    s_matrix[s_idx, 0] += 1
            else:
                u_idx = self.unseen_classes.index(tars[i])
                u_matrix[u_idx, -1] += 1
                if pred == tars[i]:
                    u_matrix[u_idx, 0] += 1
                
                us.append(torch.tensor(test_unseen[i, :]))
                pred_u.append(pred)
        
        if self.args.acc_type == 'avg':
            acc_per_sc = [float(s_matrix[i, 0] / s_matrix[i, 1]) if s_matrix[i, 1] != 0 else 0. for i in range(s_matrix.size(0))]
            acc_per_uc = [float(u_matrix[i, 0] / u_matrix[i, 1]) if u_matrix[i, 1] != 0 else 0. for i in range(u_matrix.size(0))]
            s_acc = sum(acc_per_sc) / len(acc_per_sc)
            u_acc = sum(acc_per_uc) / len(acc_per_uc)
        else:
            s_acc = float(sum(s_matrix[:, 0]) / sum(s_matrix[:, 1]))
            u_acc = float(sum(u_matrix[:, 0]) / sum(u_matrix[:, 1]))
            
        h_mean = 2 * s_acc * u_acc / (s_acc + u_acc) if not (s_acc == 0 and u_acc == 0) else 0.
        
        np.save(f'{self.save_dir}/sm.npy', np.array(s_matrix.numpy()))
        np.save(f'{self.save_dir}/um.npy', np.array(u_matrix.numpy()))

        logging.info(f'GZSL Acc: (Seen_Acc: {s_acc :.2%}, Unseen_Acc: {u_acc :.2%}, H_Mean: {h_mean :.2%})')

        return s_acc, u_acc, h_mean
    
    def save_model(self, num_cycle):
        for k in self.module_dict:
            torch.save(
                {'epoch': self.cycle_length * (num_cycle + 1),
                'state_dict': self.module_dict[k].state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'cls_optimizer': self.cls_optimizer.state_dict()},
                f'{self.save_dir}/{k}.pth.tar'
            )

    def train_eval_loop(self):
        zsl_accuracy, s_acc, u_acc, h_mean = 0, 0, 0, 0

        # train/val phase
        phases = ['train'] if self.mode == 'zsl' else \
                 ['val'] if self.mode == 'gzsl' else \
                 [] if self.mode == 'gate' else \
                 ['train', 'val']
        
        for phase in phases:
            logging.info(f'Phase: {phase}')
            logging.info('----------------------------------------')
            self.init_phase(phase)
            self.init_model()
            self.init_optimizer()

            start_epoch = 0 if not self.load_weight else self.load_epoch + 1
            best = 0

            for num_cycle in range(start_epoch // self.cycle_length, self.num_cycles):
                self.train_vae(num_cycle)
                self.train_classifier()
                zsl_acc, zm = self.zsl_eval()
                if zsl_acc > best:
                    best = zsl_acc
                    best_zm = zm
                    self.save_model(num_cycle)
                    logging.info(f'ZSL_Acc increased to {best :.2%} on cycle {num_cycle}.')
                    
                    if phase == 'train':
                        gzsl_zs_res = self.get_zs_res(self.gzsl_loader)
                        np.save(self.res_path['zs']['gzsl'], gzsl_zs_res)
                    else:
                        val_seen_zs_res = self.get_zs_res(self.val_seen_loader)
                        val_unseen_zs_res = self.get_zs_res(self.val_unseen_loader)
                        np.save(self.res_path['zs']['val_seen'], val_seen_zs_res)
                        np.save(self.res_path['zs']['val_unseen'], val_unseen_zs_res)
            
            if phase == 'train':
                zsl_accuracy = best

            logging.info('============================================================')

        if self.mode != 'zsl':
            logging.info('Phase: gzsl eval')
            logging.info('----------------------------------------')
            self.train_gate()
            s_acc, u_acc, h_mean = self.gzsl_eval()
            
        return zsl_accuracy, s_acc, u_acc, h_mean

    def start(self):
        results = {'zsl': [], 'gzsl': []}

        for nc in [51, 60, 120]:
            self.num_class = nc
            split = [5, 12] if nc == 60 else [10, 24] if nc == 120 else [4, 8]
            for s in split:
                self.split_size = s
                logging.info(f'==================  NTU{self.num_class}-u{self.split_size} / {self.text_type}  ===================')
                self.init_data()

                zsl_acc, s_acc, u_acc, h_mean = self.train_eval_loop()
                results['zsl'].append(zsl_acc)
                results['gzsl'].append(s_acc)
                results['gzsl'].append(u_acc)
                results['gzsl'].append(h_mean)
                
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                torch.cuda.init()
            
                logging.info('')
        
        logging.info('=======================    Total    ========================')
        if self.mode == 'zsl' or self.mode == '':
            zsl_accs = [round(i * 100, 2) for i in results["zsl"]]
            logging.info(f'ZSL Acc: {zsl_accs} (avg: {round(sum(zsl_accs) / len(zsl_accs), 2)})')
        if self.mode != 'zsl':
            logging.info(f'GZSL Acc: {[round(j * 100, 2) for j in results["gzsl"]]}')
        logging.info('')
            # h_idx = [2, 5, 8, 11, 14, 17]
            # logging.info(f'GZSL-avg: {round(sum([results["gzsl"][i] for i in h_idx]) / len(h_idx) * 100, 2)}')

        
        
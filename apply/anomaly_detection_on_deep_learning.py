import os
import argparse
import logging
import numpy as np
import time
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score

import pandas as pd

from apply.anomaly_detaction_base import AnomalyDetection
from tools.data_loader import get_loader_segment
from tools.memto.model.Transformer import TransformerVar
from tools.anomaly_transformer.model.AnomalyTransformer import AnomalyTransformer
from tools.memto.model.loss_functions import *
from tools.utils import *


class AnomalyDetectionOnDeepLearning(AnomalyDetection):
    def __init__(self, config, file_name):
        self.DEFAULTS = {}
        self.file_name = file_name
        self.__dict__.update(self.DEFAULTS, **config)

        self.train_loader = get_loader_segment(self.data_path, batch_size=self.batch_size, win_size=self.win_size,
                                               mode='train', dataset=self.dataset, step=self.step, file_name=self.file_name)
        self.vali_loader = get_loader_segment(self.data_path, batch_size=self.batch_size, win_size=self.win_size,
                                              mode='val', dataset=self.dataset, step=self.step, file_name=self.file_name)
        self.k_loader = get_loader_segment(self.data_path, batch_size=self.batch_size, win_size=self.win_size,
                                           mode='k_loader', dataset=self.dataset, step=self.step, file_name=self.file_name)
        self.test_loader = get_loader_segment(self.data_path, batch_size=self.batch_size, win_size=self.win_size,
                                              mode='test', dataset=self.dataset, step=self.step, file_name=self.file_name)

        self.thre_loader = self.vali_loader

        self.criterion = nn.MSELoss()

        return AnomalyDetection.__init__(self)

    def build_model(self):
        pass

    def vali(self, vali_loader):
        pass

    def train(self, training_type):
        pass

    def test(self):
        pass


class AnomalyDetectionOnMemto(AnomalyDetectionOnDeepLearning):
    def __init__(self, config, file_name):
        AnomalyDetectionOnDeepLearning.__init__(self, config, file_name)

        if self.memory_initial == "False":

            self.memory_initial = False
        else:
            self.memory_initial = True

        self.memory_init_embedding = None

        self.build_model(memory_init_embedding=self.memory_init_embedding)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.entropy_loss = EntropyLoss()

        self.logger = logging.getLogger()
        self.logger.setLevel(logging.INFO)

        formatter = logging.Formatter('%(asctime)s - %(message)s')
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        self.logger.addHandler(stream_handler)

    def build_model(self, memory_init_embedding):
        self.model = TransformerVar(win_size=self.win_size, enc_in=self.input_c, c_out=self.output_c, e_layers=3,
                                    d_model=self.d_model, n_memory=self.n_memory, device=self.device,
                                    memory_initial=self.memory_initial, memory_init_embedding=memory_init_embedding,
                                    phase_type=self.phase_type, dataset_name=self.dataset)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        if torch.cuda.is_available():
            self.model = torch.nn.DataParallel(self.model, device_ids=[0, 1, 2, 3], output_device=0).to(self.device)

    def get_memory_initial_embedding(self, training_type='second_train'):

        self.model.load_state_dict(
            torch.load(os.path.join(str(self.model_save_path),
                                    '{}_{}_{}'.format(str(self.dataset), self.file_name, 'checkpoint_first_train.pth')))
        )
        self.model.eval()

        for i, input_data in enumerate(self.k_loader):

            input = input_data.float().to(self.device)
            if i == 0:
                output = self.model(input)['queries']
            else:
                output = torch.cat([output, self.model(input)['queries']], dim=0)

        self.memory_init_embedding = k_means_clustering(x=output, n_mem=self.n_memory,
                                                        d_model=self.d_model, device='cpu')

        self.memory_initial = False

        self.build_model(memory_init_embedding=self.memory_init_embedding.detach())

        memory_item_embedding = self.train(training_type=training_type)
        # self.test()

        memory_item_embedding = memory_item_embedding[:int(self.n_memory), :]

        item_folder_path = "memory_item"
        if not os.path.exists(item_folder_path):
            os.makedirs(item_folder_path)

        item_path = os.path.join(item_folder_path, str(self.dataset) + '_memory_item.pth')

        torch.save(memory_item_embedding, item_path)

    def vali(self, vali_loader):
        self.model.eval()

        valid_loss_list = [];
        valid_re_loss_list = [];
        valid_entropy_loss_list = []

        for i, input_data in enumerate(vali_loader):
            input = input_data.float().to(self.device)
            output_dict = self.model(input)
            output, queries, mem_items, attn = output_dict['out'], output_dict['queries'], output_dict['mem'],\
                                               output_dict['attn']

            rec_loss = self.criterion(output, input)
            entropy_loss = self.entropy_loss(attn)
            loss = rec_loss + self.lambd * entropy_loss

            valid_re_loss_list.append(rec_loss.detach().cpu().numpy())
            valid_entropy_loss_list.append(entropy_loss.detach().cpu().numpy())
            valid_loss_list.append(loss.detach().cpu().numpy())

        return np.average(valid_loss_list), np.average(valid_re_loss_list), np.average(valid_entropy_loss_list)

    def train(self, training_type):
        print("======================TRAIN MODE======================")

        time_now = time.time()
        path = self.model_save_path
        if not os.path.exists(path):
            os.makedirs(path)
        early_stopping = OneEarlyStopping(patience=10, verbose=True, dataset_name=self.dataset, type=training_type)
        train_steps = len(self.train_loader)

        from tqdm import tqdm
        for epoch in tqdm(range(self.num_epochs)):
            iter_count = 0
            loss_list = []
            rec_loss_list = [];
            entropy_loss_list = []

            epoch_time = time.time()
            self.model.train()
            for i, input_data in enumerate(self.train_loader):

                self.optimizer.zero_grad()
                iter_count += 1
                input = input_data.float().to(self.device)
                output_dict = self.model(input_data)

                output, memory_item_embedding, queries, mem_items, attn = output_dict['out'], output_dict[
                    'memory_item_embedding'], output_dict['queries'], output_dict["mem"], output_dict['attn']

                rec_loss = self.criterion(output, input)
                entropy_loss = self.entropy_loss(attn)
                loss = rec_loss + self.lambd * entropy_loss

                loss_list.append(loss.detach().cpu().numpy())
                entropy_loss_list.append(entropy_loss.detach().cpu().numpy())
                rec_loss_list.append(rec_loss.detach().cpu().numpy())

                if (i + 1) % 100 == 0:
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.num_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()
                try:
                    loss.mean().backward()

                except:
                    import pdb;
                    pdb.set_trace()
                self.optimizer.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))

            train_loss = np.average(loss_list)
            train_entropy_loss = np.average(entropy_loss_list)
            train_rec_loss = np.average(rec_loss_list)
            valid_loss, valid_re_loss_list, valid_entropy_loss_list = self.vali(self.vali_loader)

            print(
                "Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f}".format(
                    epoch + 1, train_steps, train_loss, valid_loss))
            print(
                "Epoch: {0}, Steps: {1} | VALID reconstruction Loss: {3:.7f} Entropy loss Loss: {2:.7f}  ".format(
                    epoch + 1, train_steps, valid_re_loss_list, valid_entropy_loss_list))
            print(
                "Epoch: {0}, Steps: {1} | TRAIN reconstruction Loss: {3:.7f} Entropy loss Loss: {2:.7f}  ".format(
                    epoch + 1, train_steps, train_rec_loss, train_entropy_loss))

            early_stopping(valid_loss, self.model, os.path.join(path, '{}_{}'.format(self.dataset, self.file_name)))
            if early_stopping.early_stop:
                print("Early stopping")
                break
            adjust_learning_rate(self.optimizer, epoch + 1, self.lr)

        return memory_item_embedding

    def test(self):
        self.model.load_state_dict(
            torch.load(os.path.join(str(self.model_save_path),
                                    '{}_{}_{}'.format(str(self.dataset), self.file_name, 'checkpoint_second_train.pth')))
        )
        self.model.eval()

        print("======================TEST MODE======================")

        criterion = nn.MSELoss(reduce=False)
        gathering_loss = GatheringLoss(reduce=False)
        temperature = self.temperature

        train_attens_energy = []
        for i, input_data in enumerate(self.train_loader):
            input = input_data.float().to(self.device)
            output_dict = self.model(input_data)
            output, queries, mem_items = output_dict['out'], output_dict['queries'], output_dict['mem']

            rec_loss = torch.mean(criterion(input, output), dim=-1)
            latent_score = torch.softmax(gathering_loss(queries, mem_items) / temperature, dim=-1)
            loss = latent_score * rec_loss

            cri = loss.detach().cpu().numpy()
            train_attens_energy.append(cri)

        train_attens_energy = np.concatenate(train_attens_energy, axis=0).reshape(-1)
        train_energy = np.array(train_attens_energy)

        valid_attens_energy = []
        for i, input_data in enumerate(self.thre_loader):
            input = input_data.float().to(self.device)
            output_dict = self.model(input)
            output, queries, mem_items = output_dict['out'], output_dict['queries'], output_dict['mem']

            rec_loss = torch.mean(criterion(input, output), dim=-1)
            latent_score = torch.softmax(gathering_loss(queries, mem_items) / temperature, dim=-1)
            loss = latent_score * rec_loss

            cri = loss.detach().cpu().numpy()
            valid_attens_energy.append(cri)

        valid_attens_energy = np.concatenate(valid_attens_energy, axis=0).reshape(-1)
        valid_energy = np.array(valid_attens_energy)

        combined_energy = np.concatenate([train_energy, valid_energy], axis=0)

        thresh = np.percentile(combined_energy, 100 - self.anormly_ratio)
        print("Threshold :", thresh)

        distance_with_q = []
        reconstructed_output = []
        original_output = []
        rec_loss_list = []

        test_attens_energy = []
        for i, input_data in enumerate(self.test_loader):
            input = input_data.float().to(self.device)
            output_dict = self.model(input)
            output, queries, mem_items = output_dict['out'], output_dict['queries'], output_dict['mem']

            rec_loss = torch.mean(criterion(input, output), dim=-1)
            latent_score = torch.softmax(gathering_loss(queries, mem_items) / temperature, dim=-1)
            loss = latent_score * rec_loss

            cri = loss.detach().cpu().numpy()
            test_attens_energy.append(cri)

            d_q = gathering_loss(queries, mem_items) * rec_loss
            distance_with_q.append(d_q.detach().cpu().numpy())
            distance_with_q.append(gathering_loss(queries, mem_items).detach().cpu().numpy())

            reconstructed_output.append(output.detach().cpu().numpy())
            original_output.append(input.detach().cpu().numpy())
            rec_loss_list.append(rec_loss.detach().cpu().numpy())

        test_attens_energy = np.concatenate(test_attens_energy, axis=0).reshape(-1)
        test_energy = np.array(test_attens_energy)

        distance_with_q = np.concatenate(distance_with_q, axis=0).reshape(-1)

        pred = (test_energy > thresh).astype(int)

        pred = np.array(pred)

        return pred


class AnomalyDetectionOnAnomalyTransformer(AnomalyDetectionOnDeepLearning):
    def __init__(self, config, file_name):
        AnomalyDetectionOnDeepLearning.__init__(self, config, file_name)

        self.build_model()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        return

    def build_model(self):
        self.model = AnomalyTransformer(win_size=self.win_size, enc_in=self.input_c, c_out=self.output_c, e_layers=3)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        if torch.cuda.is_available():
            self.model.cuda()

        return

    def vali(self, vali_loader):
        self.model.eval()

        loss_1 = []
        loss_2 = []
        for i, input_data in enumerate(vali_loader):
            input = input_data.float().to(self.device)
            output, series, prior, _ = self.model(input)
            series_loss = 0.0
            prior_loss = 0.0
            for u in range(len(prior)):
                series_loss += (torch.mean(my_kl_loss(series[u], (
                        prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                               self.win_size)).detach())) + torch.mean(
                    my_kl_loss(
                        (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                self.win_size)).detach(),
                        series[u])))
                prior_loss += (torch.mean(
                    my_kl_loss((prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                       self.win_size)),
                               series[u].detach())) + torch.mean(
                    my_kl_loss(series[u].detach(),
                               (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                       self.win_size)))))
            series_loss = series_loss / len(prior)
            prior_loss = prior_loss / len(prior)

            rec_loss = self.criterion(output, input)
            loss_1.append((rec_loss - self.k * series_loss).item())
            loss_2.append((rec_loss + self.k * prior_loss).item())

        return np.average(loss_1), np.average(loss_2)

    def train(self):
        print("======================TRAIN MODE======================")

        time_now = time.time()
        path = self.model_save_path
        if not os.path.exists(path):
            os.makedirs(path)
        early_stopping = EarlyStopping(patience=3, verbose=True, dataset_name=self.dataset)
        train_steps = len(self.train_loader)

        for epoch in range(self.num_epochs):
            iter_count = 0
            loss1_list = []

            epoch_time = time.time()
            self.model.train()
            for i, input_data in enumerate(self.train_loader):

                self.optimizer.zero_grad()
                iter_count += 1
                input = input_data.float().to(self.device)

                output, series, prior, _ = self.model(input)

                # calculate Association discrepancy
                series_loss = 0.0
                prior_loss = 0.0
                for u in range(len(prior)):
                    series_loss += (torch.mean(my_kl_loss(series[u], (
                            prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                   self.win_size)).detach())) + torch.mean(
                        my_kl_loss((prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                           self.win_size)).detach(),
                                   series[u])))
                    prior_loss += (torch.mean(my_kl_loss(
                        (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                self.win_size)),
                        series[u].detach())) + torch.mean(
                        my_kl_loss(series[u].detach(), (
                                prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                       self.win_size)))))
                series_loss = series_loss / len(prior)
                prior_loss = prior_loss / len(prior)

                rec_loss = self.criterion(output, input)

                loss1_list.append((rec_loss - self.k * series_loss).item())
                loss1 = rec_loss - self.k * series_loss
                loss2 = rec_loss + self.k * prior_loss

                if (i + 1) % 100 == 0:
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.num_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                # Minimax strategy
                loss1.backward(retain_graph=True)
                loss2.backward()
                self.optimizer.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(loss1_list)

            vali_loss1, vali_loss2 = self.vali(self.test_loader)

            print(
                "Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} ".format(
                    epoch + 1, train_steps, train_loss, vali_loss1))
            early_stopping(vali_loss1, vali_loss2, self.model,
                           os.path.join(path, '{}_{}'.format(self.dataset, self.file_name)))

            if early_stopping.early_stop:
                print("Early stopping")
                break
            adjust_learning_rate(self.optimizer, epoch + 1, self.lr)

    def test(self):
        self.model.load_state_dict(
            torch.load(
                os.path.join(self.model_save_path, '{}_{}_{}'.format(self.dataset, self.file_name, 'checkpoint.pth')))
        )

        self.model.eval()
        temperature = 50

        print("======================TEST MODE======================")

        criterion = nn.MSELoss(reduce=False)

        # (1) stastic on the train set
        attens_energy = []
        for i, input_data in enumerate(self.train_loader):
            input = input_data.float().to(self.device)
            output, series, prior, _ = self.model(input)
            loss = torch.mean(criterion(input, output), dim=-1)
            series_loss = 0.0
            prior_loss = 0.0
            for u in range(len(prior)):
                if u == 0:
                    series_loss = my_kl_loss(series[u], (
                            prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                   self.win_size)).detach()) * temperature
                    prior_loss = my_kl_loss(
                        (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                self.win_size)),
                        series[u].detach()) * temperature
                else:
                    series_loss += my_kl_loss(series[u], (
                            prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                   self.win_size)).detach()) * temperature
                    prior_loss += my_kl_loss(
                        (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                self.win_size)),
                        series[u].detach()) * temperature

            metric = torch.softmax((-series_loss - prior_loss), dim=-1)
            cri = metric * loss
            cri = cri.detach().cpu().numpy()
            attens_energy.append(cri)

        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        train_energy = np.array(attens_energy)

        # (2) find the threshold
        attens_energy = []
        for i, input_data in enumerate(self.thre_loader):
            input = input_data.float().to(self.device)
            output, series, prior, _ = self.model(input)

            loss = torch.mean(criterion(input, output), dim=-1)

            series_loss = 0.0
            prior_loss = 0.0
            for u in range(len(prior)):
                if u == 0:
                    series_loss = my_kl_loss(series[u], (
                            prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                   self.win_size)).detach()) * temperature
                    prior_loss = my_kl_loss(
                        (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                self.win_size)),
                        series[u].detach()) * temperature
                else:
                    series_loss += my_kl_loss(series[u], (
                            prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                   self.win_size)).detach()) * temperature
                    prior_loss += my_kl_loss(
                        (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                self.win_size)),
                        series[u].detach()) * temperature
            # Metric
            metric = torch.softmax((-series_loss - prior_loss), dim=-1)
            cri = metric * loss
            cri = cri.detach().cpu().numpy()
            attens_energy.append(cri)

        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        test_energy = np.array(attens_energy)
        combined_energy = np.concatenate([train_energy, test_energy], axis=0)
        thresh = np.percentile(combined_energy, 100 - self.anormly_ratio)
        print("Threshold :", thresh)

        # (3) evaluation on the test set
        attens_energy = []
        for i, input_data in enumerate(self.thre_loader):
            input = input_data.float().to(self.device)
            output, series, prior, _ = self.model(input)

            loss = torch.mean(criterion(input, output), dim=-1)

            series_loss = 0.0
            prior_loss = 0.0
            for u in range(len(prior)):
                if u == 0:
                    series_loss = my_kl_loss(series[u], (
                            prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                   self.win_size)).detach()) * temperature
                    prior_loss = my_kl_loss(
                        (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                self.win_size)),
                        series[u].detach()) * temperature
                else:
                    series_loss += my_kl_loss(series[u], (
                            prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                   self.win_size)).detach()) * temperature
                    prior_loss += my_kl_loss(
                        (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                self.win_size)),
                        series[u].detach()) * temperature
            metric = torch.softmax((-series_loss - prior_loss), dim=-1)

            cri = metric * loss
            cri = cri.detach().cpu().numpy()
            attens_energy.append(cri)

        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        test_energy = np.array(attens_energy)

        pred = (test_energy > thresh).astype(int)

        print("pred:   ", pred.shape)

        # detection adjustment: please see this issue for more information https://github.com/thuml/Anomaly-Transformer/issues/14

        pred = np.array(pred)
        print("pred: ", pred.shape)

        return pred


def train_and_predict(model_name, retrain=False):#'anomaly_transformer'
    file_names_df = pd.read_csv(
        'D:/新建文件夹/anomaly_detection/data/MBR/file_names.csv')
    file_names = file_names_df.loc[:, 'file_name'].values.tolist()

    pred_df = pd.DataFrame()

    for file_name in file_names:
        if 'level_1' in file_name:
            print("处理文件: {}".format(file_name))
            data_path = 'D:/新建文件夹/anomaly_detection/data/MBR/original_data'
            model_save_path = 'D:/新建文件夹/anomaly_detection/tools/{}/checkpoints'.format(model_name)
            train_data = pd.read_csv(data_path + '/{}_train.csv'.format(file_name))
            test_data = pd.read_csv(data_path + '/{}_test.csv'.format(file_name))

            input_c = train_data.shape[1] - 3
            output_c = train_data.shape[1] - 3

            parser = argparse.ArgumentParser()

            if model_name == 'memto':
                parser.add_argument('--lr', type=float, default=0.01)
                parser.add_argument('--num_epochs', type=int, default=50)
                parser.add_argument('--k', type=int, default=3)
                parser.add_argument('--win_size', type=int, default=3)
                parser.add_argument('--input_c', type=int, default=input_c)
                parser.add_argument('--output_c', type=int, default=output_c)
                parser.add_argument('--batch_size', type=int, default=3)
                parser.add_argument('--step', type=int, default=3)
                parser.add_argument('--temp_param', type=float, default=0.05)
                parser.add_argument('--lambd', type=float, default=0.01)
                parser.add_argument('--pretrained_model', type=str, default=None)
                parser.add_argument('--dataset', type=str, default='MBR')
                parser.add_argument('--mode', type=str, default='train', choices=['train', 'test', 'memory_initial'])
                parser.add_argument('--data_path', type=str,
                                    default='/Users/xiefang09/xiefang/python/time_series-20230328/time_series/data/anomaly_on_deeplearning/MBR')
                parser.add_argument('--model_save_path', type=str,
                                    default='/Users/xiefang09/xiefang/python/time_series-20230328/time_series/tools/memto/checkpoints')
                parser.add_argument('--anormly_ratio', type=float, default=3)
                parser.add_argument('--device', type=str, default="cpu")  # cuda:0
                parser.add_argument('--n_memory', type=int, default=64, help='number of memory items')
                parser.add_argument('--num_workers', type=int, default=4 * torch.cuda.device_count())
                parser.add_argument('--d_model', type=int, default=128)
                parser.add_argument('--temperature', type=int, default=0.1)
                parser.add_argument('--memory_initial', type=str, default=False,
                                    help='whether it requires memory item embeddings. False: using random initialization, True: using customized intialization')
                parser.add_argument('--phase_type', type=str, default='second_train',
                                    choices=['second_train', 'first_train'],
                                    help='whether it requires memory item embeddings. False: using random initialization, True: using customized intialization')

                config = parser.parse_args()
                anomaly_transformer = AnomalyDetectionOnMemto(vars(config), file_name)

                if retrain:
                    anomaly_transformer.train(training_type='first_train')
                    anomaly_transformer.get_memory_initial_embedding(training_type='second_train')
            else:
                parser.add_argument('--lr', type=float, default=1e-4)
                parser.add_argument('--num_epochs', type=int, default=30)
                parser.add_argument('--k', type=int, default=3)
                parser.add_argument('--win_size', type=int, default=3)
                parser.add_argument('--batch_size', type=int, default=3)
                parser.add_argument('--step', type=int, default=3)
                parser.add_argument('--pretrained_model', type=str, default=None)
                parser.add_argument('--dataset', type=str, default='MBR')
                parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
                parser.add_argument('--data_path', type=str, default=data_path)
                parser.add_argument('--model_save_path', type=str, default=model_save_path)
                parser.add_argument('--anormly_ratio', type=float, default=3)

                parser.add_argument('--input_c', type=int, default=input_c)
                parser.add_argument('--output_c', type=int, default=output_c)

                config = parser.parse_args()
                anomaly_transformer = AnomalyDetectionOnAnomalyTransformer(vars(config), file_name)

                if retrain:
                    anomaly_transformer.train()

            pred = anomaly_transformer.test()
            pred_df_tmp = pd.DataFrame(pred)
            pred_df_tmp.columns = ['pred']

            index_name = test_data.loc[:, '指标名称'].values.tolist()
            month_list = test_data.loc[:, 'Month'].values.tolist()
            sequence = test_data.loc[:, 'by序列分类'].values.tolist()

            pred_df_tmp.insert(0, 'index_name', index_name)
            pred_df_tmp.insert(1, 'Month', month_list)
            pred_df_tmp.insert(2, 'sequence', sequence)

            pred_df_tmp.insert(3, 'method', file_name.split('_')[0], allow_duplicates=True)
            print(pred_df_tmp.shape)

            pred_df = pred_df.append(pred_df_tmp)

    result_path = '/Users/xiefang09/xiefang/python/time_series-20230328/time_series/result/anomaly_on_deeplearning/MBR'
    pred_df.to_csv(result_path + '/{}_prediction.csv'.format(model_name), encoding='utf_8_sig', index=False)

def analysis_transformer_memto_result():
    base_line_file = '/Users/xiefang09/xiefang/python/time_series-20230328/time_series/data/anomaly_on_deeplearning/base_line_long.csv'
    result_path = '/Users/xiefang09/xiefang/python/time_series-20230328/time_series/result/anomaly_on_deeplearning/MBR'

    dataset_folder = '/Users/xiefang09/xiefang/python/time_series-20230328/time_series/data/anomaly_on_deeplearning/MBR'
    data_set = pd.read_csv(dataset_folder + '/baseline_202309.csv')
    data_classification = pd.read_csv(dataset_folder + '/classification.csv')
    data_set = pd.merge(data_set, data_classification)

    dim_cols = ['指标名称', 'by BG/平台', 'by序列分类', '维度', 'method']
    col_name = ['2019-01', '2019-02', '2019-03', '2019-04', '2019-05', '2019-06', '2019-07', '2019-08', '2019-09',
                '2019-10', '2019-11', '2019-12', '2020-01', '2020-02', '2020-03', '2020-04', '2020-05', '2020-06',
                '2020-07', '2020-08', '2020-09', '2020-10', '2020-11', '2020-12', '2021-01', '2021-02', '2021-03',
                '2021-04', '2021-05', '2021-06', '2021-07', '2021-08', '2021-09', '2021-10', '2021-11', '2021-12',
                '2022-01', '2022-02', '2022-03', '2022-04', '2022-05', '2022-06', '2022-07', '2022-08', '2022-09',
                '2022-10', '2022-11', '2022-12', '2023-01', '2023-02', '2023-03', '2023-04', '2023-05', '2023-06',
                '2023-07', '2023-08', '2023-09']

    col_name = dim_cols + col_name

    data_long = pd.melt(data_set.loc[:, col_name], id_vars=dim_cols, var_name="time", value_name="value")
    data_long.columns = ['index_name', 'BG', 'sequence', 'dim', 'method', 'Month_', 'value']
    data_long.loc[:, 'Month_'] = data_long.loc[:, 'Month_'].apply(
        lambda x: int(x.replace('-', ''))
    )

    # anomaly_transformer
    anomaly_transformer_file_name = result_path + '/{}_prediction.csv'.format('anomaly_transformer')
    memto_file_name = result_path + '/{}_prediction.csv'.format('memto')

    anomaly_transformer_pred = pd.read_csv(anomaly_transformer_file_name)
    anomaly_transformer_pred.loc[:, 'Month_'] = anomaly_transformer_pred.loc[:, 'Month'].apply(
        lambda x: int(x[:7].replace('-', ''))
    )
    anomaly_transformer_pred = pd.merge(anomaly_transformer_pred, data_long)

    # memto
    memto_pred = pd.read_csv(memto_file_name)
    memto_pred.loc[:, 'Month_'] = memto_pred.loc[:, 'Month'].apply(
        lambda x: int(x[:7].replace('-', ''))
    )
    memto_pred = pd.merge(memto_pred, data_long)

    base_line = pd.read_csv(base_line_file)
    base_line.loc[:, 'is_anomaly'] = base_line.apply(
        lambda x: 1 if pd.notnull(x['reason_name']) else 0, axis=1
    )

    anomaly_transformer_pred = pd.merge(anomaly_transformer_pred, base_line, how='left')
    anomaly_transformer_pred.loc[:, 'is_anomaly'] = anomaly_transformer_pred.loc[:, 'is_anomaly'].apply(
        lambda x: 0 if pd.isnull(x) else x
    )

    memto_pred = pd.merge(memto_pred, base_line, how='left')
    memto_pred.loc[:, 'is_anomaly'] = memto_pred.loc[:, 'is_anomaly'].apply(
        lambda x: 0 if pd.isnull(x) else x
    )

    dims = ['index_name', 'BG', 'sequence']
    anomaly_transformer_pred['pred_lead'] = anomaly_transformer_pred.sort_values('Month_').groupby(dims)['pred'].shift(
        -1)
    anomaly_transformer_pred['pred_lag'] = anomaly_transformer_pred.sort_values('Month_').groupby(dims)['pred'].shift(1)

    anomaly_transformer_pred.to_csv(result_path + '/{}_prediction_baseline.csv'.format('anomaly_transformer'),
                                    encoding='utf_8_sig', index=False)

    dims = ['index_name', 'BG', 'sequence']
    memto_pred['pred_lead'] = memto_pred.sort_values('Month_').groupby(dims)['pred'].shift(-1)
    memto_pred['pred_lag'] = memto_pred.sort_values('Month_').groupby(dims)['pred'].shift(1)
    memto_pred.to_csv(result_path + '/{}_prediction_baseline.csv'.format('memto'), encoding='utf_8_sig', index=False)

    return

if __name__ == '__main__':
    # train_and_predict(model_name='memto', retrain=True)
    analysis_transformer_memto_result()





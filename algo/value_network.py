import os, sys, torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from collections import deque
base_path = os.getcwd()
sys.path.append(base_path)
from Stage2.models import Transformer_Value_Network, Toy_Value_Network
from Stage2.envs.state import State
from utils.utils import readFile, readFilewithload
from configs.config import get_base_config, make_config

class Truss_Dataset(Dataset):
    def __init__(self, storage, scale_value_max = None, max_value = None, 
                start_index = None, end_index = None):
        self.storage = storage
        self.scale_value_max = scale_value_max
        self.max_value = max_value     
        self.start_index = start_index
        if (self.start_index == None): self.start_index = 0
        self.end_index = end_index
        if (self.end_index == None): self.end_index = len(self.storage)
    
    def __getitem__(self, index):
        return self.storage[index + self.start_index]

    def __len__(self):
        return self.end_index - self.start_index

    def collate_fn(self, samples):
        bz = len(samples)
        truss_input = torch.zeros(bz, len(samples[0]['truss_input']))
        truss_valid = torch.zeros(bz, dtype=int)
        truss_value = torch.zeros(bz, len(samples[0]['truss_value']))
        for i in range(len(samples)):
            truss_input[i] = torch.from_numpy(samples[i]['truss_input'])
            truss_valid[i] = torch.from_numpy(samples[i]['truss_valid'])[0]
            if (self.scale_value_max != None):
                truss_value[i] = torch.from_numpy(samples[i]['truss_value'] / self.max_value * self.scale_value_max)
            else:
                truss_value[i] = torch.from_numpy(samples[i]['truss_value'])
        truss_input = truss_input.to('cuda:0')
        truss_valid = truss_valid.to('cuda:0')
        truss_value = truss_value.to('cuda:0')
        return {
            'truss_input' : truss_input,
            'truss_valid' : truss_valid,
            'truss_value' : truss_value
        }

class Value_Network:
    def __init__(self, args, storage_size = 1000000, batch_size = 32, lr = 1e-3):
        self.value_network = Transformer_Value_Network(args.prev_dims, args.hidden_dims, args.env_dims, args.env_mode, num_node = args.maxp).to('cuda:0')
        #self.value_network = Toy_Value_Network(args.maxp * 3 + args.maxp * (args.maxp - 1) // 2, 2048)
        self.storage_size = storage_size
        self.batch_size = batch_size
        self.storage = deque(maxlen = self.storage_size)
        self.valid_loss_fc = nn.CrossEntropyLoss()
        self.value_loss_fc = nn.MSELoss()
        self.value_loss_alpha = 1
        self.lr = lr
        self.max_value = -1
        self.valid_count = 0
        self.invalid_count = 0
        self.save_model_path = os.path.join(args.save_model_path, args.run_id)

    def pred(self, input):
        self.value_network.eval()
        return self.value_network(input)

    def one_dim_presentation(self, points, edges, block_rate = 0):
        r'''
        block_rate: how much of the Edges will be remove.
        '''
        state = State(args.maxp, args.env_dims, args.env_mode)
        for i in range(args.maxp):
            state.nodes[i][0] = points[i].vec.x
            state.nodes[i][1] = points[i].vec.y
            if args.env_dims == 3:
                state.nodes[i][2] = points[i].vec.z

        for e in edges:
            if (np.random.random() > block_rate):
                i, j = e.u, e.v
                if (args.env_mode == 'Area'):
                    state.edges[i][j] = e.area
                    state.edges[j][i] = e.area

                if (args.env_mode == 'DT'):
                    state.edges[i][j][0] = e.d
                    state.edges[j][i][0] = e.d
                    state.edges[i][j][1] = e.t
                    state.edges[j][i][1] = e.t

        return state

    def upd_from_storage(self, steps = 50000, scale_value_max = 10, train_ratio = 0.8):
        num_train_data = int(train_ratio * len(self.storage) + 0.5)
        choose = np.arange(len(self.storage))
        np.random.shuffle(choose)
        training_storage = [self.storage[choose[i]] for i in range(num_train_data)]
        valid_storage = [self.storage[choose[i]] for i in range(num_train_data, len(self.storage))]
        train_dataset = Truss_Dataset(training_storage, scale_value_max = scale_value_max, max_value = self.max_value)
        
        train_dataloader = DataLoader(train_dataset, collate_fn=train_dataset.collate_fn, batch_size = self.batch_size, shuffle = True)
        
        optimizer = torch.optim.Adam(self.value_network.parameters(), lr = self.lr)
        
        current_step = 0
        current_epoch = 0
        min_valid_loss = 1e9
        while (current_step < steps):
            train_losses_valid = []
            train_losses_value = []   
            with tqdm(train_dataloader, desc = "training") as pbar:
              self.value_network.train()
              for samples in pbar:
                ### Training ###
                optimizer.zero_grad()
                valid_pred, value_pred = self.value_network(samples['truss_input'])
                value_pred[samples['truss_valid'] == 0] = 0
                loss_valid = self.valid_loss_fc(valid_pred, samples['truss_valid'])
                loss_value = self.value_loss_fc(value_pred, samples['truss_value'])
                #print(loss_valid, loss_value)
                loss = loss_valid + self.value_loss_alpha * loss_value
                loss.backward()
                optimizer.step()

                ### Logging ###
                train_losses_valid.append(loss_valid.item())
                train_losses_value.append(loss_value.item())

                current_step += 1
                if (current_step >= steps): break
                
                pbar.set_description("Epoch: %d, losses_valid: %0.8f, losses_value: %0.8f, lr: %0.6f" %
                                     (current_epoch + 1, np.mean(train_losses_valid), np.mean(train_losses_value),
                                      optimizer.param_groups[0]['lr']))
            current_epoch += 1

            print('##### epoch.{} #####'.format(current_epoch))
            print('train_loss_valid:', np.mean(train_losses_valid))
            print('train_loss_value:', np.mean(train_losses_value))

            now_valid_loss = self.eval_storage(valid_storage, descending = 'valid')
            if (now_valid_loss < min_valid_loss):
                self.save_model("value_network_best.pt")
                min_valid_loss = now_valid_loss
            self.save_model("value_network_{}.pt".format(current_epoch))

            print('min_valid_loss:', min_valid_loss)
            print('now_valid_loss:', now_valid_loss)
            print("#" * 19)

    def eval_storage(self, eval_storage = None, scale_value_max = 10, descending = 'eval'):
        r'''
        evaluate the value network in a given storage
        '''
        if (eval_storage == None): eval_storage = self.storage
        eval_dataset = Truss_Dataset(eval_storage, scale_value_max = scale_value_max, max_value = self.max_value)
        eval_dataloader = DataLoader(eval_dataset, collate_fn=eval_dataset.collate_fn, batch_size = self.batch_size, shuffle = True)
        eval_losses_valid = []
        eval_losses_value = []
        correct_pred_num = 0
        total_pred_num = 0
        with tqdm(eval_dataloader, desc = descending) as pbar:
              self.value_network.eval()  
              for samples in pbar:
                ### validing ###
                with torch.no_grad():
                    valid_pred, value_pred = self.value_network(samples['truss_input'])
                    value_pred[samples['truss_valid'] == 0] = 0
                    loss_valid = self.valid_loss_fc(valid_pred, samples['truss_valid'])
                    loss_value = self.value_loss_fc(value_pred, samples['truss_value'])

                    valid_pred_label = torch.argmax(valid_pred, dim = -1, keepdim = False)
                    correct_pred_num += torch.sum(valid_pred_label == samples['truss_valid']).item()
                    total_pred_num += valid_pred_label.shape[0]

                ### Logging ###
                eval_losses_valid.append(loss_valid.item())
                eval_losses_value.append(loss_value.item())
                
                pbar.set_description("losses_eval: %0.8f, losses_eval: %0.8f" %
                                     (np.mean(eval_losses_valid), np.mean(eval_losses_value)))

        now_eval_loss = np.mean(eval_losses_valid) + self.value_loss_alpha * np.mean(eval_losses_value)
        
        print('{}_loss_valid:'.format(descending), np.mean(eval_losses_valid))
        print('{}_loss_value:'.format(descending), np.mean(eval_losses_value))
        print("predict ratio:", correct_pred_num / total_pred_num, correct_pred_num, '/', total_pred_num)

        return now_eval_loss

    def upd_storage(self, data):
        self.storage.append(data)
        
    def init_storage(self, data_path, invalid_data_path = None, threshold = None, copy_num = 1, invalid_copy_num = 1, clear = False):
        r'''
        threshold: only use the data with value smaller than the threshold
        copy_num: remove some Edges in the Truss; 
                  set to 1 means no remove
        '''
        if (clear): 
            self.storage.clear()
            self.valid_count = self.invalid_count = 0

        files = os.listdir(data_path)
        if (files[0][:-4] == '.txt'):
            files.sort(key = lambda x: int(x[:-4]))
        else: files.sort()

        for file in files:
            if (not(threshold == None or int(file[:-4]) <= threshold)): continue
            points, edges, load = readFilewithload(data_path + file)
            load = np.array(load)
            for i in range(copy_num):
                if (i == 0): state = self.one_dim_presentation(points, edges)
                else: state = self.one_dim_presentation(points, edges, block_rate = np.random.random())
                data = {
                    'truss_input' : np.concatenate([state.obs(nonexistent_edge = 0), load]),
                    'truss_valid' : np.array([1]),
                    'truss_value' : np.array([int(file[:-4])])
                }
                self.max_value = max(self.max_value, int(file[:-4]))
                self.valid_count += 1
                self.storage.append(data)
        
        if (invalid_data_path != None):
            files = os.listdir(invalid_data_path)
            for file in files:
                points, edges, load = readFilewithload(invalid_data_path + file)
                load = np.array(load)
                for i in range(invalid_copy_num):
                    if (i == 0): state = self.one_dim_presentation(points, edges)
                    else: state = self.one_dim_presentation(points, edges, block_rate = np.random.random())
                    data = {
                        'truss_input' : np.concatenate([state.obs(nonexistent_edge = 0), load]),
                        'truss_valid' : np.array([0]),
                        'truss_value' : np.array([0])
                    }
                    self.invalid_count += 1
                    self.storage.append(data)
        print(self.valid_count, self.invalid_count)

    def save_model(self, network_name = 'value_network.pt'):
        if (not os.path.exists(self.save_model_path)):
            os.mkdir(self.save_model_path)
        torch.save(self.value_network, os.path.join(self.save_model_path, network_name))
    
    def load_model(self, model_load_path = None):
        if (model_load_path == None): model_load_path = os.path.join(self.save_model_path, "value_network_best.pt")
        self.value_network = torch.load(model_load_path)
        print("load from", model_load_path)

if __name__ == '__main__':
    parser = get_base_config()
    args = parser.parse_known_args(sys.argv[1:])[0]
    config = make_config(args.config)
    for k, v in config.get("base", {}).items():
        if f"--{k}" not in args:
            setattr(args, k, v)

    Value_Net = Value_Network(args, batch_size = 32)
    Value_Net.init_storage(data_path = './results_3d/without_buckle_case1/buckle_fixed0/MASS_ALL_Result/', invalid_data_path = './results_3d/without_buckle_case1/buckle_fixed0/invalid/', copy_num = 10, invalid_copy_num = 1)
    Value_Net.upd_from_storage()
    Value_Net.eval_storage()
    #print(len(Value_Net.storage))
# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
import numpy as np

class BidirectionalModel(nn.Module):
    def __init__(self):
        super().__init__()
        
        h_size = 30
        self.rnn = nn.GRU(input_size=2, hidden_size=h_size, num_layers=1, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(h_size*2, 4)
        self.h0 = nn.Parameter(torch.zeros(self.rnn.num_layers*2, 1, h_size).normal_(std=0.01), requires_grad=True)
    
    def forward(self, x):
        h0 = self.h0.expand(-1, x.shape[0], -1).contiguous()
        x, _ = self.rnn(x, h0)
        x = self.fc(x)
        return x
    
    
class DelayedModel(nn.Module):
    def __init__(self, delay):
        super().__init__()
        
        h_size = 30
        self.delay = delay
        self.rnn = nn.GRU(input_size=2, hidden_size=h_size, num_layers=1, batch_first=True)
        self.fc = nn.Linear(h_size, 4)
        self.h0 = nn.Parameter(torch.zeros(self.rnn.num_layers, 1, h_size).normal_(std=0.01), requires_grad=True)
        self.eos = nn.Parameter(torch.zeros(1, 1, 2), requires_grad=True)
    
    def forward(self, x):
        h0 = self.h0.expand(-1, x.shape[0], -1).contiguous()
        x = torch.cat((x, self.eos.expand(x.shape[0], self.delay, 2)), dim=1)
        x, _ = self.rnn(x, h0)
        x = self.fc(x[:, self.delay:])
        return x
    
    
class PaceNetwork:
    def __init__(self, bidirectional=True):
        if bidirectional:
            self.model = BidirectionalModel()
        else:
            self.model = DelayedModel(100) # The delay is expressed in # of spline points
        self.use_cuda = False
            
    def cuda(self):
        self.model.cuda()
        self.use_cuda = True
        return self
    
    @staticmethod
    def _angle_difference(y, x):
        """
        Compute the signed angle sum y + x.
        Both y, x, and the output are represented as versors.
        """
        cosine = y[:, 0]*x[:, 0] + y[:, 1]*x[:, 1]
        sine = y[:, 1]*x[:, 0] - y[:, 0]*x[:, 1]
        return np.stack((cosine, sine), axis=1)
    
    @staticmethod
    def _angle_sum(y, x):
        """
        Compute the signed angle sum y + x.
        Both y, x, and the output are represented as versors.
        """
        cosine = y[:, 0]*x[:, 0] - y[:, 1]*x[:, 1]
        sine = y[:, 1]*x[:, 0] + y[:, 0]*x[:, 1]
        return np.stack((cosine, sine), axis=1)
    
    @staticmethod
    def _extract_features(spline, inputs_only=False, speed=None):
        inputs = np.concatenate((
                spline.get_track('curvature').reshape(-1, 1),
                (spline.get_track('average_speed') if speed is None else np.full(spline.size(), speed)).reshape(-1, 1)
            ), axis=1)
        if inputs_only:
            return inputs
        
        outputs = np.concatenate((
                spline.get_track('amplitude').reshape(-1, 1),
                spline.get_track('frequency').reshape(-1, 1),
                PaceNetwork._angle_difference(spline.get_track('direction'), spline.get_track('tangent'))
            ), axis=1)
        return inputs, outputs
        
    
    def _prepare_next_batch(self, batch_size, chunk_length, dataset, sequences):
        batch_in = np.zeros((batch_size, chunk_length, 2), dtype='float32')
        batch_out = np.zeros((batch_size, chunk_length, 4), dtype='float32')
        pseudo_passes = (len(sequences)+batch_size-1)//batch_size
        
        probs = []
        for i, (subject, action) in enumerate(sequences):
            if 'spline' not in dataset[subject][action]:
                raise KeyError('No splines found. Perhaps you forgot to compute them?')
            probs.append(dataset[subject][action]['spline'].size())
        probs = np.array(probs)/np.sum(probs)
        
        for p in range(pseudo_passes):
            idxs = np.random.choice(len(sequences), size=batch_size, replace=True, p=probs)
            for i, (subject, action) in enumerate(np.array(sequences)[idxs]):
                # Pick a random chunk from each sequence
                spline = dataset[subject][action]['spline']
                full_seq_length = spline.size()
                max_index = full_seq_length - chunk_length + 1
                start_idx = np.random.randint(0, max_index)
                end_idx = start_idx + chunk_length
                inputs, outputs = PaceNetwork._extract_features(spline)
                batch_in[i], batch_out[i] = inputs[start_idx:end_idx], outputs[start_idx:end_idx]

            yield batch_in, batch_out
    
    def train(self, dataset, sequences_train, sequences_valid, batch_size=40, chunk_length=1000, n_epochs=2000):
        np.random.seed(1234)
        self.model.train()
 
        lr = 0.001
        lr_decay = 0.999
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.L1Loss()

        if len(sequences_valid) > 0:
            inputs_valid, outputs_valid = next(self._prepare_next_batch(batch_size, chunk_length, dataset, sequences_valid))
            inputs_valid = torch.from_numpy(inputs_valid)
            outputs_valid = torch.from_numpy(outputs_valid)
            if self.use_cuda:
                inputs_valid = inputs_valid.cuda()
                outputs_valid = outputs_valid.cuda()
                
        losses = []
        valid_losses = []
        start_epoch = 0
        start_time = time.time()
        try:
            for epoch in range(n_epochs):
                batch_loss = 0.0
                N = 0
                for inputs, outputs in self._prepare_next_batch(batch_size, chunk_length, dataset, sequences_train):
                    inputs = torch.from_numpy(inputs)
                    outputs = torch.from_numpy(outputs)
                    if self.use_cuda:
                        inputs = inputs.cuda()
                        outputs = outputs.cuda()

                    optimizer.zero_grad()
                    predicted = self.model(inputs)
                    loss = criterion(predicted, outputs)
                    loss.backward()
                    optimizer.step()

                    batch_loss += loss.item() * inputs.shape[0]
                    N += inputs.shape[0]

                batch_loss /= N
                losses.append(batch_loss)

                if len(sequences_valid) > 0:
                    with torch.no_grad():
                        predicted = self.model(inputs)
                        loss = criterion(predicted, outputs)
                        valid_losses.append(loss.item())
                        print('[%d] loss %.6f valid %.6f' % (epoch + 1, losses[-1], valid_losses[-1]))
                else:
                    print('[%d] loss %.6f' % (epoch + 1, losses[-1]))

                for param_group in optimizer.param_groups:
                    param_group['lr'] *= lr_decay
                epoch += 1
                if epoch > 0 and (epoch+1) % 10 == 0:
                    time_per_epoch = (time.time() - start_time)/(epoch - start_epoch)
                    print('Benchmark:', time_per_epoch, 's per epoch')
                    start_epoch = epoch
                    start_time = time.time()
        except KeyboardInterrupt:
            print('Training aborted.')

        print('Finished Training')
        return losses, valid_losses
    
    def predict(self, spline, average_speed=None):
        """
        Annotate the given spline with the predicted outputs of this pace network.
        If 'average_speed' is None, it will be obtained from the spline track named 'average_speed'.
        """
        with torch.no_grad():
            out_spline = spline.reparameterize(5, smoothing_factor=1)
            inputs = PaceNetwork._extract_features(out_spline, inputs_only=True, speed=average_speed)
            inputs = torch.from_numpy(inputs.astype('float32')).unsqueeze(0)
            if self.use_cuda:
                inputs = inputs.cuda()
                
            outputs = self.model(inputs).squeeze(0).cpu().numpy()
            out_spline.add_track('amplitude', np.clip(outputs[:, 0], 0.1, None))
            out_spline.add_track('frequency', np.clip(outputs[:, 1], 0, None))
            outputs[:, [2, 3]] /= np.linalg.norm(outputs[:, [2, 3]], axis=1).reshape(-1, 1) + 1e-9 # Normalize
            out_spline.add_track('direction', self._angle_sum(out_spline.get_track('tangent'), outputs[:, [2, 3]]))
            return out_spline
                
    def save_weights(self, model_file):
        print('Saving weights to', model_file)
        torch.save(self.model.state_dict(), model_file)
        
    def load_weights(self, model_file):
        print('Loading weights from', model_file)
        self.model.load_state_dict(torch.load(model_file, map_location=lambda storage, loc: storage))
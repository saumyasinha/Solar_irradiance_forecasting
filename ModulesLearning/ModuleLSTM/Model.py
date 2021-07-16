import torch
import math
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
# from SolarForecasting.ModulesLearning.ModuleLSTM.functions import *
from torch.nn.modules.activation import MultiheadAttention



class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, saving_path, patience=7, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.saving_path = saving_path

    def __call__(self, val_loss, model, epoch, parallel=False):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model,parallel)
        elif (score < self.best_score + self.delta):
            self.counter += 1
            print('EarlyStopping counter: {} out of {}'.format(self.counter, self.patience))
            if self.counter >= self.patience:
                self.early_stop = True
        elif epoch>20:
            self.best_score = score
            self.save_checkpoint(val_loss, model, parallel)
            self.counter = 0

    def save_checkpoint(self, val_loss, model,parallel):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print('Validation loss decreased ({} --> {}).  Saving model ...'.format(self.val_loss_min, val_loss))
        if parallel==True:
            torch.save(model.module.state_dict(), self.saving_path)
        else:
            torch.save(model.state_dict(), self.saving_path)
        self.val_loss_min = val_loss


class quantileLSTM(nn.Module):

    # def __init__(self, num_classes, input_size, hidden_size, num_layers)
    def __init__(self, input_dim, timesteps, folder_saving, model, quantile, hidden_size, num_layers=1, alphas=None, outputs=None, valid=False):
        super(quantileLSTM, self).__init__()

        self.quantile = quantile
        if self.quantile:
            assert outputs == len(alphas), "The outputs and the quantiles should be of the same dimension"


        self.outputs = outputs
        self.num_layers = num_layers
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.timesteps = timesteps
        self.alphas = alphas
        self.valid = valid
        self.train_mode = False
        self.saving_path = folder_saving + model

        self.lstm = nn.LSTM(input_size=self.input_dim, hidden_size=self.hidden_size,
                            num_layers=self.num_layers, batch_first=True)

        self.fc = nn.Linear(hidden_size, self.outputs)

    def forward(self, x):
        h_0 = Variable(torch.zeros(
            self.num_layers, x.size(0), self.hidden_size))

        c_0 = Variable(torch.zeros(
            self.num_layers, x.size(0), self.hidden_size))

        # Propagate input through LSTM
        ula, (h_out, _) = self.lstm(x, (h_0, c_0))

        h_out = h_out.view(-1, self.hidden_size)

        out = self.fc(h_out)

        return out

#
# class SimplePositionalEncoding(torch.nn.Module):
#     def __init__(self, d_model, dropout=0.1, max_len=5000):
#         super(SimplePositionalEncoding, self).__init__()
#         self.dropout = torch.nn.Dropout(p=dropout)
#         pe = torch.zeros(max_len, d_model)
#         position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
#         div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
#         pe[:, 0::2] = torch.sin(position * div_term)
#         pe[:, 1::2] = torch.cos(position * div_term)
#         pe = pe.unsqueeze(0).transpose(0, 1)
#         self.register_buffer('pe', pe)
#
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """Creates a basic positional encoding"""
#         x = x + self.pe[:x.size(0), :]
#         return self.dropout(x)

# class EncoderLayer(torch.nn.Module):
#     def __init__(self, dim_val, dim_attn, n_heads=1):
#         super(EncoderLayer, self).__init__()
#         self.attn = MultiHeadAttentionBlock(dim_val, dim_attn, n_heads)
#         self.fc1 = nn.Linear(dim_val, dim_val)
#         self.fc2 = nn.Linear(dim_val, dim_val)
#
#         self.norm1 = nn.LayerNorm(dim_val)
#         self.norm2 = nn.LayerNorm(dim_val)
#
#     def forward(self, x):
#         a = self.attn(x)
#         x = self.norm1(x + a)
#
#         a = self.fc1(F.elu(self.fc2(x)))
#         x = self.norm2(x + a)
#
#         return x
#
#
# class DecoderLayer(torch.nn.Module):
#     def __init__(self, dim_val, dim_attn, n_heads=1):
#         super(DecoderLayer, self).__init__()
#         self.attn1 = MultiHeadAttentionBlock(dim_val, dim_attn, n_heads)
#         self.attn2 = MultiHeadAttentionBlock(dim_val, dim_attn, n_heads)
#         self.fc1 = nn.Linear(dim_val, dim_val)
#         self.fc2 = nn.Linear(dim_val, dim_val)
#
#         self.norm1 = nn.LayerNorm(dim_val)
#         self.norm2 = nn.LayerNorm(dim_val)
#         self.norm3 = nn.LayerNorm(dim_val)
#
#     def forward(self, x, enc):
#         a = self.attn1(x)
#         x = self.norm1(a + x)
#
#         a = self.attn2(x, kv=enc)
#         x = self.norm2(a + x)
#
#         a = self.fc1(F.elu(self.fc2(x)))
#
#         x = self.norm3(x + a)
#         return x
#
#
#
# class Transformer(nn.Module):
#
#     def __init__(self, input_dim, timesteps, folder_saving, model, quantile, alphas = None, outputs = None, valid = False, out_seq_len=1, n_decoder_layers=3, n_encoder_layers=3,
#                  n_heads=1):
#
#         self.outputs = outputs
#         self.n_decoder_layers = n_decoder_layers
#         self.n_encoder_layers = n_encoder_layers
#         self.input_dim = input_dim
#         self.timesteps = timesteps
#         self.alphas = alphas
#         self.valid = valid
#         self.train_mode = False
#         self.saving_path = folder_saving + model
#         self.quantile = quantile
#         self.dec_seq_len = timesteps
#         self.out_seq_len = out_seq_len
#
#         super(Transformer, self).__init__()
#         # Initiate encoder and Decoder layers
#         self.encs = []
#         for i in range(n_encoder_layers):
#             self.encs.append(EncoderLayer(self.input_dim, self.input_dim//4, n_heads))
#
#         self.decs = []
#         for i in range(n_decoder_layers):
#             self.decs.append(DecoderLayer(self.input_dim, self.input_dim//4, n_heads))
#
#         self.pos = PositionalEncoding(self.input_dim)
#
#         # Dense layers for managing network inputs and outputs
#         # self.enc_input_fc = nn.Linear(input_size, self.input_dim)
#         # self.dec_input_fc = nn.Linear(input_size, self.input_dim)
#         self.out_fc = nn.Linear(self.dec_seq_len * self.input_dim,self.out_seq_len)
#
#     def forward(self, x):
#         # encoder
#         # e = self.encs[0](self.pos(self.enc_input_fc(x)))
#         x = x.transpose(1,2)
#         e = self.encs[0](self.pos(x))
#         for enc in self.encs[1:]:
#             e = enc(e)
#
#         # decoder
#         # d = self.decs[0](self.dec_input_fc(x[:, -self.dec_seq_len:]), e)
#         d = self.decs[0](x[:, -self.dec_seq_len:], e)
#         for dec in self.decs[1:]:
#             d = dec(d, e)
#
#         # output
#         x = self.out_fc(d.flatten(start_dim=1))
#
#         return x

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        #pe.requires_grad = False
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]


class TransAm(nn.Module):
    def __init__(self, input_dim, timesteps, folder_saving, model, quantile, alphas = None, outputs = None, valid = False,num_heads=4, d_model=128, num_layers=2, dropout=0.1):
        super(TransAm, self).__init__()
        self.model_type = 'Transformer'

        self.outputs = outputs
        self.num_layers = num_layers
        self.input_dim = input_dim
        self.d_model = d_model
        self.num_heads = num_heads
        self.timesteps = timesteps
        self.alphas = alphas
        self.valid = valid
        self.train_mode = False
        self.saving_path = folder_saving + model
        self.quantile = quantile

        self.src_mask = None

        self.input_embedding = nn.Conv1d(self.input_dim, self.d_model, 1)
        self.pos_encoder = PositionalEncoding(self.d_model)

        # self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.input_dim, nhead=2, dim_feedforward=200,dropout=dropout) #embed_dim must be divisible by n_heads

        self.encoder_layer = nn.TransformerEncoderLayer(self.d_model, nhead=self.num_heads, dim_feedforward=512, dropout=dropout) #embed_dim must be divisible by n_heads

        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=self.num_layers)
        self.decoder = nn.Linear(self.d_model, self.outputs)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        src = src.transpose(1, 2)
        src = self.input_embedding(src)  #batch,d_model,timesteps
        src = src.transpose(1,2)  #batch,timesteps,d_model
        src = src.transpose(0,1) #timesteps,batch,d_model

        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self._generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask

        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, self.src_mask)  # , self.src_mask)
        output = self.decoder(output)
        return output[-1]

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask





# class MultiAttnHeadSimple(torch.nn.Module):
#     """A simple multi-head attention model inspired by Vaswani et al."""
#
#     def __init__(
#             self, input_dim, seq_len, folder_saving, model, quantile, n_layers=2, factor=12, alphas=None, outputs=None, valid=False,
#
#          output_seq_len=1, num_heads=4, d_model=96, dropout=0.2): #0.5):
#
#         super(MultiAttnHeadSimple, self).__init__()
#
#         self.outputs = outputs
#         self.input_dim = input_dim
#         self.seq_len = seq_len
#         self.alphas = alphas
#         self.valid = valid
#         self.train_mode = False
#         self.saving_path = folder_saving + model
#         self.quantile = quantile
#         self.num_heads = num_heads
#         self.n_layers = n_layers
#
#         self.output_seq_len = output_seq_len
#         self.factor = factor
#         self.d_model = d_model
#         self.dropout = dropout
#
#         # #If solving multi horizon problem
#         # if output_seq_len>1:
#         #     self.factor = self.output_seq_len
#         #
#         #
#         # #self.factor = self.seq_len #setting this when not using dense interpolation
#         #
#         self.encoder = EncoderLayer(self.input_dim, self.seq_len, self.num_heads, self.n_layers, self.d_model, self.dropout)
#         #self.dense_interpolation = DenseInterpolation(self.seq_len, self.factor)
#         #
#         # if self.output_seq_len>1:
#         #     self.fc = nn.Linear(self.d_model, self.outputs)
#         # else:
#         self.fc = nn.Linear(int(self.d_model * self.seq_len), self.outputs)
#
#         # if self.output_seq_len >1:
#             # for i in range(self.output_seq_len):
#             #     setattr(self, "dense%d" % i, DenseInterpolation(self.seq_len, self.factor))
#             #     setattr(self, "fc%d" % i, nn.Linear(int(self.d_model * self.factor), self.outputs))
#         #
#         #
#
#
#
#     #     # self.dense_shape = torch.nn.Linear(number_time_series, d_model)
#     #     self.pe = SimplePositionalEncoding(self.d_model)
#     #     self.multi_attn = MultiheadAttention(
#     #         embed_dim=self.d_model, num_heads=self.num_heads, dropout=dropout)
#     #     self.final_layer = torch.nn.Linear(self.d_model, self.outputs)
#     #     self.length_data = seq_len
#     #     self.forecast_length = output_seq_len
#     #     self.last_layer = torch.nn.Linear(seq_len, output_seq_len)
#     #
#     #
#     # def forward(self, x: torch.Tensor, mask=None) -> torch.Tensor:
#     #
#     #     # Permute to (L, B, M)
#     #     x = x.permute(2, 0, 1)
#     #     # x = self.dense_shape(x)
#     #     x = self.pe(x)
#     #     if mask is None:
#     #         x = self.multi_attn(x, x, x)[0]
#     #     else:
#     #         x = self.multi_attn(x, x, x, attn_mask=self.mask)[0]
#     #     x = self.final_layer(x)
#     #
#     #     # Switch to (B, M, L)
#     #     x = x.permute(1, 2, 0)
#     #     x = self.last_layer(x)
#     #
#     #     if self.forecast_length>1:
#     #         return x
#     #     else:
#     #         return x[:,:,-1]
#
#
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         # print("input x shape", x.shape)
#         x = self.encoder(x)
#         # print("after encoding", x.shape)
#         #x = self.dense_interpolation(x)
#         # x = x.transpose(1,2)
#         if self.output_seq_len==1:
#             x = x.contiguous().view(-1, int(self.seq_len * self.d_model))
#         x = self.fc(x)
#         # x = x.contiguous().view(-1, int(self.factor * self.d_model))
#         pred_outputs = {}
#
#         # for i in range(self.output_seq_len):
#         #     y = getattr(self, "dense%d" % i)(x)
#         #     y = y.contiguous().view(-1, int(self.factor * self.d_model))
#         #     pred_outputs[i] = getattr(self, "fc%d" % i)(y)
#
#         # print("final output", x.shape)
#         return x
#         # return pred_outputs


def trainBatchwise(trainX, trainY, epochs, batch_size, lr, validX,
                   validY, n_output_length, n_features, n_timesteps, folder_saving, model_saved, quantile, n_layers, factor, alphas, outputs, valid, output_seq_len, num_heads, d_model, patience=None, verbose=None, reg_lamdba = 0):

    train_on_gpu = torch.cuda.is_available()
    print(train_on_gpu)
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    parallel = False
    quantile_forecaster = TransAm(n_features, n_timesteps, folder_saving, model_saved, quantile,  alphas = alphas, outputs = outputs, valid=valid, num_heads=num_heads, d_model=d_model, num_layers=n_layers)
    # quantile_forecaster = MultiAttnHeadSimple(n_features, n_timesteps, folder_saving, model_saved, quantile, n_layers, factor, alphas = alphas, outputs = outputs, valid=valid, output_seq_len = output_seq_len, num_heads=num_heads, d_model=d_model)
    if train_on_gpu:
        # if torch.cuda.device_count() > 1:
            # print("Let's use", torch.cuda.device_count(), "GPUs!")
            # quantile_forecaster = nn.DataParallel(quantile_forecaster)
            # parallel=True
        quantile_forecaster = quantile_forecaster.cuda()

        # point_foreaster = point_foreaster.cuda()

    print(quantile_forecaster)

    optimizer = torch.optim.Adam(quantile_forecaster.parameters(), lr=lr) #, betas = (0.9,0.98))
    # scheduler = StepLR(optimizer, step_size=25, gamma=0.1)
    # criterion = torch.nn.MSELoss()
    # criterion = nn.L1Loss()
    samples = trainX.size()[0]
    losses = []
    valid_losses = []

    saving_path = folder_saving + model_saved
    early_stopping = EarlyStopping(saving_path, patience=patience, verbose=True)
    train_mode = False

    for epoch in range(epochs):
        if train_mode is not True:
            quantile_forecaster.train()
            train_mode = True

        indices = torch.randperm(samples)
        trainX, trainY = trainX[indices, :, :], trainY[indices]
        per_epoch_loss = 0
        count_train = 0
        for i in range(0, samples, batch_size):
            xx = trainX[i: i + batch_size, :, :]
            yy = trainY[i: i + batch_size]

            if train_on_gpu:
                xx, yy = xx.cuda(), yy.cuda()

            outputs = quantile_forecaster.forward(xx)
            optimizer.zero_grad()
            if quantile:
                if n_output_length == 1:
                    loss = quantile_loss(outputs, yy, alphas)
                else:
                    # train loss for multiple outputs or multi-task learning
                    total_loss = []
                    for n in range(n_output_length):
                        # y_pred = outputs[:,n, :]
                        y_pred = outputs[n]
                        # calculate the batch loss
                        loss = quantile_loss(y_pred, yy[:, n], alphas)
                        total_loss.append(loss)

                    loss = sum(total_loss)

            else:
                loss = criterion(outputs, yy)

            reg_loss = np.sum([weights.norm(2) for weights in quantile_forecaster.parameters()])
            total_loss = loss + reg_lamdba / 2 * reg_loss
            # backward pass: compute gradient of the loss with respect to model parameters
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(quantile_forecaster.parameters(), 0.7)
            # perform a single optimization step (parameter update)
            optimizer.step()
            # scheduler.step()
            per_epoch_loss += loss.item()
            count_train += 1

        train_loss_this_epoch = per_epoch_loss / count_train
        losses.append(train_loss_this_epoch)

        if valid:
            train_mode = False
            quantile_forecaster.eval()
            if train_on_gpu:
                validX, validY = validX.cuda(), validY.cuda()

            if quantile:
                validYPred = quantile_forecaster.forward(validX)
                # validYPred = validYPred.cpu().detach().numpy()
                # validYTrue = validY.cpu().detach().numpy()
                # valid_loss_this_epoch = self.quantile_loss(validYPred,validY).item()

                if n_output_length == 1:
                    valid_loss_this_epoch = quantile_loss(validYPred, validY, alphas).item()
                else:
                    # train loss for multiple outputs or multi-task learning
                    total_loss = []
                    for n in range(n_output_length):
                        # y_pred = validYPred[:, n, :]
                        y_pred = validYPred[n]
                        # calculate the batch loss
                        loss = quantile_loss(y_pred, validY[:, n], alphas)
                        total_loss.append(loss)

                    valid_loss_this_epoch = sum(total_loss).item()

                # valid_loss = elf.crps_score(validYPred, validYTrue, np.arange(0.05, 1.0, 0.05))s
                valid_losses.append(valid_loss_this_epoch)
                print("Epoch: %d, loss: %1.5f and valid_loss : %1.5f" % (epoch, train_loss_this_epoch, valid_loss_this_epoch))
            else:
                validYPred = quantile_forecaster.forward(validX)

                # # valid loss for multiple outputs or multi-task learning
                # total_loss = []
                # for n in range(self.outputs):
                #     y_pred = validYPred[:, n]
                #     # calculate the batch lossnb6h
                #     validloss = criterion(y_pred, validY[:, n])
                #     total_loss.append(validloss)

                # validloss = sum(total_loss)
                # valid_loss_this_epoch = validloss.item()
                valid_loss_this_epoch = criterion(validYPred, validY).item()
                # validYPred = validYPred.cpu().detach().numpy()
                # validYTrue = validY.cpu().detach().numpy()
                # valid_loss = np.sqrt(mean_squared_error(validYPred, validYTrue))
                valid_losses.append(valid_loss_this_epoch)
                print("Epoch: %d, train loss: %1.5f and valid loss : %1.5f" % (
                epoch, train_loss_this_epoch, valid_loss_this_epoch))

            # early_stopping(valid_loss, self)
            early_stopping(valid_loss_this_epoch, quantile_forecaster, epoch, parallel)

            if early_stopping.early_stop:
                print("Early stopping")
                break
        else:
            print("Epoch: %d, loss: %1.5f" % (epoch, train_loss_this_epoch))
            if parallel == True:
                torch.save(quantile_forecaster.module.state_dict(), saving_path)
            else:
                torch.save(quantile_forecaster.state_dict(), saving_path)
        # load the last checkpoint with the best model
    # self.load_state_dict(torch.load('checkpoint.pt'))
    return losses, valid_losses


def crps_score(outputs, target, alphas):
    loss = []
    for i, alpha in enumerate(alphas):
        output = outputs[:, i].reshape((-1, 1))
        covered_flag = (output <= target).astype(np.float32)
        uncovered_flag = (output > target).astype(np.float32)
        if i == 0:
            loss.append(np.mean(
                ((target - output) * alpha * covered_flag + (output - target) * (1 - alpha) * uncovered_flag)))
        else:
            loss.append(np.mean(
                ((target - output) * alpha * covered_flag + (output - target) * (1 - alpha) * uncovered_flag)))

    return 2*np.mean(np.array(loss))

def quantile_loss(outputs, target, alphas):

    for i, alpha in zip(range(len(alphas)),alphas):
        output = outputs[:, i].reshape((-1, 1))
        covered_flag = (output <= target).float()
        uncovered_flag = (output > target).float()
        if i == 0:
            loss = ((target - output) * alpha * covered_flag + (output - target) * (1 - alpha) * uncovered_flag)
        else:
            loss += ((target - output) * alpha * covered_flag + (output - target) * (1 - alpha) * uncovered_flag)

    return torch.mean(loss)






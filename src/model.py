"""Function and class definitions for the LSTM and CNN models for drugVQA."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


def conv3x3(in_channels, out_channels, stride=1):
    """Return 2D convolutional layer with 3x3 filter."""
    return nn.Conv2d(in_channels, out_channels, kernel_size=3,
                     stride=stride, padding=1, bias=False)


def conv5x5(in_channels, out_channels, stride=1):
    """Return 2D convolutional layer with 5x5 filter."""
    return nn.Conv2d(in_channels, out_channels, kernel_size=5,
                     stride=stride, padding=2, bias=False)


def conv1x1(in_channels, out_channels, stride=1):
    """Return 2D convolutional layer with 1x1 filter."""
    return nn.Conv2d(in_channels, out_channels, kernel_size=1,
                     stride=stride, padding=0, bias=False)


class ResidualBlock(nn.Module):
    """Implements one CNN residual block.

    Specifically, the block consists of up to seven elements:
        5x5 convolution
        Batch normalization
        ELU activatiojn
        3x3 convolution
        Batch normalization
        (Optional) downsampling
        ELU activation of residual and block output.
    """

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv5x5(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.elu = nn.ELU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        """Run the forward pass through the residual block."""
        # The residual to allow the block to learn a zero-mapping if necessary.
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.elu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        # The sum -- residual and block output -- is activated.
        out = self.elu(out)
        return out


class DrugVQA(torch.nn.Module):
    """
    The class is an implementation of the DrugVQA model including.

    regularization and without pruning.
    Slight modifications have been done for speedup.
    """

    def __init__(self, args, block):
        """
        Initializes parameters suggested in paper

        args:
            batch_size  : {int} batch_size used for training
            lstm_hid_dim: {int} hidden dimension for lstm
            d_a         : {int} hidden dimension for the dense layer
            r           : {int} attention-hops or attention heads
            n_chars_smi : {int} voc size of smiles
            n_chars_seq : {int} voc size of protein sequence
            dropout     : {float}
            in_channels : {int} channels of CNN block input
            cnn_channels: {int} channels of CNN block
            cnn_layers  : {int} num of layers of each CNN block
            emb_dim     : {int} embeddings dimension
            dense_hid   : {int} hidden dim for the output dense
            task_type   : [0, 1] 0-->binary_classification 1-->multiclass classification
            n_classes   : {int} number of classes

        Returns:
            self
        """
        super(DrugVQA, self).__init__()
        self.batch_size = args['batch_size']
        self.lstm_hid_dim = args['lstm_hid_dim']
        self.r = args['r']
        self.type = args['task_type']
        self.in_channels = args['in_channels']

        # LSTM attributes.
        self.embeddings = nn.Embedding(args['n_chars_smi'], args['emb_dim'])
        self.seq_embed = nn.Embedding(args['n_chars_seq'], args['emb_dim'])
        self.lstm = torch.nn.LSTM(args['emb_dim'], self.lstm_hid_dim, 2, batch_first=True, bidirectional=True, dropout=args['dropout'])
        self.linear_first = torch.nn.Linear(2*self.lstm_hid_dim, args['d_a'])
        self.linear_second = torch.nn.Linear(args['d_a'], args['r'])
        self.linear_first_seq = torch.nn.Linear(args['cnn_channels'], args['d_a'])
        self.linear_second_seq = torch.nn.Linear(args['d_a'], self.r)

        # CNN attributes.
        self.conv = conv3x3(1, self.in_channels)
        self.bn = nn.BatchNorm2d(self.in_channels)
        self.elu = nn.ELU(inplace=False)
        self.layer1 = self.make_layer(block, args['cnn_channels'], args['cnn_layers'])
        self.layer2 = self.make_layer(block, args['cnn_channels'], args['cnn_layers'])

        # Attributes relating to 
        self.linear_final_step = torch.nn.Linear(self.lstm_hid_dim*2+args['d_a'], args['dense_hid'])
        self.linear_final = torch.nn.Linear(args['dense_hid'], args['n_classes'])

        # Hidden state attributes.
        self.hidden_state = self.init_hidden()
        self.seq_hidden_state = self.init_hidden()

    def softmax(self, input, axis=1):
        """Apply softmax to the specified axis for the given Variable."""
        input_size = input.size()
        trans_input = input.transpose(axis, len(input_size)-1)
        trans_size = trans_input.size()
        input_2d = trans_input.contiguous().view(-1, trans_size[-1])
        soft_max_2d = F.softmax(input_2d)
        soft_max_nd = soft_max_2d.view(*trans_size)
        return soft_max_nd.transpose(axis, len(input_size) - 1)

    def init_hidden(self):
        """Initialize the hidden layer in the LSTM with medium."""
        return (Variable(torch.zeros(4, self.batch_size,
                                     self.lstm_hid_dim).cuda()),
                Variable(torch.zeros(4, self.batch_size,
                                     self.lstm_hid_dim)).cuda())

    def make_layer(self, block, out_channels, blocks, stride=1):
        """Add doc."""
        downsample = None
        if (stride != 1) or (self.in_channels != out_channels):
            downsample = nn.Sequential(
                conv3x3(self.in_channels, out_channels, stride=stride),
                nn.BatchNorm2d(out_channels))
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for i in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x1, x2):
        """Run the forward pass for the LSTM and CNN, and concatenate them.

        x1 refers to the SMILES sequence of the ligand, x2 to the contact map
        of the protein.
        """
        # LSTM forward pass.
        smile_embed = self.embeddings(x1)
        outputs, self.hidden_state = self.lstm(smile_embed, self.hidden_state)
        sentence_att = F.tanh(self.linear_first(outputs))
        sentence_att = self.linear_second(sentence_att)
        sentence_att = self.softmax(sentence_att, 1)
        sentence_att = sentence_att.transpose(1, 2)
        sentence_embed = sentence_att@outputs
        # Multi-head attention is used.
        avg_sentence_embed = torch.sum(sentence_embed, 1) / self.r

        # CNN forward pass.
        pic = self.conv(x2)
        pic = self.bn(pic)
        pic = self.elu(pic)
        pic = self.layer1(pic)
        pic = self.layer2(pic)
        pic_emb = torch.mean(pic, 2)
        pic_emb = pic_emb.permute(0, 2, 1)
        seq_att = F.tanh(self.linear_first_seq(pic_emb))
        seq_att = self.linear_second_seq(seq_att)
        seq_att = self.softmax(seq_att, 1)
        seq_att = seq_att.transpose(1, 2)
        seq_embed = seq_att@pic_emb
        avg_seq_embed = torch.sum(seq_embed, 1) / self.r

        # Concatenate the embeddings of both the ligand and protein.
        sscomplex = torch.cat([avg_sentence_embed, avg_seq_embed], dim=1)
        sscomplex = F.relu(self.linear_final_step(sscomplex))

        # Sigmoid-activate if the response is binary, or use softmax.
        if not bool(self.type):
            return F.sigmoid(self.linear_final(sscomplex)), seq_att
        else:
            return F.log_softmax(self.linear_final(sscomplex)), seq_att

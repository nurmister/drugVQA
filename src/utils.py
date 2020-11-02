"""Functions used in loading and processing data."""

import torch
import numpy as np
from re import compile, split
from torch.autograd import Variable


def create_variable(tensor):
    """Create Variable object from tensor, stored in the GPU."""
    return Variable(tensor.cuda())


def replace_halogen(string):
    """Replace Br & Cl with R & L in SMILES string before tokenization."""
    br = compile('Br')
    cl = compile('Cl')
    string = br.sub('R', string)
    string = cl.sub('L', string)
    return string


def letter_to_index(char, letters):
    """Return the index of a letter in the (SMILES) letters string."""
    return letters.index(char)


def line2voc_arr(line, letters):
    """."""
    arr = []
    regex = "(\[[^\[\]]{1,10}\])"
    line = replace_halogen(line)
    char_list = split(regex, line)
    for li, char in enumerate(char_list):
        if char.startswith('['):
            arr.append(letter_to_index(char, letters))
        else:
            chars = [unit for unit in char]

            for i, unit in enumerate(chars):
                arr.append(letter_to_index(unit, letters))
    return arr, len(arr)


def make_variables(lines, properties, letters):
    """Create necessary variables, lengths, and target."""
    sequence_and_length = [line2voc_arr(line, letters) for line in lines]
    vectorized_seqs = [sl[0] for sl in sequence_and_length]
    seq_lengths = torch.LongTensor([sl[1] for sl in sequence_and_length])
    return pad_sequences(vectorized_seqs, seq_lengths, properties)


def make_variables_seq(lines, letters):
    """Add docstring."""
    sequence_and_length = [line2voc_arr(line, letters) for line in lines]
    vectorized_seqs = [sl[0] for sl in sequence_and_length]
    seq_lengths = torch.LongTensor([sl[1] for sl in sequence_and_length])
    return pad_sequences_seq(vectorized_seqs, seq_lengths)


# pad sequences and sort the tensor
def pad_sequences(vectorized_seqs, seq_lengths, properties):
    """Add docstring."""
    seq_tensor = torch.zeros((len(vectorized_seqs), seq_lengths.max())).long()
    for idx, (seq, seq_len) in enumerate(zip(vectorized_seqs, seq_lengths)):
        seq_tensor[idx, :seq_len] = torch.LongTensor(seq)

    # Sort tensors by their length
    seq_lengths, perm_idx = seq_lengths.sort(0, descending=True)
    seq_tensor = seq_tensor[perm_idx]

    # Also sort the target (countries) in the same order
    target = properties.double()
    if len(properties):
        target = target[perm_idx]
    # Return variables
    # DataParallel requires everything to be a Variable
    return create_variable(seq_tensor), create_variable(seq_lengths), create_variable(target)


def pad_sequences_seq(vectorized_seqs, seq_lengths):
    """Add docstring."""
    seq_tensor = torch.zeros((len(vectorized_seqs), seq_lengths.max())).long()
    for idx, (seq, seq_len) in enumerate(zip(vectorized_seqs, seq_lengths)):
        seq_tensor[idx, :seq_len] = torch.LongTensor(seq)

    # Sort tensors by their length
    seq_lengths, perm_idx = seq_lengths.sort(0, descending=True)
    seq_tensor = seq_tensor[perm_idx]
    # Return variables
    # DataParallel requires everything to be a Variable
    return create_variable(seq_tensor), create_variable(seq_lengths)


def construct_vocabulary(smiles_list, fname):
    """Return all the characters present in a SMILES file.

    Uses regex to find characters/tokens of the format '[x]'.
    """
    add_chars = set()
    for i, smiles in enumerate(smiles_list):
        regex = "(\[[^\[\]]{1,10}\])"
        smiles = replace_halogen(smiles)
        char_list = split(regex, smiles)
        for char in char_list:
            if char.startswith('['):
                add_chars.add(char)
            else:
                chars = [unit for unit in char]
                [add_chars.add(unit) for unit in chars]

    print("Number of characters: {}".format(len(add_chars)))
    with open(fname, 'w') as f:
        f.write('<pad>' + "\n")
        for char in add_chars:
            f.write(char + "\n")
    return add_chars


def readLinesStrip(lines):
    """Return """
    for i in range(len(lines)):
        lines[i] = lines[i].rstrip('\n')
    return lines


def getProtein(path, contactMapName, contactMap=True):
    """Return the sequence (and contact map) corresponding to the path."""
    proteins = open(path + "/" + contactMapName).readlines()
    proteins = readLinesStrip(proteins)
    seq = proteins[1]
    if contactMap:
        contactMap = []
        for i in range(2, len(proteins)):
            contactMap.append(proteins[i])
        return seq, contactMap
    else:
        return seq


def getTrainDataSet(trainFoldPath):
    """Return training data for the current fold."""
    with open(trainFoldPath, 'r') as f:
        trainCpi_list = f.read().strip().split('\n')
    trainDataSet = [cpi.strip().split() for cpi in trainCpi_list]
    return trainDataSet


def getTestProteinList(testFoldPath):
    """Add docstring."""
    testProteinList = readLinesStrip(open(testFoldPath).readlines())[0].split()
    return testProteinList  # ['kpcb_2i0eA_full', 'fabp4_2nnqA_full', ....]


# make a seq-contactMap dict
def getSeqContactDict(contactPath, contactDictPath):
    """Return a dictionary mapping protein hash sequences to contact maps."""
    contactDict = open(contactDictPath).readlines()
    seqContactDict = {}
    for data in contactDict:
        _, contactMapName = data.strip().split(':')
        seq, contactMap = getProtein(contactPath, contactMapName)
        contactmap_np = [list(map(float, x.strip(' ').split(' '))) for x in contactMap]
        feature2D = np.expand_dims(contactmap_np, axis=0)
        feature2D = torch.FloatTensor(feature2D)
        seqContactDict[seq] = feature2D
    return seqContactDict


def getLetters(path):
    """Return a list of the characters in a string loaded from 'path'."""
    with open(path, 'r') as f:
        chars = f.read().split()
    return chars


def getDataDict(testProteinList, activePath, decoyPath, contactPath):
    """Add docstring."""
    dataDict = {}
    for x in testProteinList:
        xData = []
        protein = x.split('_')[0]
        print(protein)
        proteinActPath = activePath + "/" + protein + "_actives_final.ism"
        proteinDecPath = decoyPath + "/" + protein + "_decoys_final.ism"
        act = open(proteinActPath, 'r').readlines()
        dec = open(proteinDecPath, 'r').readlines()
        actives = [[x.split(' ')[0], 1] for x in act]
        decoys = [[x.split(' ')[0], 0] for x in dec]  # test
        seq = getProtein(contactPath, x, contactMap=False)
        for i in range(len(actives)):
            xData.append([actives[i][0], seq, actives[i][1]])
        for i in range(len(decoys)):
            xData.append([decoys[i][0], seq, decoys[i][1]])
        print(len(xData))
        dataDict[x] = xData
    return dataDict

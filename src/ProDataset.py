"""A PyTorch-compatible dataset representing observations."""

from torch.utils.data import Dataset
from model import *
from utils import *
from warnings import filterwarnings
from torch.utils.data import DataLoader


class ProDataset(Dataset):
    """
    A PyTorch-compatible dataset representing observations.

    Each observation is of the form (SMILES string sequence, protein contact
    map tensor, response), where response is either 0 or 1.
    """

    def __init__(self, dataSet, seqContactDict):
        # dataSet is a list of lists of the form [smile, seq, label]. "seq"
        # refers to the hash-sequence corresponding to the contact map
        # corresponding to the observation's protein.
        self.dataSet = dataSet
        # A dictionary that maps hash sequences to contact maps.
        self.dict = seqContactDict
        # The number of observations in the data set.
        self.len = len(dataSet)
        # The labels corresponding to each observation in the data set.
        properties = [int(x[2]) for x in dataSet]
        # The labels' unique values; for binary classification, this should be
        # [0, 1].
        self.property_list = list(sorted(set(properties)))

    def __getitem__(self, index):
        """Return the observation corresponding to the given index."""
        smiles, seq, label = self.dataSet[index]
        # Get the contact map of the protein associated with the current
        # observation.
        contactMap = self.dict[seq]
        return smiles, contactMap, int(label)

    def __len__(self):
        """Return the number of observations in the associated data set."""
        return self.len

    def get_properties(self):
        """Return the labels' unique values in a sorted list."""
        return self.property_list

    def get_property_id(self, property):
        """Get the index of the given label in the sorted property list."""
        return self.property_list.index(property)

    def get_property(self, id):
        """Return the label value corresponding to the given index."""
        return self.property_list[id]

testFoldPath = "./data/DUDE/dataPre/DUDE-foldTest3"
trainFoldPath = "./data/DUDE/dataPre/DUDE-foldTrain3"
contactPath = "./data/DUDE/contactMap"
contactDictPath = "./data/DUDE/dataPre/DUDE-contactDict"
smileLettersPath  = "./data/DUDE/voc/combinedVoc-wholeFour.voc"
seqLettersPath = "./data/DUDE/voc/sequence.voc"
# Get training data
trainDataSet = getTrainDataSet(trainFoldPath)
# Get sequence to contact map dictionary.
seqContactDict = getSeqContactDict(contactPath, contactDictPath)
# Get SMILES drug sequences.
smiles_letters = getLetters(smileLettersPath)
sequence_letters = getLetters(seqLettersPath)


# testProteinList = getTestProteinList(testFoldPath)# whole foldTest
testProteinList = ["tryb1_2zebA_full", "mcr_2oaxE_full", "cxcr4_3oduA_full"]
DECOY_PATH = "./data/DUDE/decoy_smile"
ACTIVE_PATH = "./data/DUDE/active_smile"
print("get protein-seq dict....")
dataDict = getDataDict(testProteinList, ACTIVE_PATH, DECOY_PATH, contactPath)

N_CHARS_SMI = len(smiles_letters)
N_CHARS_SEQ = len(sequence_letters)

train_dataset = ProDataset(dataSet=trainDataSet, seqContactDict=seqContactDict)
train_loader = DataLoader(dataset=train_dataset, batch_size=1, shuffle=True, drop_last=True)

modelArgs = {}
modelArgs["batch_size"] = 1
modelArgs["lstm_hid_dim"] = 64
modelArgs["d_a"] = 32
modelArgs["r"] = 10
modelArgs["n_chars_smi"] = 247
modelArgs["n_chars_seq"] = 21
modelArgs["dropout"] = 0.2
modelArgs["in_channels"] = 8
modelArgs["cnn_channels"] = 32
modelArgs["cnn_layers"] = 4
modelArgs["emb_dim"] = 30
modelArgs["dense_hid"] = 64
modelArgs["task_type"] = 0
modelArgs["n_classes"] = 1

trainArgs = {}
trainArgs["model"] = DrugVQA(modelArgs, block=ResidualBlock).cuda()
trainArgs["epochs"] = 30
trainArgs["lr"] = 0.0007
trainArgs["train_loader"] = train_loader
trainArgs["doTest"] = True
trainArgs["test_proteins"] = testProteinList
trainArgs["testDataDict"] = dataDict
trainArgs["seqContactDict"] = seqContactDict
trainArgs["use_regularizer"] = False
trainArgs["penal_coeff"] = 0.03
trainArgs["clip"] = True
trainArgs["criterion"] = torch.nn.BCELoss()
trainArgs["optimizer"] = torch.optim.Adam(trainArgs["model"].parameters(), lr=trainArgs["lr"])
trainArgs["doSave"] = True
trainArgs["saveNamePre"] = "DUDE30Res-fold3-"
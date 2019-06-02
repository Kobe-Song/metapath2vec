from utils import AliasTable
import pickle
from collections import Counter
import numpy as np

class DataLoader:

    def __init__(self):

        self.context_window = 1
        self.sequences = []

        self.node2id = {}
        self.id2node = {}

        self.nid2type = {}
        self.type2nidlist = {}

        self.nid2freq = {}
        # self.nid2dist = {}
        # self.type2dist = {}

        self.node_alias = []
        self.node_type_alias = {}

        self.node_pair = []

        
    def load_graph(self):
        '''
        load all node and generate an id for each node
        generate node type dict
        '''
        
        # update node2id and id2node
        with open('data/node2id.pkl', 'rb') as handle:
            self.node2id = pickle.load(handle)
        with open('data/id2node.pkl', 'rb') as handle:
            self.id2node = pickle.load(handle)
        with open('data/nid2type.pkl', 'rb') as handle:
            self.nid2type = pickle.load(handle)
        with open('data/type2nidlist.pkl', 'rb') as handle:
            self.type2nidlist = pickle.load(handle)

        return

    def load_sequence(self, seuqence_file):
        '''
        load meta path sequence,
        convert all node to node_id
        '''
        sequences = []
        self.nid2freq = dict.fromkeys(self.id2node.keys(), 0)
        with open(seuqence_file, 'r') as f:
            for line in f:
                sequence = []
                for n in line.strip().split():
                    nid = self.node2id[n]
                    sequence.append(nid)
                    self.nid2freq[nid] += 1
                sequences.append(sequence)

        self.sequences = sequences
        return

    def construct_distribution(self):
        '''
        construct node distribution for sampling
        '''
        # get node frequency distribution
        # self.nid2freq = Counter(sequences)

        # here, we keep the same index with node id

        node_prob = np.power(np.array(list(self.nid2freq.values())), 0.75)
        node_prob /= np.sum(node_prob)
        self.node_alias = AliasTable(node_prob)
        
        # get node frequency distribution for each type
        for t in self.type2nidlist:
            self.node_type_alias[t] = AliasTable(node_prob[self.type2nidlist[t]])

        return
    
    def generate_positive_samples(self):
        '''
        generate center node, context node
        '''
        # positive
        for sequence in self.sequences:
            
            seq_len = len(sequence)
            for cur in range(seq_len):
                # center node
                center = sequence[cur]

                # context node
                start = max(0, cur - self.context_window)
                end = min(seq_len, cur + self.context_window + 1)
                context = sequence[start:cur] + sequence[cur+1:end]
                self.node_pair.extend((center, c) for c in context)

        return
    
    def generate_negative_samples(self, pos_id, neg_num, is_heterogeneous = True):
        '''
        generate negative based on center and context node
        '''
        neg_list = []
        # consider heterogeneous

        if is_heterogeneous:
            node_type = self.nid2type[pos_id]
            for i in range(neg_num):
                while True:
                    neg_index = self.node_type_alias[node_type].sampling()
                    neg_node = self.type2nidlist[node_type][neg_index]
                    if neg_node != pos_id:
                        neg_list.append(neg_node)
                        break
        # not consider heterogeneous
        else:
            for i in range(neg_num):
                while True:
                    neg_node = self.node_alias.sampling()
                    if neg_node != pos_id:
                        neg_list.append(neg_node)
                        break

        return neg_list

    def get_sample_size(self):
        '''
        get size of all node pairs (samples)
        '''
        return len(self.node_pair)

    def get_node_size(self):
        '''
        get size of all node
        '''
        return len(self.node2id)

    def get_one_sample(self, count):
        '''
        get one sample for train
        '''

        return self.node_pair[count]
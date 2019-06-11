from collections import defaultdict
import numpy as np
import pickle
from utils import AliasTable
import multiprocessing as mp
import argparse

class DataGenerator:
    def __init__(self, num_walks, walk_length):
        # user - account

        self.user_account = defaultdict(list)
        self.account_user = defaultdict(list)
        self.u2a_weight = defaultdict(list)
        self.a2u_weight = defaultdict(list)
        self.user_alias = defaultdict(list)
        self.account_alias = defaultdict(list)

        self.node2id = {}
        self.id2node = {}
        self.nid2type = {}
        self.type2nidlist = defaultdict(list)

        self.num_walks = num_walks
        self.walk_length = walk_length

    def load_data(self, data_file):
        '''
        load original data
        '''
        with open(data_file, 'r') as f:
            f.readline()
            for line in f:
                lines = line.strip().split(',')
                self.user_account[lines[0]].append(lines[1])
                self.account_user[lines[1]].append(lines[0])
                self.u2a_weight[lines[0]].append(int(lines[2]))
                self.a2u_weight[lines[1]].append(int(lines[2]))

        nid = 0
        for user in self.user_account.keys():
            self.node2id[user] = nid
            self.id2node[nid] = user
            self.nid2type[nid] = 'u'
            self.type2nidlist['u'].append(nid)
            nid += 1
        for account in self.account_user.keys():
            self.node2id[account] = nid
            self.id2node[nid] = account
            self.nid2type[nid] = 'a'
            self.type2nidlist['a'].append(nid)
            nid += 1


        # construct alias table for each node
        for user in self.u2a_weight:
            weight = np.array(self.u2a_weight[user])
            prob = weight / np.sum(weight)
            self.user_alias[user] = AliasTable(prob)
        for account in self.a2u_weight:
            weight = np.array(self.a2u_weight[account])
            prob = weight / np.sum(weight)
            self.account_alias[account] = AliasTable(prob)

        return
    
    def generate_uau_process(self, out_file):
        '''
        generate mata path by user-account-user
        '''
        meta_path_seq = []
        pool = mp.Pool()
        meta_path_seq.extend(pool.map(self.generate_uau, (user for user in self.user_account)))

        metapath_str = [''.join([n + '\t' for n in s]) + '\n' for u_seq in meta_path_seq for s in u_seq]
        
        with open(out_file, 'w') as f:
            f.writelines(metapath_str)

        return

    
    def generate_uau(self, user):
        '''
        generate mata path by user-account-user for a user
        '''
        user_seq = []
        # user_seq.append(user)
        for n in range(self.num_walks):
            seq = []
            seq.append(user)
            cur_user = user
            for w_len in range(self.walk_length):
                index_a = self.user_alias[cur_user].sampling()
                cur_a = self.user_account[cur_user][index_a]
                seq.append(cur_a)
                index_u = self.account_alias[cur_a].sampling()
                cur_user = self.account_user[cur_a][index_u]
                seq.append(cur_user)
            
            user_seq.append(seq)
        
        return user_seq

    def save_data(self):
        '''
        save dict of graph on disk
        '''
        with open('data/node2id.pkl', 'wb') as handle:
            pickle.dump(self.node2id, handle)
        with open('data/id2node.pkl', 'wb') as handle:
            pickle.dump(self.id2node, handle)
        with open('data/nid2type.pkl', 'wb') as handle:
            pickle.dump(self.nid2type, handle)
        with open('data/type2nidlist.pkl', 'wb') as handle:
            pickle.dump(self.type2nidlist, handle)

        return

def parse_args():
	#Parses the arguments.
    parser = argparse.ArgumentParser(description="metapath sequence generator")
    parser.add_argument('--num_walks',type=int,default=5, help='Number of walk for each node')
    parser.add_argument('--walk_length',type=int,default=5, help='length of sequence for each walk')
    parser.add_argument('--data_file',type=str,default='test.txt', help='input graph file')
    parser.add_argument('--out_file',type=str,default='metapath.txt', help='output metapath sequence file')

    
    return parser.parse_args()

def main(args):
    generator = DataGenerator(args.num_walks, args.walk_length)
    generator.load_data(args.data_file)
    generator.generate_uau_process(args.out_file)
    generator.save_data()

if __name__ == "__main__":
    args = parse_args()
    main(args)
from collections import defaultdict
import numpy as np
import pickle
from utils import AliasTable

class DataGenerator:
    def __init__(self):
        # user - account
        # self.user2id = {}
        # self.id2user = {}
        # self.account2id = {}
        # self.id2account = {}
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
            # for uid, user in enumerate(self.user_account.keys()):
            #     self.user2id[user] = uid
            #     self.id2user[uid] = user
            # for aid, account in enumerate(self.account_user.keys()):
            #     self.account2id[account] = aid
            #     self.id2account[aid] = account
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
    
    def generate_uau(self, out_file, num_walks, walk_length):
        '''
        generate mata path by user-account-user
        '''
        with open(out_file, 'w') as f:
            for user in self.user_account:
                for n in range(num_walks):
                    cur_user = user
                    f.write(cur_user)
                    for w_len in range(walk_length):
                        index_a = self.user_alias[cur_user].sampling()
                        cur_a = self.user_account[cur_user][index_a]
                        index_u = self.account_alias[cur_a].sampling()
                        cur_user = self.account_user[cur_a][index_u]
                        f.write('\t' + cur_a + '\t' + cur_user)
                    f.write('\n')

        return

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



def main():
    data_file = 'test.txt'
    out_file = 'metapath.txt'
    num_walks = 2
    walk_length = 2

    generator = DataGenerator()
    generator.load_data(data_file)
    generator.generate_uau(out_file, num_walks, walk_length)
    generator.save_data()

if __name__ == "__main__":
    main()
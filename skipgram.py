import numpy as np

MAX_EXP = 6
EXP_TABLE_SIZE = 1000

class SkipGram:

    def __init__(self):

        self.node_embedding = []
        self.alpha = 0.001
        self.exp_table = []

    def initialize(self, node_size, dim):
        self.node_embedding = np.random.uniform(-0.5, 0.5, [node_size, dim])

        # initialize exp table
        for i in range(EXP_TABLE_SIZE + 1):
            ex = np.exp((i / EXP_TABLE_SIZE * 2 - 1) - MAX_EXP)
            self.exp_table.append(ex / (ex + 1))

        return

    def train_process(self, epochs, data_loader, neg_num):

        sample_size = data_loader.get_sample_size()
        for i in range(epochs):
            # loss = 0
            for j in range(sample_size):
                center, context = data_loader.get_one_sample(j)
                neg_list = data_loader.generate_negative_samples(context, neg_num)
                self.train(center, context, neg_list, neg_num)
                # loss += self.train(center, context, neg_list, neg_num)
            # print('epoch: ', i, loss)



        return

    def train(self, center, context, neg_list, neg_num):
        '''
        calculate loss
        '''
        self.node_embedding[center]

        context_list = [context] + neg_list

        label = np.array([1] + [0]*neg_num)
        context_emb = self.node_embedding[context_list]
        center_emb = self.node_embedding[center]

        ###### calculate loss, only if you wanna check the loss
        # pos = self.node_embedding[context].dot(center_emb)
        # neg = self.node_embedding[neg_list].dot(center_emb)
        # loss = np.log(self.sigmoid(pos)) + np.log(self.sigmoid(-neg))

        product = context_emb.dot(center_emb)
        product = self.sigmoid(product)
        
        ########### update center node
        # gradient for center node
        center_grad = (label - product).reshape((neg_num+1, -1)) * context_emb
        center_grad = np.sum(center_grad, axis=0)
        self.node_embedding[center] = center_emb + self.alpha * center_grad


        ########### update all context node
        # gradient for context node
        context_grad = (label - product).reshape((-1, 1)) * center_emb
        self.node_embedding[context_list] = context_emb + self.alpha * context_grad

        # return loss   # while calculating loss
        return
    

    def sigmoid(self, x):
        '''
        fast calculate sigmoid 
        '''
        if isinstance(x, np.float):
            x = [x]
        for i in range(len(x)):
            if x[i] > MAX_EXP:
                x[i] = 0.999999
            elif x[i] < -MAX_EXP:
                x[i] = -0.999999
            else:
                x[i] = self.exp_table[int((x[i] + MAX_EXP)/(2*MAX_EXP)*EXP_TABLE_SIZE)]

        return x
    
    def save_emb(self, emb_file):
        '''
        save node embedding on disk
        '''
        index = np.arange(len(self.node_embedding)).reshape((-1, 1))
        embedding = np.hstack([index, self.node_embedding])
        embedding_list = embedding.tolist()
        embedding_str = [str(emb[0]) + '\t' + ''.join([str(round(x, 6)) + '\t' for x in emb[1:]]) for emb in embedding_list]
        with open(emb_file, 'w') as f:
            f.writelines(embedding_str)

        return
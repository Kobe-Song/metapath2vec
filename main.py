import argparse
from data_loader import DataLoader
from skipgram import SkipGram


def parse_args():
	#Parses the arguments.
    parser = argparse.ArgumentParser(description="metapath2vec")
    parser.add_argument('--epochs',type=int,default=1, help='Number of epochs')
    parser.add_argument('--lr',type=float,default=0.001, help='learning rate')

    parser.add_argument('--emb_dim',default=32,type=int,help='embedding dimensions')
    parser.add_argument('--neg_num',default=5,type=int,help='number of negative samples')
    parser.add_argument('--edge_type',default=1,type=int,help='care type or not. if 1, it cares (i.e. heterogeneous negative sampling). If 0, it does not care (i.e. normal negative sampling). ')
    parser.add_argument('--window_size',default=1,type=int,help='context window size')

    parser.add_argument('--seq_file',default='metapath.txt',type=str)
    # parser.add_argument('--walks',type=str,required=True,help='text file that has a random walk in each line. A random walk is just a seaquence of node ids separated by a space.')
	# parser.add_argument('--types',type=str,required=True,help='text file that has node types. each line is "node id <space> node type"')
    # parser.add_argument('--batch',type=int,default=1, help='Batch size.Only batch one is supported now...')
    # parser.add_argument('--log',required=True,type=str,help='log directory')
	# parser.add_argument('--log-interval',default=-1,type=int,help='log intervals. -1 means per epoch')
	# parser.add_argument('--max-keep-model',default=10,type=int,help='number of models to keep saving')
    return parser.parse_args()

def main(args):
    data_loader = DataLoader()
    data_loader.load_graph()
    data_loader.load_sequence(args.seq_file)
    data_loader.construct_distribution()
    data_loader.generate_positive_samples()
    # data_loader.generate_negative_samples(3,1)

    model = SkipGram()
    model.initialize(data_loader.get_node_size(), args.emb_dim)
    model.train_process(args.epochs, data_loader, args.neg_num)
    



if __name__ == "__main__":
    args = parse_args()
    main(args)
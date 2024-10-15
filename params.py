import argparse

def parse_args():
	parser = argparse.ArgumentParser(description='Model Hyperparameters')

	parser.add_argument('--emb_size', default=64, type=int, help='embedding size')
	parser.add_argument('--temp_size', default=16, type=int, help='temporal size')
	parser.add_argument('--n_layers', default=3, type=int, help='gnn layers')
	parser.add_argument('--dataset', default='Tmall', type=str, help='name of dataset')
	
	parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
	parser.add_argument('--batch', default=2048, type=int, help='batch size')
	parser.add_argument('--l2_reg', default=0.01, type=float, help='weight decay regularizer')
	parser.add_argument('--ssl_reg', default=0.01, type=float, help='scale of InfoNCE Loss')
	parser.add_argument('--club_reg', default=0.01, type=float, help='scale of CLUB Loss')
	parser.add_argument('--club_train_step', default=5, type=int, help='CLUB train step')
	parser.add_argument('--epochs', default=600, type=int, help='number of epochs')
	parser.add_argument('--topks', default=10, type=int, help='@k test list')
	
	parser.add_argument('--wandb', default=1, type=int, help="enable wandb")
	parser.add_argument('--data_dir', default='./data/', type=str, help='dataset directory')
	parser.add_argument('--target', default='buy', type=str, help='target behavior')
	parser.add_argument('--save_model', default=1, type=int, help='whether to save model')
	parser.add_argument('--param_path', default='./checkpoints/', type=str, help="checkpoint path")
	return parser.parse_args()
args = parse_args()

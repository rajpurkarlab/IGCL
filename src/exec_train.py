import sys
sys.path.append('./training')

import argparse
from attrdict import AttrDict
from train import load_data, load_clip, make, train, save

parser = argparse.ArgumentParser(description='Process parameters to be used during training.')

# File/Directory locations
parser.add_argument("--graph_root", type=str, 
                help="Location of graph-related dataset files.")
parser.add_argument("--raw_image_path", type=str, 
                help="Location of raw image dataset files.")
parser.add_argument("--processed_image_path", type=str, 
                help="Location of processed image dataset files.")
parser.add_argument("--processed_graph_path", type=str, 
                help="Location of processed graph dataset files.")
parser.add_argument("--model_root", type=str, 
                help="Location where pretrained models will be/are saved.")

# Training hyperparams
parser.add_argument("--optimizer", type=str, default="sgd",
                help="The optimizer to use for contrastive learning.")
parser.add_argument('--use_pool', action="store_true",
                help='Whether or not to use pooling in DRACON model.')
parser.add_argument('--batch_size', type=int, default=32,
                help='Batch size to be used for contrastive learning.')
parser.add_argument('--epochs', type=int, default=8,
                help='Number of epochs to train model for contrastive learning.')
parser.add_argument('--graph_transform', type=str, default="reaction",
                help='Graph transformation type used.')
parser.add_argument('--image_freeze_interval', type=int, default=0,
                help='How many batches to freeze image encoder learning for. (See Locked-image text Tuning paper for details.)')
parser.add_argument('--lr', type=float, default=1e-4,
                help='Learning rate to be used for contrastive learning.')
parser.add_argument('--momentum', type=float, default=1e-4,
                help='Momentum to be used for contrastive learning.')

# Architecture hyperparams
parser.add_argument("--graph_architecture", type=str, default="DRACON", 
                help="Graph architecture to utilize for evaluation.")
parser.add_argument('--attn_heads', type=int, default=1,
                help='Number of attention heads. DRACON-only operator.')
parser.add_argument('--use_rgcn', action="store_true",
                help='Whether or not to use RGCN in DRACON model for convolution.')
parser.add_argument('--trans_layers', type=int, default=1,
                help='Number of transformer layers to be used in DRACON architecture.')
parser.add_argument('--fc_layers', type=int, default=1,
                help='Number of fully-connected layers to be used in DRACON architecture.')
parser.add_argument('--graph_layers', type=int, default=2,
                help='Number of graph convolution layers to be used in DRACON architecture.')
parser.add_argument('--graph_hidden', type=int, default=512,
                help='Dimension of final enocoding from graph encoder.')
parser.add_argument('--num_runs', type=int, default=10,
                help='Number of model runs to be evaluated.')

def execute_cl_training():
	args = parser.parse_args()

	# Compile config dictionary based on commandline inputs.
	config = AttrDict(
	    batch_size=args.batch_size,
	    epochs=args.epochs,
	    optimizer=args.optimizer,
	    lr=args.lr,
	    momentum=args.momentum,
	    log_interval=200,
	    save_interval=200,
	    image_freeze_interval=args.image_freeze_interval,
	    model_name=f"ViT-{args.graph_architecture}",
	    model_root=args.model_root,
	    
	    pretrained=True,
	    node_features=772,
	    edge_features=4,
	    graph_layers=args.graph_layers,
	    graph_hidden=args.graph_hidden,

	    trans_layers=args.trans_layers,
	    fc_layers=args.fc_layers,
	    attn_heads=args.attn_heads,
	    
	    graph_architecture=args.graph_architecture,
	    graph_transform=args.graph_transform,
	    use_pool=args.use_pool,
	    dataloader='DataLoader',
	)

	for index in range(args.num_runs):
		model, train_data_loader, val_data_loader, device, criterion, optimizer = make(
		              config, "train", args.graph_root, args.raw_image_path,
		              args.processed_graph_path, args.processed_image_path, model_path=None
		)

		train(args.model_root, model, train_data_loader, val_data_loader, device,
		              criterion, optimizer, config, index)

if __name__ == "__main__":
	execute_cl_training()

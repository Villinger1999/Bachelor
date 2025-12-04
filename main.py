import sys 
num_clients = int(sys.argv[1]) # e.g 3
num_rounds = int(sys.argv[2]) # e.g 2
local_epochs = int(sys.argv[3]) # e.g 1
batch_size = int(sys.argv[4]) # e.g 24
C = int(sys.argv[5]) # e.g 1

# # Build list with validation and train set paths
# val = glob.glob(os.path.join(VAL_IMG_DIR, '*.JPEG')) 
# train = glob.glob(os.path.join(TRAIN_IMG_DIR, '**', '*.JPEG'), recursive=True)
from utils_data import DLoader
from utils_func import collect_all_pairs, make_dataset_ids
from tokenizer import Tokenizer
from config import Config

import pickle
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

def make_dataset(config:Config, datanum):
    img_folder = config.data_path + 'images/'
    caption_file = config.datapath + 'captions.txt'
    all_pairs = collect_all_pairs(caption_file)

    # transforms
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    trans = transforms.Compose([transforms.Resize((self.img_size, self.img_size)), 
            transforms.ToTensor(),
            transforms.Normalize(mean, std)])

    # datanum만큼 선택
    all_pairs = all_pairs[:datanum]

    trainset_id, val_set_id = make_dataset_ids(len(all_pairs), datanum // 5)
    tokenizer = Tokenizer(config, all_pairs, trainset_id)

    # make train set
    trainset = DLoader(img_folder, all_pairs, trans, trainset_id, tokenizer, config.max_len)
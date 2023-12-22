import json
import os.path as op
from .common import qd_tqdm as tqdm
from .common import json_dump
from .common import pilimg_from_base64
from .common import get_mpi_rank, get_mpi_size, get_mpi_local_rank

from .tsv_io import TSVFile, tsv_writer, tsv_reader
from .common import write_to_file
import torch
import PIL
from pprint import pformat
import logging
from transformers import BertTokenizer
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from PIL import Image
from azfuse import File
from torch.utils.data import DataLoader, Dataset

from .common import init_logging
from .common import parse_general_args
from .tsv_io import load_from_yaml_file
from .torch_common import torch_load
from .torch_common import load_state_dict
from .process_image import load_image_by_pil
from .model import get_git_model
import pytorch_lightning as pl
import random
def get_data(image_file,prefix,target,tokenizer,image_transform):
    max_text_len=40
    prefix_encoding = tokenizer(
        prefix, padding='do_not_pad',
        add_special_tokens=False,
        truncation=True, max_length=max_text_len)
    target_encoding = tokenizer(
        target, padding='do_not_pad',
        add_special_tokens=False,
        truncation=True, max_length=max_text_len)
    need_predict=[0]*len(prefix_encoding['input_ids'])+[1]*len(target_encoding['input_ids'])
    payload=prefix_encoding['input_ids']+target_encoding['input_ids']
    if len(payload)>max_text_len:
        payload = payload[-(max_text_len - 2):]
        need_predict = need_predict[-(max_text_len - 2):]
    input_ids = [tokenizer.cls_token_id] + payload + [tokenizer.sep_token_id]
    need_predict = [0] + need_predict + [1]
    img = [load_image_by_pil(i) for i in image_file]
    img = [image_transform(i) for i in img]
    image_file=random.choices(image_file, k=2)
    # img=[interpolate(i, size=(100,150), scale_factor=None, mode='nearest', align_corners=None, recompute_scale_factor=None, antialias=False) for i in img]

    data = {
        'caption_tokens': torch.tensor(input_ids),
        'need_predict': torch.tensor(need_predict),
        'image': img,
        'caption': {},
        'iteration': 0,
    }

class dataset(Dataset):
    def __init__(self,data,tokenzier,image_transform):
        self.data=data
        self.tokenizer=tokenzier
        self.image_transform=image_transform
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self,idx):
        image=[]
        for i in range(len(self.data[idx]['image_path'])):
            image.append(op.join('dataset_path', self.data[idx]['image_path'][i]))
        question=self.data[idx]['question']
        answer=self.data[idx]['answer']
        qid=(self.data[idx]['questionId'])
        return get_data(image,question,answer,self.tokenizer,self.image_transform)




class DataModule(pl.LightningDataModule):
    def __init__(self, train_dataset=None, val_dataset=None,batch_size=32):
        super(DataModule, self).__init__()
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.predict_dataset=val_dataset
        self.batch_size = batch_size

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

class MinMaxResizeForTest(object):
    def __init__(self, min_size, max_size):
        self.min_size = min_size
        self.max_size = max_size

    def get_size(self, image_size):
        w, h = image_size
        size = self.min_size
        max_size = self.max_size

        min_original_size = float(min((w, h)))
        max_original_size = float(max((w, h)))
        if max_original_size / min_original_size * size > max_size:
            size = int(round(max_size * min_original_size / max_original_size))

        if (w <= h and w == size) or (h <= w and h == size):
            return (h, w)

        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)

        return (oh, ow)

    def __repr__(self):
        return 'MinMaxResizeForTest({}, {})'.format(
            self.min_size, self.max_size)

    def __call__(self, img):
        size = self.get_size(img.size)
        import torchvision.transforms.functional as F
        image = F.resize(img, size, interpolation=PIL.Image.BICUBIC)
        return image
    
def get_image_transform(param):
    crop_size = param.get('test_crop_size', 224)
    if 'test_respect_ratio_max' in param:
        trans = [
            MinMaxResizeForTest(crop_size, param['test_respect_ratio_max'])
        ]
    else:
        trans = [
            Resize(crop_size, interpolation=Image.BICUBIC),
            CenterCrop(crop_size),
            lambda image: image.convert("RGB"),

        ]
    trans.extend([
        ToTensor(),
        Normalize(
            (0.48145466, 0.4578275, 0.40821073),
            (0.26862954, 0.26130258, 0.27577711),
        ),
    ])
    transforms = Compose(trans)
    return transforms

def test_git_inference_single_image(image_path,model_name,prefix):
    param={}
    if File.isfile(f'aux_data/models/{model_name}/parameter.yaml'):
            param = load_from_yaml_file(f'aux_data/models/{model_name}/parameter.yaml')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    transforms=get_image_transform(param)
    model =get_git_model(tokenizer,param)
    pretrained=f'output/{model_name}/snapshot/model.pt'
    checkpoint = torch_load(pretrained)['model']
    load_state_dict(model,checkpoint)
    train_path="../newsvqa/git1/aux_data/new_train.json"
    train_data = json.load(open(train_path))
    val_path ="./news_val11.json"
    val_data = json.load(open(val_path))
    train_ds = dataset(train_data, tokenizer,transforms)
    val_ds = dataset(val_data, tokenizer,transforms)
    learning_rate = 1e-2
    max_steps = 1000000
    batch_size=32
    datamodule = DataModule(train_ds, val_ds, batch_size)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(dirpath="./textvqa1/",filename='git_allgpt_{epoch}')
    trainer = pl.Trainer(accelerator="gpu", max_epochs=52,callbacks=[checkpoint_callback],devices=1)
    trainer.fit(model, datamodule)

if __name__=="__main__":
    init_logging()
    kwargs=parse_general_args()
    logging.info('param:\n{}'.format(pformat(kwargs)))
    function_name=kwargs['type']
    del kwargs['type']
    locals()[function_name](**kwargs)

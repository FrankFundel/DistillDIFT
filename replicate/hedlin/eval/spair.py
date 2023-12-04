r"""SPair-71k dataset"""
import json
import glob
import os

from PIL import Image
import numpy as np
import torch

from .dataset import CorrespondenceDataset, random_crop


class SPairDataset(CorrespondenceDataset):
    r"""Inherits CorrespondenceDataset"""
    def __init__(self, benchmark, datapath, thres, device, split, augmentation, feature_size, sub_class="all", item_index=-1):
        r"""SPair-71k dataset constructor"""
        super(SPairDataset, self).__init__(benchmark, datapath, thres, device, split, augmentation, feature_size)

        self.train_data = open(self.spt_path).read().split('\n')
        self.train_data = self.train_data[:len(self.train_data) - 1]
        
        if sub_class != "all":
            
            new_data = []
            for i in range(len(self.train_data)):
                if sub_class in self.train_data[i]:
                    new_data.append(self.train_data[i])
                    
            self.train_data = new_data
            
        if item_index != -1:
            self.train_data = [self.train_data[item_index]]
                    
        
        self.src_imnames = list(map(lambda x: x.split('-')[1] + '.jpg', self.train_data))
        self.trg_imnames = list(map(lambda x: x.split('-')[2].split(':')[0] + '.jpg', self.train_data))
        self.cls = os.listdir(self.img_path)
        self.cls.sort()
        
        # import ipdb; ipdb.set_trace()

        anntn_files = []
        for data_name in self.train_data:
            anntn_files.append(glob.glob('%s/%s.json' % (self.ann_path, data_name))[0])
        anntn_files = list(map(lambda x: json.load(open(x)), anntn_files)) 
        self.src_kps = list(map(lambda x: torch.tensor(x['src_kps']).t().float(), anntn_files))
        self.trg_kps = list(map(lambda x: torch.tensor(x['trg_kps']).t().float(), anntn_files))
        self.src_bbox = list(map(lambda x: torch.tensor(x['src_bndbox']).float(), anntn_files))
        self.trg_bbox = list(map(lambda x: torch.tensor(x['trg_bndbox']).float(), anntn_files))
        self.cls_ids = list(map(lambda x: self.cls.index(x['category']), anntn_files))

        self.vpvar = list(map(lambda x: torch.tensor(x['viewpoint_variation']), anntn_files))
        self.scvar = list(map(lambda x: torch.tensor(x['scale_variation']), anntn_files))
        self.trncn = list(map(lambda x: torch.tensor(x['truncation']), anntn_files))
        self.occln = list(map(lambda x: torch.tensor(x['occlusion']), anntn_files))

    def __getitem__(self, idx):
        r"""Constructs and return a batch for SPair-71k dataset"""
        batch = super(SPairDataset, self).__getitem__(idx)

        if self.split == 'trn' and self.augmentation:
            batch['src_img'], batch['src_kps'] = random_crop(batch['src_img'], batch['src_kps'], self.src_bbox[idx].clone(), size=(self.imside,)*2)
            batch['trg_img'], batch['trg_kps'] = random_crop(batch['trg_img'], batch['trg_kps'], self.trg_bbox[idx].clone(), size=(self.imside,)*2)

        
        batch['src_bbox'] = self.get_bbox(self.src_bbox, idx, batch['src_imsize'])
        batch['trg_bbox'] = self.get_bbox(self.trg_bbox, idx, batch['trg_imsize'])
        batch['pckthres'] = self.get_pckthres(batch, batch['trg_imsize'])

        batch['vpvar'] = self.vpvar[idx]
        batch['scvar'] = self.scvar[idx]
        batch['trncn'] = self.trncn[idx]
        batch['occln'] = self.occln[idx]
        
        batch['idx'] = idx

        return batch
    
    def collect_results(self):
        # create a dict to store the results
        # the keys will be each item in self.cls
        # the values will be initialized to empty lists
        results = {cls: [] for cls in self.cls}
        
        from glob import glob
        # files = sorted(glob("spair_results/results_unflipped/*"))
        files = sorted(glob("spair_results/results_unflipped/*/pck_array_*.txt"))
        
        for file_name in files:
            file_number = int(file_name.split("/")[-2])
            
            data_name = self.train_data[file_number]
            obj_name = data_name.split(":")[-1]
            
            assert obj_name in results.keys()
            
            # open file and append the value to the list
            # file = torch.load(file_name)
            # performance = float(file['pck'][0])
            # load the text file
            performance = np.loadtxt(file_name)[0]

            
            results[obj_name].append(performance)
            
        return results

    def get_image(self, img_names, idx):
        r"""Returns image tensor"""
        path = os.path.join(self.img_path, self.cls[self.cls_ids[idx]], img_names[idx])

        return Image.open(path).convert('RGB')

    def get_bbox(self, bbox_list, idx, imsize):
        r"""Returns object bounding-box"""
        bbox = bbox_list[idx].clone()
        bbox[0::2] *= (self.imside / imsize[0])
        bbox[1::2] *= (self.imside / imsize[1])
        return bbox

import csv
import os
import timm
import torch
from timm.data import ImageDataset
from timm.data.parsers.parser import Parser
from pathlib import Path
from PIL import Image


class AUCDDParser(Parser):
    def __init__(self, root, img_list):
        super(AUCDDParser, self).__init__()

        self.root = Path(root)
        self.classes = []
        self.samples = []

        with open(img_list, 'r') as f:
            csvreader = csv.reader(f)
            fields = next(csvreader)
            for row in csvreader:
                split, cls, img = row
                self.classes.append(cls)
                img_path = self.root / split / cls / img
                self.samples.append(img_path)

        classes = sorted(set(self.classes), key=lambda s: s.lower())
        self.class_to_idx = {c: idx for idx, c in enumerate(classes)}

    def __getitem__(self, index):
        path = self.samples[index]
        target = self.class_to_idx[self.classes[index]]
        return open(path, 'rb'), target

    def __len__(self):
        return len(self.samples)

    def _filename(self, index, basename=False, absolute=False):
        filename = self.samples[index]  # absolute path

        if basename:
            filename = filename.parts[-1]
        elif not absolute:
            filename = filename.relative_to(self.root)

        return filename


class AUCDDMTLParser(AUCDDParser):
    def __init__(self, root, img_list, emo_path):
        super(AUCDDMTLParser, self).__init__(root, img_list)

        emo_list = str(emo_path / "emo_list.csv")
        emo_label_dict = {}
        with open(emo_list, 'r') as f:
            csvreader = csv.reader(f)
            for row in csvreader:
                img_rel_path, emo_label = row
                emo_label_dict[img_rel_path] = int(emo_label)

        self.emo_labels = []
        self.face_images = []
        for path in self.samples:
            path = str(path)
            split, cls, image_name = path.split('/')[-3:]
            img_id = split + '/' + cls + '/' + image_name
            emo_label = emo_label_dict[img_id]
            self.emo_labels.append(emo_label)
            if emo_label == -1:
                self.face_images.append("")
            else:
                image_name = image_name.split('.')[0]
                face_img_path = emo_path / 'imgs' / cls / f"{image_name}_face.jpg"
                assert face_img_path.exists(), "face image does not exist"
                self.face_images.append(face_img_path)

    def __getitem__(self, index):
        face_img_path = self.face_images[index]
        emo_target = self.emo_labels[index]
        face_img = None
        if face_img_path:
            face_img = open(face_img_path, 'rb')

        return *super(AUCDDMTLParser, self).__getitem__(index), face_img, emo_target

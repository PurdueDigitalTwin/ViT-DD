import csv

from timm.data.parsers.parser import Parser
from pathlib import Path


class SFDDDParser(Parser):
    def __init__(self, root, img_list):
        super(SFDDDParser, self).__init__()

        self.root = Path(root)
        self.subjects = []
        self.classes = []
        self.samples = []

        with open(img_list, 'r') as f:
            csvreader = csv.reader(f)
            fields = next(csvreader)
            for row in csvreader:
                subject, cls, img = row
                self.subjects.append(subject)
                self.classes.append(cls)
                img_path = self.root / "imgs" / "train" / cls / img
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


class SFDDDMTLParser(SFDDDParser):
    def __init__(self, root, img_list, emo_path):
        super(SFDDDMTLParser, self).__init__(root, img_list)

        emo_list = str(emo_path / "emo_list.csv")
        emo_label_dict = {}
        with open(emo_list, 'r') as f:
            csvreader = csv.reader(f)
            for row in csvreader:
                img_rel_path, emo_label = row
                img_id = img_rel_path.split('/')[-1]
                emo_label_dict[img_id] = int(emo_label)

        self.emo_labels = []
        self.face_images = []
        for path in self.samples:
            path = str(path)
            img_id = path.split('/')[-1]
            emo_label = emo_label_dict[img_id]
            self.emo_labels.append(emo_label)
            if emo_label == -1:
                self.face_images.append("")
            else:
                img_id = img_id.split('.')[0]
                face_img_path = emo_path / 'imgs' / f"{img_id}_face.jpg"
                assert face_img_path.exists(), "face image does not exist"
                self.face_images.append(face_img_path)

    def __getitem__(self, index):
        face_img_path = self.face_images[index]
        emo_target = self.emo_labels[index]
        face_img = None
        if face_img_path:
            face_img = open(face_img_path, 'rb')

        return *super(SFDDDMTLParser, self).__getitem__(index), face_img, emo_target

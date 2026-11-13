"""Class order, inherited splits, label transfer, and display-only function contracts."""
import importlib.util
import copy
import tempfile
import json
import unittest
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from PIL import Image

from app.core.gestures import CLASSES, GESTURE_FUNCTIONS
from training.metrics import classification_average_precision
from training.prepare_dataset import fingerprint
from training.select_three_class import preservation_failures


class ThreeClassTests(unittest.TestCase):
    def test_display_only_mapping(self):
        self.assertEqual(CLASSES, ('closed_fist', 'open_palm', 'thumbs_up'))
        self.assertEqual(GESTURE_FUNCTIONS, {'open_palm':'STOP ROBOT', 'thumbs_up':'DONE / RESUME',
                                            'closed_fist':'RETURN TO BASE'})

    def test_three_class_data_and_inherited_splits(self):
        path=Path('data/three_class/manifest.json')
        if not path.exists():self.skipTest('Three-class data not prepared')
        data=json.loads(path.read_text())
        inherited=json.loads(Path(data['source_manifest']).read_text())
        old={r['processed_sha256']:r for r in inherited['manifest']}
        records=data['manifest'];groups={};hashes={}
        for row in records:
            digest,_=fingerprint(Path(row['processed']))
            self.assertEqual(digest,row['processed_sha256'])
            hashes.setdefault(digest,set()).add(row['split'])
            groups.setdefault(row['group'],set()).add(row['split'])
            if row['label']!='closed_fist':
                self.assertEqual((row['split'],row['group']), (old[digest]['split'],old[digest]['group']))
        self.assertTrue(all(len(s)==1 for s in [*groups.values(),*hashes.values()]))
        for split in ['train','val','test']:
            self.assertEqual({r['label'] for r in records if r['split']==split},set(CLASSES))
        for i,a in enumerate(records):
            for b in records[:i]:
                if a['split']!=b['split']:
                    self.assertGreater((a['phash']^b['phash']).bit_count(),8)
        self.assertEqual(data['closed_fist_unique_captures'],7)

    def test_average_precision_handles_tied_scores(self):
        result=classification_average_precision(['a','b','a','b'],[{'a':.5,'b':.5}]*4,['a','b'])
        self.assertEqual(result['mean_average_precision'],.5)
        perfect=classification_average_precision(['a','b'],[{'a':.9,'b':.1},{'a':.1,'b':.9}],['a','b'])
        self.assertEqual(perfect['mean_average_precision'],1.)

    def test_selection_blocks_legacy_regression_and_test_selection(self):
        classes={name:{'precision':1.,'recall':1.} for name in CLASSES}
        report=dict(split='val',manifest_sha256='same',confidence=.8,classes=list(CLASSES),unsupported_classes=[],
                    conditions={'clean':{'per_class':classes}},clean_views={'isolated':{'accuracy':1.}},
                    overall={'per_class':classes},stability={'stable_correct_after_warmup':1.})
        self.assertEqual(preservation_failures(report,report),[])
        regression=copy.deepcopy(report)
        regression['conditions']['clean']['per_class']['thumbs_up']['recall']=.9
        self.assertTrue(preservation_failures(report,regression))
        invalid=copy.deepcopy(report);invalid['split']='test'
        self.assertTrue(preservation_failures(report,invalid))

    @unittest.skipUnless(importlib.util.find_spec('torch'), 'PyTorch unavailable')
    def test_expansion_preserves_existing_class_rows_by_name(self):
        import torch
        from training.transforms import transfer_class_rows
        old=torch.nn.Linear(2,2);new=torch.nn.Linear(2,3)
        with torch.no_grad():
            old.weight.copy_(torch.tensor([[1.,2.],[3.,4.]]));old.bias.copy_(torch.tensor([5.,6.]))
        new_fist=new.weight[0].detach().clone()
        previous=SimpleNamespace(model=[SimpleNamespace(linear=old)],names={0:'open_palm',1:'thumbs_up'})
        target=SimpleNamespace(model=[SimpleNamespace(linear=new)])
        copied=transfer_class_rows(target,previous,dict(enumerate(CLASSES)))
        self.assertEqual(copied,['open_palm','thumbs_up'])
        torch.testing.assert_close(new.weight[1:],old.weight)
        torch.testing.assert_close(new.bias[1:],old.bias)
        torch.testing.assert_close(new.weight[0],new_fist)

    @unittest.skipUnless(importlib.util.find_spec('torch'), 'PyTorch unavailable')
    def test_training_only_class_balancing(self):
        from training.transforms import BalancedBackgroundGestureDataset
        from ultralytics.cfg import get_cfg
        args=get_cfg(overrides={'imgsz':320,'workers':0,'auto_augment':None,'erasing':0.})
        root=Path('data/three_class')
        if not root.exists():self.skipTest('Three-class data not prepared')
        train=BalancedBackgroundGestureDataset(root/'train',args,augment=True,prefix='train')
        validation=BalancedBackgroundGestureDataset(root/'val',args,augment=False,prefix='val')
        self.assertEqual(set(Counter(row[1] for row in train.samples).values()),{24})
        self.assertEqual(len(validation),13)

    @unittest.skipUnless(importlib.util.find_spec('torch'), 'PyTorch unavailable')
    def test_balancing_keeps_ram_cache_aligned(self):
        from training.transforms import BalancedBackgroundGestureDataset
        from ultralytics.cfg import get_cfg
        args=get_cfg(overrides={'imgsz':64,'cache':True,'auto_augment':None,'erasing':0.})
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            for count,label in enumerate(CLASSES,1):
                (root/label).mkdir()
                for index in range(count):
                    Image.new('RGB',(40,40),(40+index*10,70,100)).save(root/label/f'{index}.png')
            dataset=BalancedBackgroundGestureDataset(root,args,augment=True,prefix='train')
            self.assertEqual(len(dataset),9)
            self.assertEqual(len(dataset.img_cache),9)
            for index in range(len(dataset)):
                self.assertEqual(dataset[index]['cls'],dataset.samples[index][1])

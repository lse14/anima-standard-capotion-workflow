from __future__ import annotations
import json,sys,tempfile,unittest
from unittest.mock import patch
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'core'/'src'))
from anima_core import export_staging
from anima_core.export_staging import create_hardlink_staging,replace_business_annotation,reveal_staging
from anima_core.moss_sidecar import serialize_moss_json


def _ocr_sidecar(relative_image_path, *, status='success'):
 value={
  'schemaVersion':1,'relativeImagePath':relative_image_path,
  'image':{'width':2,'height':2,'sizeBytes':4,'sha256':'a'*64},'status':status,
  'engine':{'backend':'paddle','resourceId':'ocr-ppocrv5-server-paddle-v1','resourceFingerprint':'b'*64},
  'settings':{'llmMinConfidence':0.5,'inference':{'useDocOrientationClassify':False,'useDocUnwarping':False,'useTextlineOrientation':True,'textRecScoreThresh':0,'textDetLimitSideLen':1920,'textDetLimitType':'max'}},
  'items':[],'error':None,
 }
 if status=='success': value['items']=[{'index':0,'text':'Hello','confidence':0.9,'polygonPixels':[[0,0],[1,0],[1,1],[0,1]],'polygon':[[0,0],[0.5,0],[0.5,0.5],[0,0.5]],'bboxPixels':[0,0,1,1],'bbox':[0,0,0.5,0.5],'position':'top-left','textlineOrientationDegrees':0,'includedForLlm':True}]
 if status=='failed': value['error']={'code':'ocr_inference_failed','message':'OCR engine failed.','retriable':True}
 return json.dumps(value,separators=(',',':')).encode('utf-8')

class ExportStagingTests(unittest.TestCase):
 def test_moss_sidecar_requires_the_moss_json_schema(self):
  with tempfile.TemporaryDirectory() as t:
   root=Path(t);source=root/'dataset';source.mkdir();(source/'image.png').write_bytes(b'image');stage=create_hardlink_staging(source,root/'.dataset.stage')
   replace=getattr(export_staging,'replace_moss_sidecar',None)
   self.assertTrue(callable(replace),'MOSS staging replacement API is missing')
   value={'caption':{'long':'A cat.','medium':'A cat.','short':'A cat.'},'watermark':[],'borders':['none']}
   payload=serialize_moss_json(value)
   target=replace(stage,'image.png',payload)
   self.assertEqual(payload,target.read_bytes())
   for malformed in (b'{}',payload.replace(b'"none"',b'"left","none"')):
    with self.subTest(malformed=malformed),self.assertRaises(export_staging.StagingError):replace(stage,'image.png',malformed)

 def test_hardlinked_staging_preserves_source(self):
  with tempfile.TemporaryDirectory() as t:
   root=Path(t);source=root/'d';source.mkdir();(source/'nested').mkdir();file=source/'nested'/'a.png';file.write_bytes(b'x');stage=create_hardlink_staging(source,root/'.d.stage')
   copy=stage/'nested'/'a.png';self.assertEqual(b'x',copy.read_bytes());self.assertEqual(file.stat().st_ino,copy.stat().st_ino);self.assertTrue(file.exists())
   original=source/'note.json';original.write_bytes(b'old');copy=stage/'note.json';import os;os.link(original,copy);replace_business_annotation(stage,'note','.json',b'new');self.assertEqual(b'old',original.read_bytes());self.assertEqual(b'new',copy.read_bytes())
 def test_latent_is_hardlinked_but_business_annotations_are_independent(self):
  with tempfile.TemporaryDirectory() as t:
   root=Path(t);source=root/'dataset';source.mkdir();(source/'image.png').write_bytes(b'image');(source/'image.safetensors').write_bytes(b'latent');(source/'image.json').write_bytes(b'{"old":true}\n');(source/'image.txt').write_bytes(b'old\n')
   stage=create_hardlink_staging(source,root/'.dataset.stage');replace_business_annotation(stage,'image','.json',b'{"new":true}\n');replace_business_annotation(stage,'image','.txt',b'new\n')
   self.assertEqual((source/'image.png').stat().st_ino,(stage/'image.png').stat().st_ino);self.assertEqual((source/'image.safetensors').stat().st_ino,(stage/'image.safetensors').stat().st_ino)
   self.assertNotEqual((source/'image.json').stat().st_ino,(stage/'image.json').stat().st_ino);self.assertNotEqual((source/'image.txt').stat().st_ino,(stage/'image.txt').stat().st_ino);self.assertEqual(b'{"old":true}\n',(source/'image.json').read_bytes())
 def test_hardlink_failure_never_falls_back_to_copying_the_dataset(self):
  with tempfile.TemporaryDirectory() as t:
   root=Path(t);source=root/'dataset';source.mkdir();(source/'image.png').write_bytes(b'image');stage=root/'.dataset.stage'
   with patch('anima_core.export_staging.os.link',side_effect=OSError('hardlink unavailable')):
    with self.assertRaises(OSError):create_hardlink_staging(source,stage)
   self.assertTrue((source/'image.png').is_file());self.assertFalse(stage.exists())
 def test_staging_is_hidden_until_it_is_revealed_for_the_dataset_switch(self):
  import os
  if os.name!='nt':self.skipTest('the hidden attribute only exists on Windows')
  import ctypes;hidden=lambda p:bool(ctypes.windll.kernel32.GetFileAttributesW(str(p))&0x2)
  with tempfile.TemporaryDirectory() as t:
   root=Path(t);source=root/'dataset';source.mkdir();(source/'image.png').write_bytes(b'image');stage=create_hardlink_staging(source,root/'.dataset.stage')
   self.assertTrue(hidden(stage));reveal_staging(stage);self.assertFalse(hidden(stage))
   os.replace(stage,root/'promoted');self.assertFalse(hidden(root/'promoted'))
 def test_sibling_control_plane_files_never_enter_staging(self):
  with tempfile.TemporaryDirectory() as t:
   root=Path(t);source=root/'dataset';source.mkdir();(source/'image.png').write_bytes(b'image');(root/'state.db').write_bytes(b'db');(root/'state.db-wal').write_bytes(b'wal');(root/'logs').mkdir();(root/'logs'/'job.log').write_bytes(b'log')
   stage=create_hardlink_staging(source,root/'.dataset.stage')
   self.assertFalse((stage/'state.db').exists());self.assertFalse((stage/'state.db-wal').exists());self.assertFalse((stage/'logs').exists());self.assertTrue((stage/'image.png').is_file())

 def test_ocr_sidecars_use_distinct_image_extension_paths_and_replace_hardlinks_atomically(self):
  with tempfile.TemporaryDirectory() as t:
   root=Path(t);source=root/'dataset';source.mkdir();(source/'poster.jpg').write_bytes(b'jpg');(source/'poster.png').write_bytes(b'png')
   formal=source/'ocr_annotations';formal.mkdir();(formal/'poster.jpg.ocr.json').write_bytes(_ocr_sidecar('poster.jpg',status='no_text'));(formal/'poster.png.ocr.json').write_bytes(_ocr_sidecar('poster.png',status='no_text'))
   stage=create_hardlink_staging(source,root/'.dataset.stage')
   replace=getattr(export_staging,'replace_ocr_sidecar',None)
   self.assertTrue(callable(replace),'OCR staging replacement API is missing')
   replacement=_ocr_sidecar('poster.jpg',status='failed')
   replace(stage,'poster.jpg',replacement)
   self.assertEqual(_ocr_sidecar('poster.jpg',status='no_text'),(source/'ocr_annotations'/'poster.jpg.ocr.json').read_bytes())
   self.assertEqual(replacement,(stage/'ocr_annotations'/'poster.jpg.ocr.json').read_bytes())
   self.assertEqual(_ocr_sidecar('poster.png',status='no_text'),(stage/'ocr_annotations'/'poster.png.ocr.json').read_bytes())
   self.assertNotEqual((source/'ocr_annotations'/'poster.jpg.ocr.json').stat().st_ino,(stage/'ocr_annotations'/'poster.jpg.ocr.json').stat().st_ino)

 def test_ocr_sidecar_replacement_rejects_bad_data_and_unsafe_paths(self):
  with tempfile.TemporaryDirectory() as t:
   root=Path(t);source=root/'dataset';source.mkdir();(source/'image.png').write_bytes(b'image');stage=create_hardlink_staging(source,root/'.dataset.stage')
   replace=getattr(export_staging,'replace_ocr_sidecar',None)
   self.assertTrue(callable(replace),'OCR staging replacement API is missing')
   for relative,data in (('..\\image.png',_ocr_sidecar('image.png')),('image.txt',_ocr_sidecar('image.png')),('image.txt',_ocr_sidecar('image.txt')),('image.png',b'not-json'),('image.png',_ocr_sidecar('different.png'))):
    with self.subTest(relative=relative,data=data[:10]),self.assertRaises(export_staging.StagingError):replace(stage,relative,data)

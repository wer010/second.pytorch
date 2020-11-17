# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: second/protos/pipeline.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
from google.protobuf import descriptor_pb2
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from second.protos import input_reader_pb2 as second_dot_protos_dot_input__reader__pb2
from second.protos import model_pb2 as second_dot_protos_dot_model__pb2
from second.protos import train_pb2 as second_dot_protos_dot_train__pb2


DESCRIPTOR = _descriptor.FileDescriptor(
  name='second/protos/pipeline.proto',
  package='second.protos',
  syntax='proto3',
  serialized_pb=_b('\n\x1csecond/protos/pipeline.proto\x12\rsecond.protos\x1a second/protos/input_reader.proto\x1a\x19second/protos/model.proto\x1a\x19second/protos/train.proto\"\xe8\x01\n\x17TrainEvalPipelineConfig\x12,\n\x05model\x18\x01 \x01(\x0b\x32\x1d.second.protos.DetectionModel\x12\x36\n\x12train_input_reader\x18\x02 \x01(\x0b\x32\x1a.second.protos.InputReader\x12\x30\n\x0ctrain_config\x18\x03 \x01(\x0b\x32\x1a.second.protos.TrainConfig\x12\x35\n\x11\x65val_input_reader\x18\x04 \x01(\x0b\x32\x1a.second.protos.InputReaderb\x06proto3')
  ,
  dependencies=[second_dot_protos_dot_input__reader__pb2.DESCRIPTOR,second_dot_protos_dot_model__pb2.DESCRIPTOR,second_dot_protos_dot_train__pb2.DESCRIPTOR,])




_TRAINEVALPIPELINECONFIG = _descriptor.Descriptor(
  name='TrainEvalPipelineConfig',
  full_name='second.protos.TrainEvalPipelineConfig',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='model', full_name='second.protos.TrainEvalPipelineConfig.model', index=0,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='train_input_reader', full_name='second.protos.TrainEvalPipelineConfig.train_input_reader', index=1,
      number=2, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='train_config', full_name='second.protos.TrainEvalPipelineConfig.train_config', index=2,
      number=3, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='eval_input_reader', full_name='second.protos.TrainEvalPipelineConfig.eval_input_reader', index=3,
      number=4, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=136,
  serialized_end=368,
)

_TRAINEVALPIPELINECONFIG.fields_by_name['model'].message_type = second_dot_protos_dot_model__pb2._DETECTIONMODEL
_TRAINEVALPIPELINECONFIG.fields_by_name['train_input_reader'].message_type = second_dot_protos_dot_input__reader__pb2._INPUTREADER
_TRAINEVALPIPELINECONFIG.fields_by_name['train_config'].message_type = second_dot_protos_dot_train__pb2._TRAINCONFIG
_TRAINEVALPIPELINECONFIG.fields_by_name['eval_input_reader'].message_type = second_dot_protos_dot_input__reader__pb2._INPUTREADER
DESCRIPTOR.message_types_by_name['TrainEvalPipelineConfig'] = _TRAINEVALPIPELINECONFIG
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

TrainEvalPipelineConfig = _reflection.GeneratedProtocolMessageType('TrainEvalPipelineConfig', (_message.Message,), dict(
  DESCRIPTOR = _TRAINEVALPIPELINECONFIG,
  __module__ = 'second.protos.pipeline_pb2'
  # @@protoc_insertion_point(class_scope:second.protos.TrainEvalPipelineConfig)
  ))
_sym_db.RegisterMessage(TrainEvalPipelineConfig)


# @@protoc_insertion_point(module_scope)

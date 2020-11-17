# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: second/protos/preprocess.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
from google.protobuf import descriptor_pb2
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='second/protos/preprocess.proto',
  package='second.protos',
  syntax='proto3',
  serialized_pb=_b('\n\x1esecond/protos/preprocess.proto\x12\rsecond.protos\"\xd6\x01\n\x19\x44\x61tabasePreprocessingStep\x12\x43\n\x14\x66ilter_by_difficulty\x18\x01 \x01(\x0b\x32#.second.protos.DBFilterByDifficultyH\x00\x12U\n\x18\x66ilter_by_min_num_points\x18\x02 \x01(\x0b\x32\x31.second.protos.DBFilterByMinNumPointInGroundTruthH\x00\x42\x1d\n\x1b\x64\x61tabase_preprocessing_step\"4\n\x14\x44\x42\x46ilterByDifficulty\x12\x1c\n\x14removed_difficulties\x18\x01 \x03(\x05\"\xc3\x01\n\"DBFilterByMinNumPointInGroundTruth\x12\x64\n\x13min_num_point_pairs\x18\x01 \x03(\x0b\x32G.second.protos.DBFilterByMinNumPointInGroundTruth.MinNumPointPairsEntry\x1a\x37\n\x15MinNumPointPairsEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\r\n\x05value\x18\x02 \x01(\r:\x02\x38\x01\x62\x06proto3')
)




_DATABASEPREPROCESSINGSTEP = _descriptor.Descriptor(
  name='DatabasePreprocessingStep',
  full_name='second.protos.DatabasePreprocessingStep',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='filter_by_difficulty', full_name='second.protos.DatabasePreprocessingStep.filter_by_difficulty', index=0,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='filter_by_min_num_points', full_name='second.protos.DatabasePreprocessingStep.filter_by_min_num_points', index=1,
      number=2, type=11, cpp_type=10, label=1,
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
    _descriptor.OneofDescriptor(
      name='database_preprocessing_step', full_name='second.protos.DatabasePreprocessingStep.database_preprocessing_step',
      index=0, containing_type=None, fields=[]),
  ],
  serialized_start=50,
  serialized_end=264,
)


_DBFILTERBYDIFFICULTY = _descriptor.Descriptor(
  name='DBFilterByDifficulty',
  full_name='second.protos.DBFilterByDifficulty',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='removed_difficulties', full_name='second.protos.DBFilterByDifficulty.removed_difficulties', index=0,
      number=1, type=5, cpp_type=1, label=3,
      has_default_value=False, default_value=[],
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
  serialized_start=266,
  serialized_end=318,
)


_DBFILTERBYMINNUMPOINTINGROUNDTRUTH_MINNUMPOINTPAIRSENTRY = _descriptor.Descriptor(
  name='MinNumPointPairsEntry',
  full_name='second.protos.DBFilterByMinNumPointInGroundTruth.MinNumPointPairsEntry',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='key', full_name='second.protos.DBFilterByMinNumPointInGroundTruth.MinNumPointPairsEntry.key', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='value', full_name='second.protos.DBFilterByMinNumPointInGroundTruth.MinNumPointPairsEntry.value', index=1,
      number=2, type=13, cpp_type=3, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=_descriptor._ParseOptions(descriptor_pb2.MessageOptions(), _b('8\001')),
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=461,
  serialized_end=516,
)

_DBFILTERBYMINNUMPOINTINGROUNDTRUTH = _descriptor.Descriptor(
  name='DBFilterByMinNumPointInGroundTruth',
  full_name='second.protos.DBFilterByMinNumPointInGroundTruth',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='min_num_point_pairs', full_name='second.protos.DBFilterByMinNumPointInGroundTruth.min_num_point_pairs', index=0,
      number=1, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[_DBFILTERBYMINNUMPOINTINGROUNDTRUTH_MINNUMPOINTPAIRSENTRY, ],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=321,
  serialized_end=516,
)

_DATABASEPREPROCESSINGSTEP.fields_by_name['filter_by_difficulty'].message_type = _DBFILTERBYDIFFICULTY
_DATABASEPREPROCESSINGSTEP.fields_by_name['filter_by_min_num_points'].message_type = _DBFILTERBYMINNUMPOINTINGROUNDTRUTH
_DATABASEPREPROCESSINGSTEP.oneofs_by_name['database_preprocessing_step'].fields.append(
  _DATABASEPREPROCESSINGSTEP.fields_by_name['filter_by_difficulty'])
_DATABASEPREPROCESSINGSTEP.fields_by_name['filter_by_difficulty'].containing_oneof = _DATABASEPREPROCESSINGSTEP.oneofs_by_name['database_preprocessing_step']
_DATABASEPREPROCESSINGSTEP.oneofs_by_name['database_preprocessing_step'].fields.append(
  _DATABASEPREPROCESSINGSTEP.fields_by_name['filter_by_min_num_points'])
_DATABASEPREPROCESSINGSTEP.fields_by_name['filter_by_min_num_points'].containing_oneof = _DATABASEPREPROCESSINGSTEP.oneofs_by_name['database_preprocessing_step']
_DBFILTERBYMINNUMPOINTINGROUNDTRUTH_MINNUMPOINTPAIRSENTRY.containing_type = _DBFILTERBYMINNUMPOINTINGROUNDTRUTH
_DBFILTERBYMINNUMPOINTINGROUNDTRUTH.fields_by_name['min_num_point_pairs'].message_type = _DBFILTERBYMINNUMPOINTINGROUNDTRUTH_MINNUMPOINTPAIRSENTRY
DESCRIPTOR.message_types_by_name['DatabasePreprocessingStep'] = _DATABASEPREPROCESSINGSTEP
DESCRIPTOR.message_types_by_name['DBFilterByDifficulty'] = _DBFILTERBYDIFFICULTY
DESCRIPTOR.message_types_by_name['DBFilterByMinNumPointInGroundTruth'] = _DBFILTERBYMINNUMPOINTINGROUNDTRUTH
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

DatabasePreprocessingStep = _reflection.GeneratedProtocolMessageType('DatabasePreprocessingStep', (_message.Message,), dict(
  DESCRIPTOR = _DATABASEPREPROCESSINGSTEP,
  __module__ = 'second.protos.preprocess_pb2'
  # @@protoc_insertion_point(class_scope:second.protos.DatabasePreprocessingStep)
  ))
_sym_db.RegisterMessage(DatabasePreprocessingStep)

DBFilterByDifficulty = _reflection.GeneratedProtocolMessageType('DBFilterByDifficulty', (_message.Message,), dict(
  DESCRIPTOR = _DBFILTERBYDIFFICULTY,
  __module__ = 'second.protos.preprocess_pb2'
  # @@protoc_insertion_point(class_scope:second.protos.DBFilterByDifficulty)
  ))
_sym_db.RegisterMessage(DBFilterByDifficulty)

DBFilterByMinNumPointInGroundTruth = _reflection.GeneratedProtocolMessageType('DBFilterByMinNumPointInGroundTruth', (_message.Message,), dict(

  MinNumPointPairsEntry = _reflection.GeneratedProtocolMessageType('MinNumPointPairsEntry', (_message.Message,), dict(
    DESCRIPTOR = _DBFILTERBYMINNUMPOINTINGROUNDTRUTH_MINNUMPOINTPAIRSENTRY,
    __module__ = 'second.protos.preprocess_pb2'
    # @@protoc_insertion_point(class_scope:second.protos.DBFilterByMinNumPointInGroundTruth.MinNumPointPairsEntry)
    ))
  ,
  DESCRIPTOR = _DBFILTERBYMINNUMPOINTINGROUNDTRUTH,
  __module__ = 'second.protos.preprocess_pb2'
  # @@protoc_insertion_point(class_scope:second.protos.DBFilterByMinNumPointInGroundTruth)
  ))
_sym_db.RegisterMessage(DBFilterByMinNumPointInGroundTruth)
_sym_db.RegisterMessage(DBFilterByMinNumPointInGroundTruth.MinNumPointPairsEntry)


_DBFILTERBYMINNUMPOINTINGROUNDTRUTH_MINNUMPOINTPAIRSENTRY.has_options = True
_DBFILTERBYMINNUMPOINTINGROUNDTRUTH_MINNUMPOINTPAIRSENTRY._options = _descriptor._ParseOptions(descriptor_pb2.MessageOptions(), _b('8\001'))
# @@protoc_insertion_point(module_scope)

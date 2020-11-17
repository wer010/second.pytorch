# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: second/protos/sampler.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
from google.protobuf import descriptor_pb2
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from second.protos import preprocess_pb2 as second_dot_protos_dot_preprocess__pb2


DESCRIPTOR = _descriptor.FileDescriptor(
  name='second/protos/sampler.proto',
  package='second.protos',
  syntax='proto3',
  serialized_pb=_b('\n\x1bsecond/protos/sampler.proto\x12\rsecond.protos\x1a\x1esecond/protos/preprocess.proto\"}\n\x05Group\x12?\n\x0fname_to_max_num\x18\x01 \x03(\x0b\x32&.second.protos.Group.NameToMaxNumEntry\x1a\x33\n\x11NameToMaxNumEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\r\n\x05value\x18\x02 \x01(\r:\x02\x38\x01\"\xd8\x01\n\x07Sampler\x12\x1a\n\x12\x64\x61tabase_info_path\x18\x01 \x01(\t\x12+\n\rsample_groups\x18\x02 \x03(\x0b\x32\x14.second.protos.Group\x12\x45\n\x13\x64\x61tabase_prep_steps\x18\x03 \x03(\x0b\x32(.second.protos.DatabasePreprocessingStep\x12/\n\'global_random_rotation_range_per_object\x18\x04 \x03(\x02\x12\x0c\n\x04rate\x18\x05 \x01(\x02\x62\x06proto3')
  ,
  dependencies=[second_dot_protos_dot_preprocess__pb2.DESCRIPTOR,])




_GROUP_NAMETOMAXNUMENTRY = _descriptor.Descriptor(
  name='NameToMaxNumEntry',
  full_name='second.protos.Group.NameToMaxNumEntry',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='key', full_name='second.protos.Group.NameToMaxNumEntry.key', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='value', full_name='second.protos.Group.NameToMaxNumEntry.value', index=1,
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
  serialized_start=152,
  serialized_end=203,
)

_GROUP = _descriptor.Descriptor(
  name='Group',
  full_name='second.protos.Group',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='name_to_max_num', full_name='second.protos.Group.name_to_max_num', index=0,
      number=1, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[_GROUP_NAMETOMAXNUMENTRY, ],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=78,
  serialized_end=203,
)


_SAMPLER = _descriptor.Descriptor(
  name='Sampler',
  full_name='second.protos.Sampler',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='database_info_path', full_name='second.protos.Sampler.database_info_path', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='sample_groups', full_name='second.protos.Sampler.sample_groups', index=1,
      number=2, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='database_prep_steps', full_name='second.protos.Sampler.database_prep_steps', index=2,
      number=3, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='global_random_rotation_range_per_object', full_name='second.protos.Sampler.global_random_rotation_range_per_object', index=3,
      number=4, type=2, cpp_type=6, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='rate', full_name='second.protos.Sampler.rate', index=4,
      number=5, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
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
  serialized_start=206,
  serialized_end=422,
)

_GROUP_NAMETOMAXNUMENTRY.containing_type = _GROUP
_GROUP.fields_by_name['name_to_max_num'].message_type = _GROUP_NAMETOMAXNUMENTRY
_SAMPLER.fields_by_name['sample_groups'].message_type = _GROUP
_SAMPLER.fields_by_name['database_prep_steps'].message_type = second_dot_protos_dot_preprocess__pb2._DATABASEPREPROCESSINGSTEP
DESCRIPTOR.message_types_by_name['Group'] = _GROUP
DESCRIPTOR.message_types_by_name['Sampler'] = _SAMPLER
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

Group = _reflection.GeneratedProtocolMessageType('Group', (_message.Message,), dict(

  NameToMaxNumEntry = _reflection.GeneratedProtocolMessageType('NameToMaxNumEntry', (_message.Message,), dict(
    DESCRIPTOR = _GROUP_NAMETOMAXNUMENTRY,
    __module__ = 'second.protos.sampler_pb2'
    # @@protoc_insertion_point(class_scope:second.protos.Group.NameToMaxNumEntry)
    ))
  ,
  DESCRIPTOR = _GROUP,
  __module__ = 'second.protos.sampler_pb2'
  # @@protoc_insertion_point(class_scope:second.protos.Group)
  ))
_sym_db.RegisterMessage(Group)
_sym_db.RegisterMessage(Group.NameToMaxNumEntry)

Sampler = _reflection.GeneratedProtocolMessageType('Sampler', (_message.Message,), dict(
  DESCRIPTOR = _SAMPLER,
  __module__ = 'second.protos.sampler_pb2'
  # @@protoc_insertion_point(class_scope:second.protos.Sampler)
  ))
_sym_db.RegisterMessage(Sampler)


_GROUP_NAMETOMAXNUMENTRY.has_options = True
_GROUP_NAMETOMAXNUMENTRY._options = _descriptor._ParseOptions(descriptor_pb2.MessageOptions(), _b('8\001'))
# @@protoc_insertion_point(module_scope)

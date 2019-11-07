# Copyright (c) Fraunhofer MEVIS, Germany. All rights reserved.
# **InsertLicense** code

__author__ = 'gchlebus'


def get_data_provider(cf):
  interface = cf.data_server
  port = cf.data_port
  print("Connecting to remote data server on %s:%d..." % (interface, port))
  from dnn.remote.client import connect_socket
  client = connect_socket(interface, port)

  print("Initialising data streams...")
  if not client.initialize_streams():
    raise RuntimeError("Initializing streams failed")

  training_case_count = client.training_case_count()
  validation_case_count = client.validation_case_count()
  print('Connection established. remote side offers %d training and %d validation cases.' % (
    training_case_count, validation_case_count))

  if not training_case_count:
    raise RuntimeError('empty data source (no training cases)')
  if not validation_case_count:
    raise RuntimeError('empty data source (no validation cases)')

  return client

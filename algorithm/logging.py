""" Adapted From torch/utils/tensorboard/summary.py """

import torch
from torch.utils.tensorboard.writer import SummaryWriter
from google.protobuf.struct_pb2 import Value, ListValue
from tensorboard.compat.proto.summary_pb2 import (Summary, SummaryMetadata)
from tensorboard.plugins.hparams.api_pb2 import (Experiment, HParamInfo, MetricInfo, MetricName, Status, DataType)
from tensorboard.plugins.hparams.metadata import (PLUGIN_NAME, PLUGIN_DATA_VERSION, EXPERIMENT_TAG, SESSION_START_INFO_TAG, SESSION_END_INFO_TAG)
from tensorboard.plugins.hparams.plugin_data_pb2 import (HParamsPluginData, SessionEndInfo, SessionStartInfo)

# Helper function to make SumaryMetadata
smd = lambda content: SummaryMetadata(plugin_data=SummaryMetadata.PluginData(plugin_name=PLUGIN_NAME, content=content.SerializeToString()))

def write_session_start(writer: SummaryWriter, hparam_dict: dict, hparam_domain_discrete={}):
  """Writes `SessionStartInfo` (key-value pairs of the hyperparameters). Args:
    writer: SummaryWriter to log to
    hparam_dict: A dictionary that contains names of the hyperparameters and their values.
    hparam_domain_discrete: (Optional[Dict[str, List[Any]]]) A dictionary that contains names of 
        the hyperparameters and all discrete values they can hold
  Returns: Hyperparameter Infos (needed for write_experiement)"""
  ssi, hp_info = SessionStartInfo(), []
  assert isinstance(hparam_dict, dict), 'parameter: hparam_dict should be a dictionary, nothing logged.'
  assert isinstance(hparam_domain_discrete, dict), "parameter: hparam_domain_discrete should be a dictionary, nothing logged."
  for k, v in hparam_domain_discrete.items(): 
    assert(k in hparam_dict or isinstance(v, list) or all(isinstance(d, type(hparam_dict[k])) for d in v)), f"parameter: hparam_domain_discrete[{k}] should be a list of same type as hparam_dict[{v}]."

  # Helper to generate HP-Info entries & Helper to generate domain_discrete-info
  hpi = lambda name, type, domain_discrete: HParamInfo(name=name, type=DataType.Value(type), domain_discrete=domain_discrete)
  ddi = lambda key, val: ListValue(values=[val(d) for d in hparam_domain_discrete[key]]) if key in hparam_domain_discrete else None
  for key, val in hparam_dict.items():
    if val is None: continue
    if isinstance(val, int) or isinstance(val, float):
      hp_info.append(hpi(key, "DATA_TYPE_FLOAT64", ddi(key, lambda v: Value(number_value=v))))
      ssi.hparams[key].number_value = val; continue
    if isinstance(val, str):
      hp_info.append(hpi(key, "DATA_TYPE_STRING", ddi(key, lambda v: Value(string_value=v))))
      ssi.hparams[key].string_value = val; continue
    if isinstance(val, bool):
      hp_info.append(hpi(key, "DATA_TYPE_BOOL", ddi(key, lambda v: Value(bool_value=v))))
      ssi.hparams[key].bool_value = val; continue
    if isinstance(val, torch.Tensor):
      hp_info.append(HParamInfo(name=key, type=DataType.Value("DATA_TYPE_FLOAT64")))
      ssi.hparams[key].number_value = val.cpu().numpy()[0]; continue
    raise ValueError('value should be one of int, float, str, bool, or torch.Tensor')
  content = HParamsPluginData(session_start_info=ssi, version=PLUGIN_DATA_VERSION)
  ssi = Summary(value=[Summary.Value(tag=SESSION_START_INFO_TAG, metadata=smd(content))])
  writer.file_writer.add_summary(ssi)
  return hp_info

def write_experiment(writer: SummaryWriter, metric_list: list, hp_info: list):
  """Writes `Experiment` (keeps the metadata of an experiment, such as the name of the hyperparameters and the name of the metrics.). Args:
    writer: SummaryWriter to log to
    metric_list: A list that contains names of the metrics."""
  assert isinstance(metric_list, list), 'parameter: metric_list should be a list.'
  assert isinstance(hp_info, list), 'parameter: hp_info should be a list.'
  mts = [MetricInfo(name=MetricName(tag=tag)) for tag in metric_list]
  exp = Experiment(hparam_infos=hp_info, metric_infos=mts)
  content = HParamsPluginData(experiment=exp, version=PLUGIN_DATA_VERSION)
  exp = Summary(value=[Summary.Value(tag=EXPERIMENT_TAG, metadata=smd(content))])
  writer.file_writer.add_summary(exp)

def write_session_end(writer: SummaryWriter, status='STATUS_SUCCESS'):
  """Writes `SessionEndInfo` (describes status of the experiment e.g. STATUS_SUCCESS). Args:
    writer: SummaryWriter to log to
    status: status of the eperiment ('STATUS_SUCCESS' 'STATUS_FAILURE' 'STATUS_UNKNOWN' 'STATUS_RUNNING') """
  sei = SessionEndInfo(status=Status.Value(status)) 
  content = HParamsPluginData(session_end_info=sei, version=PLUGIN_DATA_VERSION)
  sei = Summary(value=[Summary.Value(tag=SESSION_END_INFO_TAG, metadata=smd(content))])
  writer.file_writer.add_summary(sei)
  writer.flush()

def write_hyperparameters(model, metrics: list, status='STATUS_SUCCESS'):
  hpinfos = write_session_start(model.writer, model.get_hparams())
  write_experiment(model.writer, metrics, hpinfos)
  write_session_end(model.writer, status)

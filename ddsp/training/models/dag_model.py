# Copyright 2020 The DDSP Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""Model wrapper of a DAGLayer."""

import ddsp
from ddsp import dag_ops
from ddsp.training.models.model import Model


class DAGModel(Model):
  """Model with methods and modules all specified in gin configs."""

  def __init__(self, dag, output_losses=None, **kwargs):
    keras_kwargs, kwargs = dag_ops.split_keras_kwargs(kwargs)
    super().__init__(**keras_kwargs)

    # Create properties/submodules from kwargs and dags.
    modules = dag_ops.filter_by_value(kwargs, dag_ops.is_module)
    self.__dict__.update(modules)
    self.modules = list(modules.values())

    loss_objs = dag_ops.filter_by_value(kwargs, dag_ops.is_loss)
    self.__dict__.update(loss_objs)
    self.loss_objs = list(loss_objs.values())

    dag, modules = dag_ops.extract_modules_from_dag(dag, dag_ops.is_module)
    self.__dict__.update(modules)
    self.modules += list(modules.values())
    self.__dict__['dag'] = dag  # Needed to avoid ListWrapper.

    output_losses, loss_objs = dag_ops.extract_modules_from_dag(output_losses,
                                                                dag_ops.is_loss)
    self.__dict__.update(loss_objs)
    self.loss_objs += list(loss_objs.values())
    self.__dict__['output_losses'] = output_losses  # Avoid ListWrapper.

  def _update_losses_from_outputs(self, outputs, **kwargs):
    """Update losses from tensors in outputs dict."""
    if self.output_losses is not None:
      for loss_key, input_keys in self.output_losses:
        loss_obj = getattr(self, loss_key)
        loss_inputs = [ddsp.core.nested_lookup(k, outputs) for k in input_keys]
        self._update_losses_dict(loss_obj, *loss_inputs)

  def get_audio_from_outputs(self, outputs):
    # 'out' is a reserved key for final dag output.
    out = outputs['out']
    if isinstance(out, dict):
      audio = out.get('signal')
      if audio is None:
        audio = out.get('0')
      if audio is None:
        raise ValueError(f'Audio output `signal` or `0` not '
                         f'found in dag_layer outputs: {out}')
      return audio
    else:
      return out

  def call(self, batch, **kwargs):
    """Forward pass."""
    outputs = dag_ops.run_dag(self, self.dag, batch, **kwargs)
    outputs['audio_gen'] = self.get_audio_from_outputs(outputs)
    self._update_losses_from_outputs(outputs)
    return outputs


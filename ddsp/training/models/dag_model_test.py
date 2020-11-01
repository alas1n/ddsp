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
"""Tests for ddsp.training.models.dag_model."""

from absl.testing import parameterized
from ddsp.training import models
import gin
import tensorflow as tf


class DAGModelTest(parameterized.TestCase, tf.test.TestCase):

  def setUp(self):
    """Create some dummy input data for the chain."""
    super().setUp()
    # Create inputs.
    self.n_batch = 4
    self.x_dims = 5
    self.z_dims = 2
    self.x = tf.ones([self.n_batch, self.x_dims])
    self.inputs = {'test_data': self.x}
    self.gin_config = f"""
    import ddsp
    import ddsp.training

    dag_ops.run_dag.verbose = True

    ### Modules
    DAGModel.dag = [
        (@encoder/nn.Fc(), ['test_data'], ),
        (@middle/nn.Fc(), ['encoder/0'], ['output'] ),
        (@decoder/nn.Fc(), ['middle/output'], ),
    ]
    encoder/Fc.name = 'encoder'
    encoder/Fc.ch = {self.z_dims}

    middle/Fc.name = 'middle'
    middle/Fc.ch = {self.z_dims}

    decoder/Fc.name = 'decoder'
    decoder/Fc.ch = {self.x_dims}
    """

  def test_build_model(self):
    """Tests if Model builds properly and produces audio of correct shape."""
    with gin.unlock_config():
      gin.clear_config()
      gin.parse_config(self.gin_config)

    model = models.DAGModel()
    outputs = model(self.inputs)
    self.assertIsInstance(outputs, dict)

    z = outputs['middle']['output']
    x_rec = outputs['decoder']['0']

    # Confirm that model generates correctly sized tensors.
    self.assertEqual(outputs['test_data'].shape, self.x.shape)
    self.assertEqual(x_rec.shape, self.x.shape)
    self.assertEqual(z.shape[-1], self.z_dims)

if __name__ == '__main__':
  tf.test.main()

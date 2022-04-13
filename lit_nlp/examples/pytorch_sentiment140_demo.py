
import sys

from absl import app
from absl import flags
from absl import logging

from lit_nlp import dev_server
from lit_nlp import server_flags
from lit_nlp.api import model as lit_model
from lit_nlp.api import types as lit_types
# Use the regular GLUE data loaders, because these are very simple already.
from lit_nlp.examples.datasets import sentiment140
from lit_nlp.lib import utils
# import lit_nlp.dev_server
import torch
import transformers
from lit_nlp.examples.simple_pytorch_demo import *



def main(_):
  # Normally path is a directory; if it's an archive file, download and
  # extract to the transformers cache.
  model_path = FLAGS.model_path
  if model_path.endswith(".tar.gz"):
    model_path = transformers.file_utils.cached_path(
        model_path, extract_compressed_file=True)

  # Load the model we defined above.
  models = {"sst": SimpleSentimentModel(model_path)}
  # Load SST-2 validation set from TFDS.
  datasets = {"sentiment140_test": sentiment140.Sentiment140Data("test")}

  # Start the LIT server. See server_flags.py for server options.
  lit_demo = dev_server.Server(models, datasets, **server_flags.get_flags())
  return lit_demo.serve()


if __name__ == "__main__":
  app.run(main)

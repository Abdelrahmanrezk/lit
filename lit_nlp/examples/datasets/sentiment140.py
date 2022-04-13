
# Lint as: python3
"""GLUE benchmark datasets, using TFDS.

See https://gluebenchmark.com/ and
https://www.tensorflow.org/datasets/catalog/glue

Note that this requires the TensorFlow Datasets package, but the resulting LIT
datasets just contain regular Python/NumPy data.
"""
from lit_nlp.api import dataset as lit_dataset
from lit_nlp.api import types as lit_types

import tensorflow_datasets as tfds


def load_tfds(*args, do_sort=True, **kw):
  """Load from TFDS, with optional sorting."""
  # Materialize to NumPy arrays.
  # This also ensures compatibility with TF1.x non-eager mode, which doesn't
  # support direct iteration over a tf.data.Dataset.
  # print("="*50)
  ret = list(tfds.as_numpy(tfds.load(*args, download=True, try_gcs=True, **kw)))
  # print("="*50)
  if do_sort:
    # Recover original order, as if you loaded from a TSV file.
    ret.sort(key=lambda ex: ex['polarity'])

  return ret




class Sentiment140Data(lit_dataset.Dataset):
  """Stanford Sentiment Treebank, binary version (SST-2).

  See https://www.tensorflow.org/datasets/catalog/glue#gluesst2.
  """

  LABELS = ['0', '1']

  def __init__(self, split: str):
    self._examples = []
    for ex in load_tfds('sentiment140', split=split):
      if ex['polarity'] == 2:
        continue
      if ex['polarity'] == 4:
        ex['polarity'] = 1

      self._examples.append({
          'sentence': ex['text'].decode('utf-8'),
          'label': self.LABELS[ex['polarity']],
      })

    # print("="*50)

  def spec(self):
    return {
        'sentence': lit_types.TextSegment(),
        'label': lit_types.CategoryLabel(vocab=self.LABELS)
    }
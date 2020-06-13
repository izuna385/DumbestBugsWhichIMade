# DumbestBugsWhichIMade

* [Torch](#torch)
* [Allennlp](#allennlp)

## Torch

* Careless squeezing

  ```
  tensor = tensor.squeeze()
  ```

  * This may cause error when `batch_size == 1`.

  * For avoiding this, simply clarify dimensions which you want to trim.


* Unnecessary GPU occupation with `nn.embedding`.

  ```
    class MyModel(Model):
      def __init__(args):
        self.args = args
        ...
        self.embedding = nn.Embedding(idx_size, dim)
        self.embedding.weight.data.copy_(torch.from_numpy(pretrained_weight))
  ```  

  * Even if you declare `nn.embedding.requires_grad_ = False`, it takes gpu when model is called with `.cuda()` methods, like `model.cuda()`.

  *  This may cause uncesessary gpu-memory occupation. When you use `nn.Embedding` only for converter from index to embedding, you don't have to put this on gpu.

  * Actually, when `idx_size` is ~ million, `nn.embedding` takes much gpu-memory.

  * For avoiding this, you can use `numpy.take()` methods for converting index to embedding tensor.

  ```
  embs = torch.from_numpy(numpyEmbMatrix.take(
    idxsForEmbeddings.squeeze(1), axis=0)).cuda()
  ```

## AllenNLP

* Do not wrap split word with double Token class.

  `wrapped = Token(Token("###ly"))`

  * This cause deletion of sub-word information and still do **not** return error.

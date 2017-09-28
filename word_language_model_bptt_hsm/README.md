# Word-level language modeling RNN with memory efficient BPTT

The original example [README is here](https://github.com/pytorch/examples/blob/master/word_language_model/README.md) and still applies to this example.  This version adds simple memory efficient BPTT similar to that described as selective hidden state memorization (BPTT-HSM) in [[Gruslys 2016 Memory-Efficient Backpropagation Through Time](https://arxiv.org/pdf/1606.03401.pdf) via a new flag '--bptt_step LENGTH'.

The approach snips a desired backpropagation sequence length (given with --bptt) into smaller pieces (given with --bptt_step) and runs backpropagation on each piece, attaching input gradients to outputs (since we're going in reverse).  The memory savings occurs because the maximum  chain of Variable history needed is the smaller step value instead of the full BPTT length.  

For example, a 512 bptt sequence can be run in two 256 sequence lengths for less memory with an extra 256 sequence forward step computational cost:

```bash
python main.py --cuda --epochs 6 --bptt 512 --bptt_step 512
```

On a Titan X (pascal), this takes 11.8 seconds per epoch and uses 2585 MiB.

```bash
python main.py --cuda --epochs 6 --bptt 512 --bptt_step 256
```

On the same gpu, this takes 13.1 seconds per epoch and uses 1543 MiB (as reported by nvidia-smi).  Base memory use before training is 463 MiB so memory use is pretty close to 0.50 of original.  Measurements use pytorch 0.2.0.post3.

This code assumes the hidden state returned by the model is a sequence of Variables; if instead it is a sequence of sequences (if layers are kept in seperate sequences for example), code will need modified.

This implementation also turns off size_average of the CrossEntropyLoss criterion and does gradient averaging after all backprop is done in order to match default behavior.

There may be small differences in loss when using bptt_step < bptt due to the non-associativity of floating point arithmetic (a+(b+c) != (a+b)+c).

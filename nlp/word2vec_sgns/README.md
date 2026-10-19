# Distributed Representations of Words and Phrases and their Compositionality

NIPS 2013 | [Notion summary](https://road-lamp-b0c.notion.site/3fa64a44030c810894b0f06a25d430ab) | [Paper](https://arxiv.org/abs/1310.4546) | [Official code](https://code.google.com/archive/p/word2vec/)

## What is in the code
- Skip-gram with negative sampling; negatives drawn from the unigram distribution to the 3/4 power
- Subsampling of frequent words with keep probability sqrt(t / f(w))
- Analogy test by vector arithmetic (capital - country + country')

## Differences from the paper
Trained on a synthetic corpus of country, capital and language sentences instead of billions of words; phrase detection is not implemented.

## Run
```bash
python main.py
```

## Output
```text
epoch 0  pairs=164322  loss=2.424
epoch 1  pairs=164174  loss=1.974
epoch 2  pairs=163662  loss=1.946
epoch 3  pairs=163820  loss=1.950
epoch 4  pairs=164614  loss=1.943
epoch 5  pairs=163556  loss=1.942
epoch 6  pairs=163872  loss=1.940
epoch 7  pairs=164318  loss=1.942
epoch 8  pairs=163452  loss=1.940
epoch 9  pairs=164680  loss=1.941
epoch 10  pairs=163690  loss=1.945
epoch 11  pairs=163712  loss=1.942
epoch 12  pairs=164472  loss=1.941
epoch 13  pairs=163338  loss=1.941
epoch 14  pairs=163784  loss=1.942

Nearest to 'seoul': ['rome', 'madrid', 'tokyo']
Analogy accuracy (capital - country + country'): 26/30
Example: paris - france + korea -> seoul
```

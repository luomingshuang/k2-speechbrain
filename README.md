# k2-speechbrain
In this repository, I try to combine k2 with speechbrain to decoding well and fastly.

- At the basis of the codes[https://gist.github.com/csukuangfj/c68697cd144c8f063cc7ec4fd885fd6f] from @csukuangfj , I try to combine k2 with the pretrained transformer encoder and get some results on LibriSpeech.
- I use the public pretrained transformer encoder from the speechbrain team.
- I test the two datasets' samples (test-clean and test-other) one by one. And in my experiments, I found that the decoding process based on k2 was much faster than speechbrain (transformer-LM).
- Some results I get are as follows:
``` 
                             Method                             |  test-clean(WER%) | test-other(WER%)
----------------------------------------------------------------------------------------------------
                      speechbrain (public)                      |       2.46        |      5.77
----------------------------------------------------------------------------------------------------
             k2+pre-encoder (use-whole-lattices=False)          |       8.49        |      17.42
----------------------------------------------------------------------------------------------------
       k2+pre-encoder (use-whole-lattices=True, lm-scale=1.2)   |       6.39        |      16.67
----------------------------------------------------------------------------------------------------                         
````

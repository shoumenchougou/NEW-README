# RWKV-V2

> [!NOTE]
> 
> **The following content is an archive of RWKV-V2 content, RWKV-V2 [first committed](https://github.com/BlinkDL/RWKV-LM/commit/1f189a4034c7ff1d0c145ef556327de1fa90da51) on Mar 22, 2022.**
>
> Last version of V2 : [README](https://github.com/BlinkDL/RWKV-LM/blob/92da17d68f9ecb6ab1b8c5119c222c25c6991c06/README.md)

## The pseudocode (execution from top to bottom):

![RWKV-v2-RNN](./img/RWKV-v2-RNN.png)

The a b c d factors work together to build a time-decay curve: [X, 1, W, W^2, W^3, ...].

Write out the formulas for "token at pos 2" and "token at pos 3" and you will get the idea:
* a and b: EMAs of kv and k.
* c and d: these are a and b combined with "self-attention".

kv / k is the memory mechanism. The token with high k can be remembered for a long duration, if W is close to 1 in the channel.

The R-gate is important for performance. k = info strength of this token (to be passed to future tokens). r = whether to apply the info to this token.


## RWKV-3 improvements

Use different trainable TimeMix factors for R / K / V in SA and FF layers. Example:
```python
xx = self.time_shift(x)
xk = x * self.time_mix_k + xx * (1 - self.time_mix_k)
xv = x * self.time_mix_v + xx * (1 - self.time_mix_v)
xr = x * self.time_mix_r + xx * (1 - self.time_mix_r)
```

Use preLN instead of postLN (more stable & faster convergence):
```python
if self.layer_id == 0:
	x = self.ln0(x)
x = x + self.att(self.ln1(x))
x = x + self.ffn(self.ln2(x))
```

## Multimodal ideas

I have an idea for [text --> 32x32 RGB image] using a LM (transformer, RWKV, etc.). Will test it soon.

Firstly, LM loss (instead of L2 loss), so the image will not be blurry.

Secondly, color quantization. For example, only allowing 8 levels for R/G/B. Then the image vocab size is 8x8x8 = 512 (for each pixel), instead of 2^24.
Therefore, a 32x32 RGB image = a len1024 sequence of vocab512 (image tokens), which is a typical input for usual LMs.
(Later we can use diffusion models to upsample and generate RGB888 images. We might be able to use a LM for this too.)

Thirdly, 2D positional embeddings that are easy for the model to understand.
For example, add one-hot X & Y coords to the first 64(=32+32) channels. Say if the pixel is at x=8, y=20, then we will add 1 to channel 8 and channel 52 (=32+20).
Moreover probably we can add the float X & Y coords (normalized to 0~1 range) to another 2 channels. And other periodic pos. encoding might help too (will test). 

Finally, RandRound when doing the color quantization in the DataLoader.
For example, if the float level is 4.578, then there is a 57.8% chance to use 5, and (1-57.8%) chance to use 4.
And we can allow both 4 and 5 in the prediction, but the loss will be higher if the prediction is 4.

Multi-task training might help too. I will try this dataset format:
[TxtFirst] [Desc of Img (txt tokens)] [Img] [img tokens]
and sometimes
[ImgFirst] [img tokens] [Txt] [Desc of Img (txt tokens)]
... the order of the imgs should be randomized in the DataLoader, and [TxtFirst] [ImgFirst] [Img] [Txt] are special tokens
and do random sampling of the full dataset. So sometimes the model will see the img tokens first and then the corresponding txt tokens, which is a [img -> txt] task. And the model will see some partial imgs and partial txts. I think a char-level LM might help the model to write correct text on images.

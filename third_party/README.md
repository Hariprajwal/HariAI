# Optional Open-JEV reference

HariAI can call the typed-choice interface from the independent
[Open-JEV reference](https://github.com/kyegomez/open-jev), checked at commit
`93843ef288f3d35fd91263377c40b6f917d77da0`. It is licensed Apache 2.0;
see its own LICENSE and notices. Clone it locally with:

```sh
git clone https://github.com/kyegomez/open-jev.git third_party/open-jev
```

Upstream explicitly reports that its weights are random and does not provide
a trained checkpoint. HariAI therefore requires an independently trained,
compatible checkpoint with `config` (a JevConfig dictionary),
`model_state_dict`, and positive `training_steps` before using this adapter.
Only load trusted checkpoints. A checkpoint's recorded steps do not establish
accuracy or calibration. Its file size is separate from HariNano's weight size.

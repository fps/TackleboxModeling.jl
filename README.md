This software is in part inspired by Jim Lil's excellent "where does the tone come from..." series of videos. Especially his video titled "Tested: Where Does The Tone Come From In A Guitar Amplifier?" which you can watch here:

[https://www.youtube.com/watch?v=wcBEOcPtlYk](https://www.youtube.com/watch?v=wcBEOcPtlYk)

In this video Jim belabors the point that most common amplifiers can be broken down into three tone shaping stages, each possibly followed by a non-linearity:

- Input tone shaping
- Nonlinearity
- Intermediate tone shaping
- Nonlinearity
- Cabinet tone shaping

The neural model then becomes:

- 1D-convolution of size (256) (1 channel)
- dist_aa2
- 1D-convolution of size (512) (1 channel)
- dist_aa2
- 1D-convolution of size (1024) (1 channel)

Here the 1D-convolutions take on the role of the tone shaping stages and the `dist_aa2` activation functions perform the non-linearity/distortion. `dist_aa2` is the non-linearity `x / sqrt(1+x^2)` cast into the "transparent" antiderivative antialiasing form described in "Note on Alias Suppression in Digital Distortion" by Martin Vicanek (2024.) See equation 10 in that paper.

This architecture allows an efficient implementation in a plugin using partitioned convolution (about 1/10 of the processing load compared to standard NAMs). Since the number of non-linearities is quite limited it is possible to implement oversampling for only those relatively efficiently (ca. 50 % cpu load increase over the non-oversampling variant). 

The code is flexible enough to add additional stages which can be useful for higher gain models.

The two main parts of this software are:

- Julia code to train a model. It uses CUDA.jl and cuDNN.jl in tandem with Flux.jl to perform the training on a GPU.
- A simple LV2 plugin that allows the user to select one of the previously trained models. It would be easy to add model parameter loading from an e.g. JSON file but I don't need it. PRs welcome though. The plugin implements optional 2x oversampling (using a Chebyshev type II interpolation and decimation filter.)

# Examples

The models which are included with the plugin have been trained on input/output pairs produced by neural amp modeller (NAM) models. So they are "2nd-generation models" ;) To my ears they sound quite similar on my little test snippet. You can find them in the examples/ folder in the respective model subdirectory (the prefix "nam_" denotes the audio files rendered by NAM. the "test_" prefix denotes the audio files rendered by the tacklebox). You should get playable links below if you visit the github pages version of this repository: [https://fps.github.io/TackleboxModeling.jl](https://fps.github.io/TackleboxModeling.jl).

To reproduce these models you will have to download the corresponding .nam files. These are their md5 hashes:

```bash
e0619b10bd08354d4e1953fc1d0bc104  data/Fender Deluxe Reverb/model.nam
bef7ef275c9a9a8c33cd097004b7549f  data/BrianMay/model.nam
15e51e9fd7742b575fd2d212e7ee6b23  data/marshall bluesbreaker 1962/model.nam
9eb7e7d50ccac9f6765a62499613fcf7  data/EVH 5150/model.nam
```

You will also need to download the `nam_training_input.wav` file from the neural-amp-modeler site. The file might be called differently originally, but the version I used has this hash:

```bash
36cd1af62985c2fac3e654333e36431e  data/nam_training_input.wav
```

After downloading them you'll have to process the `nam_training_input.wav` file to create the corresponding `nam_training_output.wav` file using the `nam_wavenet` example from https://github.com/fps/anna and put it into the corresponding folder below `data`. Then you change the `run_chunks.jl` script to start with the right amount of layers (3 for all of them except the `EVH 5150` model which needs 4.) and run it (I usually do it by `include ("run_chunks")` on the julia prompt. Once you are satisified with the error (below 0.03 is very good, below 0.04 sometimes acceptable) you can `include("write_test_output.jl")` and then `include("write_model_output.jl")`. The former runs the julia version of the trained model on the `data/Take1_Audio\ 1-1_short.wav` example and the latter produces the `model.cc` code which is included into the tacklbox LV2 plugin.

## Fender Deluxe Reverb

<p>
<video controls width="300" height="50">
  <source src="examples/Fender Deluxe Reverb/nam_Take1_Audio 1-1_short.wav" type="audio/mpeg">
  Your browser does not support the video tag.
</video>
NAM
</p>

<p>
<video controls width="300" height="50">
  <source src="examples/Fender Deluxe Reverb/test_Take1_Audio 1-1_short.wav" type="audio/mpeg">
  Your browser does not support the video tag.
</video>
Tacklebox
</p>

## Marshal Bluesbreaker 1962

<p>
<video controls width="300" height="50">
  <source src="examples/marshall bluesbreaker 1962/nam_Take1_Audio 1-1_short.wav" type="audio/mpeg">
  Your browser does not support the video tag.
</video>
NAM
</p>

<p>
<video controls width="300" height="50">
  <source src="examples/marshall bluesbreaker 1962/test_Take1_Audio 1-1_short.wav" type="audio/mpeg">
  Your browser does not support the video tag.
</video>
Tacklebox
</p>

## BrianMay (includes antialiasing demonstration using a sine sweep)

<p>
<video controls width="300" height="50">
  <source src="examples/BrianMay/nam_Take1_Audio 1-1_short.wav" type="audio/mpeg">
  Your browser does not support the video tag.
</video>
NAM
</p>

<p>
<video controls width="300" height="50">
  <source src="examples/BrianMay/test_Take1_Audio 1-1_short.wav" type="audio/mpeg">
  Your browser does not support the video tag.
</video>
Tacklebox
</p>

<p>
<video controls width="300" height="50">
  <source src="examples/BrianMay/oversampled_Take1_Audio 1-1_short.wav" type="audio/mpeg">
  Your browser does not support the video tag.
</video>
Tacklebox, 2x oversampled
</p>

<p>
<video controls width="300" height="50">
  <source src="examples/BrianMay/nam_sweep.wav" type="audio/mpeg">
  Your browser does not support the video tag.
</video>
sweep (NAM)
</p>

<p>
<video controls width="300" height="50">
  <source src="examples/BrianMay/sweep.wav" type="audio/mpeg">
  Your browser does not support the video tag.
</video>
sweep (Tacklebox)
</p>

<p>
<video controls width="300" height="50">
  <source src="examples/BrianMay/oversampled_sweep.wav" type="audio/mpeg">
  Your browser does not support the video tag.
</video>
sweep (Tacklebox, 2x oversampled)
</p>

This code is just a proof of concept.

# Building the plugin

```bash
meson setup build -Dbuildtype=release
meson compile -vC build
```

# Installing the plugin

- Add the `build/plugin/lv2` directory to your `LV2_PATH` or move the `build/plugin/lv2/tacklebox.lv2` directory to a location on your `LV2_PATH`.

# TODOs / Limitations

- Try oversampling for the non-linearity in the plugin and check whether that alters the sound too much (done: it doesn't. Implemented: 2x oversampling in combibation with anti-derivative antialiasing)
- Implement better model selection in the LV2 plugin
- Implement noise-free model switching in the LV2 plugin
- Improve the training code from being a stinking pile of poop to something reusable
- Implement time-distributed partitioned convolution to make the plugin more efficient
- Add audio level calibration info to the models
- Evaluate cheaper to compute nonlinearity (cheaper than tanh) (done: x / sqrt(1 + x^2))
- Experiment with tone shaping controls between stages

# License

This software is free software available under the GPL v2 license. If you require different license terms, feel free to contact me.

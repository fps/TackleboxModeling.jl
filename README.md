This software is in part inspired by Jim Lil's excellent "where does the tone come from..." series of videos. Especially his video titled "Tested: Where Does The Tone Come From In A Guitar Amplifier?" which you can watch here:

https://www.youtube.com/watch?v=wcBEOcPtlYk

In this video Jim belabors the point that most common amplifiers can be broken down into three tone shaping stages, each possibly followed by a non-linearity:

- Input tone shaping
- Nonlinearity
- Intermediate tone shaping
- Nonlinearity
- Cabinet tone shaping

The neural model then becomes:

- 1D-convolution of size (256) (1 channel)
- tanh
- 1D-convolution of size (512) (1 channel)
- tanh
- 1D-convolution of size (1024) (1 channel)

Here the 1D-convolutions take on the role of the tone shaping stages and the tanh activation functions perform the nonlinearity/distortion.

This architecture allows an efficient implementation in a plugin using partitioned convolution (about 1/10 of the processing load compared to standard NAMs). 

The code is flexible enough to add additional stages which can be useful for higher gain models//.

The two main parts of this software are:

- Julia code to train a model. It uses CUDA.jl and cuDNN.jl in tandem with Flux.jl to perform the training on a GPU.
- A simple LV2 plugin that allows the user to select one of the previously trained models. It would be easy to add model parameter loading from an e.g. JSON file but I don't need it. PRs welcome though.

The models which are included with the plugin have been trained on input/output pairs produced by neural amp modeller (NAM) models. So they are "2nd-generation models" ;) To my ears they sound quite similar on my little test snippet. You can find them in the data/ folder in the respective model subdirectory (the prefix "nam_" denotes the audio files rendered by NAM. the "test_" prefix denotes the audio files rendered by the tacklebox).

<video controls width="300" height="50">
  <source src="examples/BrianMay/nam_Take1_Audio 1-1_short.wav" type="audio/mpeg">
  Your browser does not support the video tag.
</video>
[BrianMay (NAM)](examples/BrianMay/nam_Take1_Audio 1-1_short.wav)

This code is just a proof of concept.

# Building the plugin

```bash
meson setup build -Dbuildtype=release
meson compile -vC build
```

# Installing the plugin

- Add the `build/plugin/lv2` directory to your `LV2_PATH` or move the `build/plugin/lv2/tacklebox.lv2` directory to a location on your `LV2_PATH`.

# TODOs / Limitations

- Try oversampling for the non-linearity in the plugin and check whether that alters the sound too much
- Implement better model selection in the LV2 plugin
- Improve the training code from being a stinking pile of poop to something reusable
- Implement time-distributed partitioned convolution to make the plugin more efficient

# License

This software is free software available under the GPL v2 license. If you require different license terms, feel free to contact me.

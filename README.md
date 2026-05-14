This software is in part inspired by Jim Lil's excellent "where does the tone come from..." series of videos. Especially his video titled "Tested: Where Does The Tone Come From In A Guitar Amplifier?" which you can watch here:

https://www.youtube.com/watch?v=wcBEOcPtlYk

The model used in this software is a very simple 1-dimensional convolutional network modeled after the three most common stages in a guitar amplifier:

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

This neural network allows an efficient implementation using partitioned convolution.

The two main parts of this software are:

- Julia code to train a model. It uses CUDA.jl and cuDNN.jl in tandem with Flux.jl to perform the training on a GPU.
- A simple LV2 plugin that allows the user to select one of the previously trained models. It would be easy to add model parameter loading from an e.g. JSON file but I don't need it. PRs welcome though.

# Building the plugin

```bash
meson setup build -Dbuildtype=release
meson compile -vC build
```

# License

This software is free software available under the GPL v2 license. If you require different license terms, feel free to contact me.

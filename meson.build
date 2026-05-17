project('tacklebox', 'cpp')

lv2 = dependency('lv2')
fftw3f = dependency('fftw3f')

add_project_arguments('-DNDEBUG', language: 'cpp')
add_project_arguments('-DAUDIOFFT_FFTW3', language: 'cpp')
add_project_arguments('-march=native', language: 'cpp')

fftconvolver = static_library('fftconvolver', [ './FFTConvolver/AudioFFT.cpp', './FFTConvolver/FFTConvolver.cpp', './FFTConvolver/TwoStageFFTConvolver.cpp', './FFTConvolver/Utilities.cpp' ])

local_includes = include_directories('plugin/include', 'FFTConvolver')

subdir('plugin')

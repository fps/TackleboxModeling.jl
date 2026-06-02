project('tacklebox', 'cpp', version : '0.1', default_options : 'warning_level=1')

lv2 = dependency('lv2')

fftw3f = dependency('fftw3f', static : true, required : get_option('fftw'))
if fftw3f.found()
  add_project_arguments('-DAUDIOFFT_FFTW3', language: 'cpp')
endif

samplerate = dependency('samplerate', static : true, required : get_option('samplerate'))
if samplerate.found()
  add_project_arguments('-DSAMPLERATE_FOUND', language : 'cpp')
endif

add_project_arguments('-DNDEBUG', language: 'cpp')
# add_project_arguments('-D_GLIBCXX_DEBUG', language: 'cpp')
# add_project_arguments('-march=native', language: 'cpp')
# add_project_arguments('-DFFTCONVOLVER_DONT_USE_SSE', language: 'cpp')
# add_project_arguments('-U_GLIBCXX_ASSERTIONS', language: 'cpp')

fftconvolver = static_library('fftconvolver', [ './FFTConvolver/AudioFFT.cpp', './FFTConvolver/FFTConvolver.cpp', './FFTConvolver/TwoStageFFTConvolver.cpp', './FFTConvolver/Utilities.cpp' ])

local_includes = include_directories('plugin/include', 'FFTConvolver', 'cascaded_biquad/include', '.')

subdir('plugin')

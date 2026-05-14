lv2 = dependency('lv2')
fftw3f = dependency('fftw3f')

library('tacklebox', 'tacklebox.cc', '../../FFTConvolver/AudioFFT.cpp', '../../FFTConvolver/FFTConvolver.cpp', '../../FFTConvolver/TwoStageFFTConvolver.cpp', '../../FFTConvolver/Utilities.cpp', include_directories : local_includes, dependencies : [ lv2, fftw3f ])


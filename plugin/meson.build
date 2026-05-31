subdir('lv2/tacklebox.lv2')

executable('test_tacklebox', 'test_tacklebox.cc', dependencies : [ fftw3f, samplerate ], link_with : [ fftconvolver ], include_directories : local_includes)
executable('test_oversample', 'test_oversample.cc', include_directories : local_includes)

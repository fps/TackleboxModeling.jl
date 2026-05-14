library('tacklebox', 'tacklebox.cc', include_directories : local_includes, dependencies : [ lv2, fftw3f ], link_with : [ fftconvolver ])

fs = import('fs')

fs.copyfile('tacklebox.ttl', 'tacklebox.ttl')
fs.copyfile('manifest.ttl', 'manifest.ttl')


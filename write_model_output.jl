include("TackleboxModeling.jl")

open("$(outpath)/model.cc", "w") do f;  print(f, model_to_c(m_min)); end
model_to_bson("$outpath/model.bson", m_min, x_scale, x_mean, y_scale, y_mean)

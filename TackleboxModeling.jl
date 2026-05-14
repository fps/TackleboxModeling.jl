
function model_to_c(m)
  """
  float x_scale = $(big(x_scale));
  float x_mean = $(big(x_mean));

  float y_scale = $(big(y_scale));
  float y_mean = $(big(y_mean));

  float w1[] = { $(join(big.(cpu(m[1].weight)), ", ")) };

  float w2[] = { $(join(big.(cpu(m[2].weight)), ", ")) };

  float w3[] = { $(join(big.(cpu(m[3].weight)), ", ")) };

  float b1 = $(big(cpu(m[1].bias)[1]));
  float b2 = $(big(cpu(m[2].bias)[1]));
  float b3 = $(big(cpu(m[3].bias)[1]));
  """
end

function models_to_c(ms)
  
end

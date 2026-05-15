
function layer_to_c(l)
"""
        {
          { $(join(big.(cpu(l.weight)), ", ")) },
          $(big(cpu(l.bias)[1])),
          $(if l.σ == Flux.tanh; "\"tanh\""; else "\"nothing\""; end),
        },
"""
end

function model_to_c(m)
"""
    {
      {
$(join([layer_to_c(l) for l in m]))
      },

      $(big(x_scale)),
      $(big(x_mean)),
  
      $(big(y_scale)),
      $(big(y_mean)),
    },
"""
end

function models_to_c(ms)
"""
  #include <vector>
  #include <string>

  struct layer
  {
    std::vector<float> weights;
    float bias;
    std::string activation;
  };

  struct model
  {
    std::vector<layer> layers;
    float x_scale;
    float x_mean;
    float y_scale;
    float y_mean;
  };

  std::vector<model> models = 
  {
$(join([model_to_c(m) for m in ms]))
  };
"""
end

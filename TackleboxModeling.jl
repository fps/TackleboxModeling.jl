function activation_to_c(a)
  if a == dist_tanh
    return "\"tanh\","
  end
  if a == dist_aa
    return "\"dist_aa\","
  end
  "\"nothing\","
end

function layer_to_c(l)
"""
        {
          { $(join(big.(cpu(l[1].weight)), ", ")) },
          $(big(cpu(l[1].bias)[1])),
          $(if length(l) > 1; activation_to_c(l[2]); else; "\"nothing\""; end)
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

function model_to_bson(model_path, m, x_scale, x_mean, y_scale, y_mean)
  m_cpu = m |> cpu
  BSON.@save model_path m_cpu x_scale x_mean y_scale y_mean
end

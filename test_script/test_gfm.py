from transformers import GraphormerModel, GraphormerConfig
m = GraphormerModel.from_pretrained("clefourrier/graphormer-base-pcqm4mv2")
print("Max nodes:", m.config.max_nodes)
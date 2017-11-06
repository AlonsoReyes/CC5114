from TareaClase4.Network import Network

t = Network(2)
t.create_layer(2)
t.create_layer(2)
print("Weights and bias by layer")
t.get_net_info()
print("")
t.feed_forward([1, 1])
t.back_propagation([0, 1])
t.update([1, 0])
t.get_info()
print("\nWeights and bias by layer")
t.get_net_info()


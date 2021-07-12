
from tvm.relay.testing import resnet
from tvm.contrib import relay_viz

# We use pre-defined resnet18 network in Relay.
batch_size = 1
num_of_image_class = 1000
image_shape = (3, 224, 224)
output_shape = (batch_size, num_of_image_class)
relay_mod, relay_param = resnet.get_workload(num_layers=18, batch_size=1, image_shape=image_shape)

print("Running RelayVisualizer..")
vizer = relay_viz.RelayVisualizer(relay_mod, relay_param=relay_param)
vizer.render("resnet18.html")
print("output html...")

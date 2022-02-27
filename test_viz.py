
import tvm
from tvm import relay
from tvm.relay import transform
from tvm.contrib import relay_viz
from tvm.relay.op.annotation import compiler_begin, compiler_end
from tvm.relay.testing import resnet

def test_subgraph():

    # Inputs and Weights
    x = relay.var("x", shape=(10, 10))
    w0 = relay.var("w0", shape=(10, 10))
    w1 = relay.var("w1", shape=(10, 10))

    # z0 = x + w0
    x_ = compiler_begin(x, "ccompiler")
    w0_ = compiler_begin(w0, "ccompiler")
    z0_ = relay.add(x_, w0_)
    z0 = compiler_end(z0_, "ccompiler")

    # z1 = z0 + w1
    z0__ = compiler_begin(z0, "ccompiler")
    w1_ = compiler_begin(w1, "ccompiler")
    z1_ = relay.add(z0__, w1_)
    z1 = compiler_end(z1_, "ccompiler")

    # z2 = z0 + z1
    z2 = relay.add(z0, z1)

    f = relay.Function([x, w0, w1], z2)
    mod = tvm.IRModule()
    mod["main"] = f

    # merge compiler regions
    mod = transform.MergeCompilerRegions()(mod)
    mod = transform.PartitionGraph("mod_name")(mod)
    mod = transform.InferType()(mod)
    return mod

mod_with_subgraph = test_subgraph()
mod_wo_subgraph, param = resnet.get_workload(num_layers=18)

graph_attr = {"color": "red"}
node_attr = {"color": "blue"}
edge_attr = {"color": "black"}
# VizNode is passed to the callback.
def get_node_attr(node):
    if "nn.conv2d" in node.type_name and "NCHW" in node.detail:
        return {
            "fillcolor": "green",
            "style": "filled",
            "shape": "box",
        }
    if "Var" in node.type_name:
        return {"shape": "ellipse"}
    return {"shape": "box"}
dot_plotter = relay_viz.DotPlotter(
    graph_attr=graph_attr,
    node_attr=node_attr,
    edge_attr=edge_attr,
    get_node_attr=get_node_attr)

# with subgraph
print("Outputing mod_with_subgraph.pdf ...")
viz = relay_viz.RelayVisualizer(
    mod_with_subgraph,
    plotter=dot_plotter,
    parser=relay_viz.DotVizParser())
viz.render("mod_with_subgraph")

# without subgraph
print("Outputing mod_wo_subgraph.pdf ...")
viz = relay_viz.RelayVisualizer(
    mod_wo_subgraph,
    relay_param=param,
    plotter=dot_plotter,
    parser=relay_viz.DotVizParser())
viz.render("mod_wo_subgraph")

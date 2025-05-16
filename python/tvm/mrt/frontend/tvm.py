TVM_IR = "RELAY"
try:
    from tvm import relay
    from .expr import *
except:
    TVM_IR = "RELAX"

if TVM_IR == "RELAX":
    try:
        from tvm import relax
    except:
        print("TVM relay/relax not supported.")

    from .relax import *

print("USE TVM API:", TVM_IR)


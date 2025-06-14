from rllab.core.serializable import Serializable
import inspect

print("Loaded from:", Serializable.__module__)
print("Defined at:", inspect.getfile(Serializable))
print("Serializable class structure:\n")
print(inspect.getsource(Serializable))
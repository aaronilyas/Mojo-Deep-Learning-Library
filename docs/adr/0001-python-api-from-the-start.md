# Python API from the start, Mojo as the engine

Users write Python; Mojo implements the kernels and runtime underneath, exposed as Python extension modules. We chose this over a Mojo-first API with bindings later because the library is for other people doing local AI development, and nearly all of them (and the coding agents working for them) live in Python. The cost: every public API must be designed to cross the Python/Mojo boundary from day one, and the Python surface is the public contract.

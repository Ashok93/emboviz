"""The adapter subsystem — every backend talks through ZeroMQ workers.

emboviz core has **no model dependencies** (no torch, no transformers,
no lerobot). Each backend (VLA family, perception model, ...) lives in
its own pip-installable adapter package (``emboviz-openvla``,
``emboviz-oft``, ``emboviz-pi0``, ``emboviz-gr00t``, ``emboviz-sam3``)
whose heavy deps install into an isolated venv on first use.

When the CLI asks for ``model.predict(scene)``:

1. The registry resolves a CLI alias (``"openvla"``) →
   :class:`AdapterSpec` declared by the matching adapter package's
   entry point.
2. The lifecycle layer either attaches to a user-started worker on its
   known endpoint, or spawns one in the adapter's runtime venv via
   ``subprocess.Popen`` and waits until it answers ``ping``.
3. The :class:`ZMQAdapterClient` (or, for SAM3 and friends, a sibling
   :class:`RpcClient` subclass shipped by that adapter) wraps the live
   ZMQ DEALER socket in a Python-side facade so callers never know
   they're talking to another process.

Adapter authors don't subclass the protocol — they ship a small
**Service Handler** class whose ``methods`` property enumerates which
wire-method names are exposed and how each maps to model methods. The
server loop dispatches via that dict; unknown method names raise
explicitly. For VLA adapters we provide the ready-made
:class:`VLAModelHandler` that lists every VLAModel protocol method.
"""

from emboviz.adapters.client import (
    AdapterRpcError,
    RpcClient,
    ZMQAdapterClient,
    ZMQReaderClient,
    ZMQWorldModelClient,
    default_endpoint,
)
from emboviz.adapters.lifecycle import (
    WorkerHandle,
    connect,
    connect_reader,
    connect_world_model,
    install_venv,
    shutdown,
    venv_path,
    venv_python,
    venv_root,
)
from emboviz.adapters.protocol import AdapterSpec
from emboviz.adapters.registry import find_adapter, list_adapters
from emboviz.adapters.reader_registry import find_reader, list_readers
from emboviz.adapters.world_model_registry import (
    find_world_model,
    list_world_models,
)
from emboviz.adapters.server_base import (
    DatasetReaderHandler,
    ServiceHandler,
    VLAModelHandler,
    WorldModelHandler,
    serve,
)

__all__ = [
    "AdapterRpcError",
    "AdapterSpec",
    "RpcClient",
    "ServiceHandler",
    "VLAModelHandler",
    "DatasetReaderHandler",
    "WorldModelHandler",
    "WorkerHandle",
    "ZMQAdapterClient",
    "ZMQReaderClient",
    "ZMQWorldModelClient",
    "connect",
    "connect_reader",
    "connect_world_model",
    "default_endpoint",
    "find_adapter",
    "find_reader",
    "find_world_model",
    "install_venv",
    "list_adapters",
    "list_readers",
    "list_world_models",
    "serve",
    "shutdown",
    "venv_path",
    "venv_python",
    "venv_root",
]

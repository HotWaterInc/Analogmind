from src.ai.variants.exploration.pipelines import test_pipeline
from src.modules.visualizations.entry import visualization_3d_target_surface
from functools import wraps
from src.ai.runtime_storages.new.storage_runtime_data import *


def randf(new_node):
    print("randomfn")


if __name__ == "__main__":
    st = StorageRuntimeData()
    print(st)

    st.subscribe_to_crud_operations(
        data_type=TypeAlias.NODE_DATA,
        operation_type=OperationsAlias.CREATE,
        subscriber=randf
    )
    st.create_node(
        name="node_1",
        data=[[1, 2, 3], [4, 5, 6]],
        params={"param_1": 1, "param_2": 2}
    )

    pass

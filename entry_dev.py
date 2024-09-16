from src.ai.runtime_storages.new import storage_module as storage


def my_func(storage, new_ndoej):
    print("ceva", new_ndoej)


if __name__ == "__main__":
    storage_data = storage.create_storage()

    storage.subscribe_to_crud_operation(storage=storage_data, data_alias=storage.DataAlias.NODE_DATA,
                                        operation_type=storage.OperationsAlias.CREATE, subscriber=my_func)
    # storage.create_node(self=storage_data, data=[1, 2, 3], name="node1", params={"param1": 1})
    storage.create_node(storage_data, data=[[1]], name="node1", params={"param1": 1})

    pass

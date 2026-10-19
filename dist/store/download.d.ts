import type { DeleteModelOptions, ModelSpec, ModelStore, PrepareModelOptions } from "../tensor/types";
declare const 
/** Download every file of a model into the store. */
prepareModel: (options: PrepareModelOptions & {
    store: ModelStore;
}) => Promise<ModelSpec>, 
/** Remove every file of a model from the store. */
deleteModel: (options: DeleteModelOptions & {
    store: ModelStore;
}) => Promise<void>;
export { prepareModel, deleteModel };

import type { GgufFile, ParseGgufOptions } from "../tensor/types";
declare const 
/** Parse a GGUF file into metadata and tensor entries. */
parseGguf: (options: ParseGgufOptions) => GgufFile;
export { parseGguf };

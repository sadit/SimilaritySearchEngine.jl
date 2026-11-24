/**
 * `similarity-search-client`: a client of the HTTP API of `similarity-search-serve`, the server
 * of SimilaritySearchEngine.jl (PLAN.md §5.17).
 *
 * ```ts
 * import { SimilaritySearchClient, NotFound } from "similarity-search-client";
 * ```
 *
 * @module
 */

export { SimilaritySearchClient } from "./client.js";
export type { ClientOptions } from "./client.js";
export * from "./errors.js";
export type * from "./types.js";
